"""
    TransformerHawkesCell(embed_dim::Integer, num_heads::Integer; init_scale=0.02f0)

Causal multi-head attention cell with a learnable continuous Hawkes time-decay mask.

Inputs:
- `x` with shape `(embed_dim, T, B)`
- `times` with shape `(T, B)`

Output:
- tensor with shape `(embed_dim, T, B)`

This implementation uses batched matrix multiplication instead of constructing
each attention head with nested `map`/`hcat`/`vcat` calls.
"""
struct TransformerHawkesCell{T<:AbstractFloat} <: Lux.AbstractLuxLayer
    embed_dim::Int
    num_heads::Int
    head_dim::Int
    init_scale::T

    function TransformerHawkesCell(
        embed_dim::Integer,
        num_heads::Integer;
        init_scale::T=0.02f0,
    ) where {T<:AbstractFloat}

        embed_dim > 0 || throw(ArgumentError("embed_dim must be positive"))
        num_heads > 0 || throw(ArgumentError("num_heads must be positive"))
        embed_dim % num_heads == 0 ||
            throw(ArgumentError("embed_dim must be divisible by num_heads"))

        return new{T}(Int(embed_dim), Int(num_heads), Int(embed_dim ÷ num_heads), init_scale)
    end
end

function Lux.initialparameters(rng::AbstractRNG, layer::TransformerHawkesCell{T}) where {T}
    E = layer.embed_dim
    H = layer.num_heads

    return (
        Wq = layer.init_scale .* randn(rng, T, E, E),
        Wk = layer.init_scale .* randn(rng, T, E, E),
        Wv = layer.init_scale .* randn(rng, T, E, E),
        Wo = layer.init_scale .* randn(rng, T, E, E),

        bq = zeros(T, E),
        bk = zeros(T, E),
        bv = zeros(T, E),
        bo = zeros(T, E),

        decay = layer.init_scale .* randn(rng, T, H),
    )
end

Lux.initialstates(::AbstractRNG, ::TransformerHawkesCell) = NamedTuple()

function _linear_project(W, b, x::AbstractArray{<:AbstractFloat, 3})
    E, T_seq, B = size(x)

    x2 = reshape(x, E, T_seq * B)
    b2 = reshape(b, E, 1)

    y2 = W * x2 .+ b2

    return reshape(y2, E, T_seq, B)
end

function _split_heads(z::AbstractArray{<:AbstractFloat, 3}, D::Integer, H::Integer)
    E, T_seq, B = size(z)

    E == D * H ||
        throw(DimensionMismatch("embed_dim must equal head_dim * num_heads"))

    # Input:  (E, T, B)
    # Output: (D, T, H, B)
    return permutedims(reshape(z, D, H, T_seq, B), (1, 3, 2, 4))
end

function _merge_heads(z::AbstractArray{<:AbstractFloat, 4})
    # Input:  (D, T, H, B)
    # Output: (D * H, T, B)
    D, T_seq, H, B = size(z)

    return reshape(permutedims(z, (1, 3, 2, 4)), D * H, T_seq, B)
end

function _causal_mask(T_seq::Integer, ::Type{S}) where {S<:AbstractFloat}
    neg_inf = -S(Inf)

    # rows = source index s
    # cols = target/query index t
    return [
        s <= t ? zero(S) : neg_inf
        for s in 1:T_seq, t in 1:T_seq
    ]
end

function _time_deltas(times::AbstractMatrix{<:AbstractFloat}, ::Type{S}) where {S<:AbstractFloat}
    T_seq, B = size(times)

    # Δ[s, t, b] = max(times[t, b] - times[s, b], 0)
    return [
        max(S(times[t, b] - times[s, b]), zero(S))
        for s in 1:T_seq, t in 1:T_seq, b in 1:B
    ]
end

function (layer::TransformerHawkesCell)(
    x::AbstractArray{<:AbstractFloat, 3},
    times::AbstractMatrix{<:AbstractFloat},
    ps,
    st::NamedTuple,
)
    E, T_seq, B = size(x)

    E == layer.embed_dim ||
        throw(DimensionMismatch("x first dimension must equal embed_dim"))

    size(times) == (T_seq, B) ||
        throw(DimensionMismatch("times must have shape (T, B) matching x"))

    H = layer.num_heads
    D = layer.head_dim

    S = promote_type(eltype(x), eltype(times), eltype(ps.Wq))

    q = _linear_project(ps.Wq, ps.bq, x)
    k = _linear_project(ps.Wk, ps.bk, x)
    v = _linear_project(ps.Wv, ps.bv, x)

    # (E, T, B) -> (D, T, H, B)
    qh = _split_heads(q, D, H)
    kh = _split_heads(k, D, H)
    vh = _split_heads(v, D, H)

    # Collapse heads and batch into one batched dimension:
    # (D, T, H, B) -> (D, T, H * B)
    qn = reshape(qh, D, T_seq, H * B)
    kn = reshape(kh, D, T_seq, H * B)
    vn = reshape(vh, D, T_seq, H * B)

    # Content attention:
    # KᵀQ gives (source_time, target_time, head_batch)
    kt = permutedims(kn, (2, 1, 3))
    content_scores = NNlib.batched_mul(kt, qn) .* inv(sqrt(S(D)))

    # Hawkes decay mask:
    # Δ has shape (source_time, target_time, batch)
    Δ = _time_deltas(times, S)

    # Expand to (source_time, target_time, head, batch)
    Δ4 = reshape(Δ, T_seq, T_seq, 1, B)
    decay4 = reshape(abs.(S.(ps.decay)), 1, 1, H, 1)

    hawkes_mask4 = exp.(-decay4 .* Δ4)

    # Collapse to match content_scores:
    # (T, T, H, B) -> (T, T, H * B)
    hawkes_mask = reshape(hawkes_mask4, T_seq, T_seq, H * B)

    # Causal mask:
    causal = reshape(_causal_mask(T_seq, S), T_seq, T_seq, 1)

    # Final attention scores:
    # content is modulated by Hawkes decay, then future positions are masked.
    scores = content_scores .* hawkes_mask .+ causal

    # Softmax across source positions for each target position.
    weights = NNlib.softmax(scores; dims=1)

    # Weighted value aggregation:
    # V * attention_weights -> (D, T, H * B)
    yn = NNlib.batched_mul(vn, weights)

    # Restore shape:
    # (D, T, H * B) -> (D, T, H, B) -> (E, T, B)
    yh = reshape(yn, D, T_seq, H, B)
    y_heads = _merge_heads(yh)

    y = _linear_project(ps.Wo, ps.bo, y_heads)

    return y, st
end
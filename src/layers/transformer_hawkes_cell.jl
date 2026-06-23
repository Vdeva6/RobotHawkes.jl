"""
    TransformerHawkesCell(embed_dim::Integer, num_heads::Integer; init_scale=0.02f0)

Causal multi-head attention cell with a learnable continuous Hawkes time-decay bias.

Inputs:
- `x` with shape `(embed_dim, T, B)`
- `times` with shape `(T, B)`

Output:
- tensor with shape `(embed_dim, T, B)`

This implementation uses batched matrix multiplication for attention.

Scientific note:
This cell uses a log-domain Hawkes attention bias:

    score(s, t, h, b) =
        dot(k[s], q[t]) / sqrt(head_dim)
        - abs(decay[h]) * max(times[t, b] - times[s, b], 0)
        + causal_mask(s, t)

Because softmax exponentiates logits internally, the additive formulation makes
the attention probability proportional to:

    exp(content_score) * exp(-decay * Δt)

This file intentionally isolates the core attention primitive in
`_hawkes_attention(qh, kh, vh, times, decay)`. That function is the future
target for a custom ChainRulesCore adjoint or an Enzyme-compatible kernel.
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

function _validate_attention_inputs(qh, kh, vh, times, decay)
    size(qh) == size(kh) ||
        throw(DimensionMismatch("qh and kh must have matching shape"))

    size(qh) == size(vh) ||
        throw(DimensionMismatch("qh and vh must have matching shape"))

    _, T_seq, H, B = size(qh)

    size(times) == (T_seq, B) ||
        throw(DimensionMismatch("times must have shape (T, B) matching qh/kh/vh"))

    length(decay) == H ||
        throw(DimensionMismatch("decay must have length equal to num_heads"))

    return nothing
end

function _content_attention_scores(qh, kh)
    D, T_seq, H, B = size(qh)

    qn = reshape(qh, D, T_seq, H * B)
    kn = reshape(kh, D, T_seq, H * B)

    # KᵀQ gives (source_time, target_time, head_batch)
    kt = permutedims(kn, (2, 1, 3))

    S = promote_type(eltype(qh), eltype(kh))
    scale = inv(sqrt(S(D)))

    return NNlib.batched_mul(kt, qn) .* scale
end

function _hawkes_log_bias(times, decay, H::Integer, ::Type{S}) where {S<:AbstractFloat}
    T_seq, B = size(times)

    Δ = _time_deltas(times, S)

    # Expand to (source_time, target_time, head, batch)
    Δ4 = reshape(Δ, T_seq, T_seq, 1, B)
    decay4 = reshape(abs.(S.(decay)), 1, 1, H, 1)

    # Collapse to (source_time, target_time, head * batch)
    return reshape(decay4 .* Δ4, T_seq, T_seq, H * B)
end

function _causal_log_bias(T_seq::Integer, ::Type{S}) where {S<:AbstractFloat}
    return reshape(_causal_mask(T_seq, S), T_seq, T_seq, 1)
end

function _apply_attention_weights(vh, weights)
    D, T_seq, H, B = size(vh)

    vn = reshape(vh, D, T_seq, H * B)

    # V * attention_weights -> (D, T, H * B)
    yn = NNlib.batched_mul(vn, weights)

    # Restore to per-head representation.
    return reshape(yn, D, T_seq, H, B)
end

function _hawkes_attention(qh, kh, vh, times, decay)
    _validate_attention_inputs(qh, kh, vh, times, decay)

    D, T_seq, H, B = size(qh)
    S = promote_type(eltype(qh), eltype(kh), eltype(vh), eltype(times), eltype(decay))

    content_scores = _content_attention_scores(qh, kh)
    hawkes_bias = _hawkes_log_bias(times, decay, H, S)
    causal_bias = _causal_log_bias(T_seq, S)

    # Final attention logits:
    #
    #   logits = content - decay * Δt + causal_mask
    #
    # Softmax is applied over source positions.
    scores = content_scores .- hawkes_bias .+ causal_bias
    weights = NNlib.softmax(scores; dims=1)

    return _apply_attention_weights(vh, weights)
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

    q = _linear_project(ps.Wq, ps.bq, x)
    k = _linear_project(ps.Wk, ps.bk, x)
    v = _linear_project(ps.Wv, ps.bv, x)

    qh = _split_heads(q, D, H)
    kh = _split_heads(k, D, H)
    vh = _split_heads(v, D, H)

    yh = _hawkes_attention(qh, kh, vh, times, ps.decay)

    y_heads = _merge_heads(yh)

    y = _linear_project(ps.Wo, ps.bo, y_heads)

    return y, st
end
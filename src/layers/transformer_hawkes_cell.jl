"""
    TransformerHawkesCell(embed_dim::Integer, num_heads::Integer; init_scale=0.02f0)

Causal multi-head attention cell with a learnable continuous Hawkes time-decay mask.

Inputs:
- `x` with shape `(embed_dim, T, B)`
- `times` with shape `(T, B)`

Output:
- tensor with shape `(embed_dim, T, B)`

This implementation avoids array mutation so that Zygote can differentiate
through the attention computation.
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

function _hawkes_head_attention(
    layer::TransformerHawkesCell,
    q,
    k,
    v,
    times,
    decay_h,
    b::Integer,
    h::Integer,
)
    D = layer.head_dim
    offset = (h - 1) * D
    T_seq = size(q, 2)

    S = promote_type(eltype(q), eltype(times), eltype(decay_h))
    scale = inv(sqrt(S(D)))
    decay_abs = abs(S(decay_h))

    cols = map(1:T_seq) do t
        scores = map(1:t) do s
            dot_qk = sum(
                S(q[offset + d, t, b]) * S(k[offset + d, s, b])
                for d in 1:D
            )

            Δ = max(S(times[t, b] - times[s, b]), zero(S))
            hawkes_mask = exp(-decay_abs * Δ)

            dot_qk * scale * hawkes_mask
        end

        max_score = maximum(scores)
        weights_unnorm = exp.(scores .- max_score)
        weights = weights_unnorm ./ sum(weights_unnorm)

        [
            sum(
                weights[s] * S(v[offset + d, s, b])
                for s in 1:t
            )
            for d in 1:D
        ]
    end

    return hcat(cols...)
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

    q = _linear_project(ps.Wq, ps.bq, x)
    k = _linear_project(ps.Wk, ps.bk, x)
    v = _linear_project(ps.Wv, ps.bv, x)

    batches = map(1:B) do b
        heads = map(1:H) do h
            _hawkes_head_attention(layer, q, k, v, times, ps.decay[h], b, h)
        end

        reduce(vcat, heads)
    end

    y_heads = cat(batches...; dims=3)

    y = _linear_project(ps.Wo, ps.bo, y_heads)

    return y, st
end
"""
    TransformerHawkesCell(embed_dim::Integer, num_heads::Integer; init_scale=0.02f0)

Causal multi-head attention cell with a learnable continuous Hawkes time-decay mask.

Inputs:
- `x` with shape `(embed_dim, T, B)`
- `times` with shape `(T, B)`

Output:
- tensor with shape `(embed_dim, T, B)`

For query time `t` attending to source time `s`, where `s ≤ t`:

    score(t, s, h) =
        dot(q[t, h], k[s, h]) / sqrt(head_dim) *
        exp(-abs(decay[h]) * (times[t] - times[s]))

This is a first correct, type-stable implementation. It prioritizes clarity.
Later phases will optimize allocations and explore fused kernels.
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

    S = promote_type(eltype(x), eltype(times), eltype(ps.Wq))

    y_heads = similar(x, S, E, T_seq, B)

    scale = inv(sqrt(S(D)))

    @inbounds for b in 1:B
        for h in 1:H
            offset = (h - 1) * D
            decay_h = abs(S(ps.decay[h]))

            for t in 1:T_seq
                scores = Vector{S}(undef, t)

                for s in 1:t
                    dot_qk = zero(S)

                    for d in 1:D
                        idx = offset + d
                        dot_qk += S(q[idx, t, b]) * S(k[idx, s, b])
                    end

                    Δ = max(S(times[t, b] - times[s, b]), zero(S))
                    hawkes_mask = exp(-decay_h * Δ)

                    scores[s] = dot_qk * scale * hawkes_mask
                end

                max_score = maximum(scores)

                denom = zero(S)
                for s in 1:t
                    scores[s] = exp(scores[s] - max_score)
                    denom += scores[s]
                end

                for d in 1:D
                    idx = offset + d
                    acc = zero(S)

                    for s in 1:t
                        α = scores[s] / denom
                        acc += α * S(v[idx, s, b])
                    end

                    y_heads[idx, t, b] = acc
                end
            end
        end
    end

    y = _linear_project(ps.Wo, ps.bo, y_heads)

    return y, st
end
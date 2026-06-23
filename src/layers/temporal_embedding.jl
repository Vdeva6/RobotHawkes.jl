"""
    TemporalEmbedding(embed_dim::Integer; init_scale=0.02f0)

Learnable continuous-time embedding layer for asynchronous event sequences.

Input:
- `Δt` with shape `(T, B)` where:
  - `T` = sequence length
  - `B` = batch size

Output:
- tensor with shape `(embed_dim, T, B)`

The embedding uses learnable sinusoidal frequencies, phases, and amplitudes:

    sin(ωᵢ Δt + ϕᵢ), cos(ωᵢ Δt + ϕᵢ)

This is more appropriate than fixed positional embeddings for Hawkes processes
because event times are continuous and irregularly spaced.
"""
struct TemporalEmbedding{T<:AbstractFloat} <: Lux.AbstractLuxLayer
    embed_dim::Int
    init_scale::T

    function TemporalEmbedding(embed_dim::Integer; init_scale::T=0.02f0) where {T<:AbstractFloat}
        embed_dim > 0 || throw(ArgumentError("embed_dim must be positive"))
        iseven(embed_dim) || throw(ArgumentError("embed_dim must be even for sin/cos pairs"))
        return new{T}(Int(embed_dim), init_scale)
    end
end

function Lux.initialparameters(rng::AbstractRNG, layer::TemporalEmbedding{T}) where {T}
    half_dim = layer.embed_dim ÷ 2

    return (
        logfreq = layer.init_scale .* randn(rng, T, half_dim),
        phase   = layer.init_scale .* randn(rng, T, half_dim),
        scale   = ones(T, half_dim),
    )
end

Lux.initialstates(::AbstractRNG, ::TemporalEmbedding) = NamedTuple()

function (layer::TemporalEmbedding)(Δt::AbstractMatrix, ps, st::NamedTuple)
    half_dim = layer.embed_dim ÷ 2

    size(ps.logfreq, 1) == half_dim || throw(DimensionMismatch("logfreq has wrong length"))
    size(ps.phase, 1) == half_dim || throw(DimensionMismatch("phase has wrong length"))
    size(ps.scale, 1) == half_dim || throw(DimensionMismatch("scale has wrong length"))

    # Δt:        (T, B)
    # reshape:   (1, T, B)
    # params:    (H, 1, 1)
    # output:    (2H, T, B) = (embed_dim, T, B)
    Δt3 = reshape(Δt, 1, size(Δt, 1), size(Δt, 2))

    freq = reshape(exp.(ps.logfreq), half_dim, 1, 1)
    phase = reshape(ps.phase, half_dim, 1, 1)
    scale = reshape(ps.scale, half_dim, 1, 1)

    angles = freq .* Δt3 .+ phase

    y_sin = scale .* sin.(angles)
    y_cos = scale .* cos.(angles)

    y = vcat(y_sin, y_cos)

    return y, st
end
"""
    EventEmbedding(num_events::Integer, embed_dim::Integer; init_scale=0.02f0)

Learnable embedding table for discrete event types.

Input:
- `event_ids` with shape `(T, B)`

Output:
- tensor with shape `(embed_dim, T, B)`
"""
struct EventEmbedding{T<:AbstractFloat} <: Lux.AbstractLuxLayer
    num_events::Int
    embed_dim::Int
    init_scale::T

    function EventEmbedding(
        num_events::Integer,
        embed_dim::Integer;
        init_scale::T=0.02f0,
    ) where {T<:AbstractFloat}
        num_events > 0 || throw(ArgumentError("num_events must be positive"))
        embed_dim > 0 || throw(ArgumentError("embed_dim must be positive"))

        return new{T}(Int(num_events), Int(embed_dim), init_scale)
    end
end

function Lux.initialparameters(rng::AbstractRNG, layer::EventEmbedding{T}) where {T}
    return (
        table = layer.init_scale .* randn(rng, T, layer.embed_dim, layer.num_events),
    )
end

Lux.initialstates(::AbstractRNG, ::EventEmbedding) = NamedTuple()

function (layer::EventEmbedding)(event_ids::AbstractMatrix{<:Integer}, ps, st::NamedTuple)
    T_seq, B = size(event_ids)

    size(ps.table, 1) == layer.embed_dim ||
        throw(DimensionMismatch("embedding table has wrong embed_dim"))

    size(ps.table, 2) == layer.num_events ||
        throw(DimensionMismatch("embedding table has wrong num_events"))

    minimum(event_ids) >= 1 ||
        throw(ArgumentError("event ids must be >= 1"))

    maximum(event_ids) <= layer.num_events ||
        throw(ArgumentError("event ids must be <= $(layer.num_events)"))

    # ps.table[:, vec(event_ids)] gives shape (embed_dim, T * B)
    # reshape restores our standard layout: (embed_dim, T, B)
    y = reshape(ps.table[:, vec(event_ids)], layer.embed_dim, T_seq, B)

    return y, st
end
"""
    EventEmbedding(num_events::Integer, embed_dim::Integer; init_scale=0.02f0)

Learnable embedding table for discrete event types.

Input:
- `event_ids` with shape `(T, B)` where:
  - `T` = sequence length
  - `B` = batch size
  - each value is an integer event id in `1:num_events`

Output:
- tensor with shape `(embed_dim, T, B)`

This layer represents the discrete identity of each event, while
`TemporalEmbedding` represents the continuous time gap between events.
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

    y = similar(ps.table, layer.embed_dim, T_seq, B)

    @inbounds for b in axes(event_ids, 2)
        for t in axes(event_ids, 1)
            event_id = event_ids[t, b]

            1 <= event_id <= layer.num_events ||
                throw(ArgumentError("event id $event_id is outside 1:$(layer.num_events)"))

            for d in axes(ps.table, 1)
                y[d, t, b] = ps.table[d, event_id]
            end
        end
    end

    return y, st
end
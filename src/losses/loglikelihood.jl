"""
    observed_loglikelihood(λ, event_ids)

Compute the observed-event log-likelihood term:

    Σ log λ[event_ids[t, b], t, b]

Inputs:
- `λ` with shape `(num_events, T, B)`
- `event_ids` with shape `(T, B)`

Returns:
- scalar log-likelihood

This is only the event-observation term. The continuous integral term
will be added separately.
"""
function observed_loglikelihood(
    λ::AbstractArray{<:AbstractFloat, 3},
    event_ids::AbstractMatrix{<:Integer},
)
    K, T_seq, B = size(λ)

    size(event_ids) == (T_seq, B) ||
        throw(DimensionMismatch("event_ids must have shape (T, B) matching λ"))

    S = eltype(λ)
    ll = zero(S)

    @inbounds for b in 1:B
        for t in 1:T_seq
            k = event_ids[t, b]

            1 <= k <= K ||
                throw(ArgumentError("event id $k is outside 1:$K"))

            ll += log(λ[k, t, b])
        end
    end

    return ll
end

"""
    observed_nll(λ, event_ids; normalize=true)

Compute the negative observed-event log-likelihood.

If `normalize=true`, divide by the number of observed events.
"""
function observed_nll(
    λ::AbstractArray{<:AbstractFloat, 3},
    event_ids::AbstractMatrix{<:Integer};
    normalize::Bool=true,
)
    nll = -observed_loglikelihood(λ, event_ids)

    if normalize
        return nll / length(event_ids)
    else
        return nll
    end
end

"""
    model_observed_nll(model, event_ids, Δt, ps, st; normalize=true)

Run the model and compute the observed-event negative log-likelihood.

Returns:
- `loss`
- updated state
"""
function model_observed_nll(
    model::TransformerHawkesModel,
    event_ids::AbstractMatrix{<:Integer},
    Δt::AbstractMatrix{<:AbstractFloat},
    ps,
    st::NamedTuple;
    normalize::Bool=true,
)
    λ, st_new = model(event_ids, Δt, ps, st)
    loss = observed_nll(λ, event_ids; normalize=normalize)

    return loss, st_new
end
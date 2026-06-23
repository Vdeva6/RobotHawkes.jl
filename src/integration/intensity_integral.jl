"""
    total_intensity_integral(λ, Δt; normalize=true)

Compute the piecewise-constant total intensity integral:

    Σ_b Σ_t Δt[t, b] * Σ_k λ[k, t, b]

Inputs:
- `λ` with shape `(num_events, T, B)`
- `Δt` with shape `(T, B)`

Returns:
- scalar integral estimate

This is the sequence-discretized Hawkes compensator term. It assumes
the model intensity at each event position is constant over the preceding
time interval `Δt[t, b]`.

If `normalize=true`, divide by the number of observed events.
"""
function total_intensity_integral(
    λ::AbstractArray{<:AbstractFloat, 3},
    Δt::AbstractMatrix{<:AbstractFloat};
    normalize::Bool=true,
)
    K, T_seq, B = size(λ)

    size(Δt) == (T_seq, B) ||
        throw(DimensionMismatch("Δt must have shape (T, B) matching λ"))

    S = promote_type(eltype(λ), eltype(Δt))

    total = zero(S)

    @inbounds for b in 1:B
        for t in 1:T_seq
            λ_sum = zero(S)

            for k in 1:K
                λ_sum += S(λ[k, t, b])
            end

            total += S(Δt[t, b]) * λ_sum
        end
    end

    if normalize
        return total / length(Δt)
    else
        return total
    end
end

"""
    full_hawkes_nll(λ, event_ids, Δt; normalize=true)

Compute the sequence-discretized Hawkes negative log-likelihood:

    -Σ log λ_observed + Σ Δt * Σₖ λₖ

If `normalize=true`, both terms are normalized by number of observed events.
"""
function full_hawkes_nll(
    λ::AbstractArray{<:AbstractFloat, 3},
    event_ids::AbstractMatrix{<:Integer},
    Δt::AbstractMatrix{<:AbstractFloat};
    normalize::Bool=true,
)
    event_term = observed_nll(λ, event_ids; normalize=normalize)
    integral_term = total_intensity_integral(λ, Δt; normalize=normalize)

    return event_term + integral_term
end

"""
    model_full_hawkes_nll(model, event_ids, Δt, ps, st; normalize=true)

Run the model and compute the full sequence-discretized Hawkes NLL.

Returns:
- `loss`
- updated state
"""
function model_full_hawkes_nll(
    model::TransformerHawkesModel,
    event_ids::AbstractMatrix{<:Integer},
    Δt::AbstractMatrix{<:AbstractFloat},
    ps,
    st::NamedTuple;
    normalize::Bool=true,
)
    λ, st_new = model(event_ids, Δt, ps, st)
    loss = full_hawkes_nll(λ, event_ids, Δt; normalize=normalize)

    return loss, st_new
end

"""
    quadgk_integral(f, a, b; reltol=1e-6, abstol=1e-6)

Small convenience wrapper around Integrals.jl + QuadGKJL for 1D scalar integration.

Example:

    quadgk_integral(t -> t^2, 0.0, 1.0)

This prepares the package for later continuous-time intensity interpolation.
"""
function quadgk_integral(
    f,
    a::Real,
    b::Real;
    reltol::Real=1e-6,
    abstol::Real=1e-6,
)
    domain = (float(a), float(b))
    prob = IntegralProblem((u, p) -> f(u), domain)
    sol = solve(prob, QuadGKJL(); reltol=reltol, abstol=abstol)

    return sol.u
end
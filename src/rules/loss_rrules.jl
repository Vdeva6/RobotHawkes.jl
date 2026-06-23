"""
    _observed_nll_lambda_bar(λ, event_ids, ybar; normalize)

Analytic gradient of `observed_nll` with respect to `λ`.

For each observed event `(t, b)` with event type `k = event_ids[t, b]`:

    d/dλ[k,t,b] [-log(λ[k,t,b])] = -1 / λ[k,t,b]

All unobserved event types receive zero gradient.
"""
function _observed_nll_lambda_bar(λ, event_ids, ybar; normalize::Bool)
    K, T_seq, B = size(λ)

    size(event_ids) == (T_seq, B) ||
        throw(DimensionMismatch("event_ids must have shape (T, B)"))

    S = promote_type(eltype(λ), typeof(ybar))
    λbar = zeros(S, size(λ))

    norm = normalize ? S(T_seq * B) : one(S)
    scale = S(ybar) / norm

    @inbounds for b in 1:B
        for t in 1:T_seq
            k = event_ids[t, b]

            1 <= k <= K ||
                throw(ArgumentError("event id $k at ($t, $b) is outside 1:$K"))

            λbar[k, t, b] += -scale / S(λ[k, t, b])
        end
    end

    return λbar
end

"""
    _integral_lambda_bar(λ, Δt, ybar; normalize)

Analytic gradient of `total_intensity_integral` with respect to `λ`.

Current compensator approximation:

    integral = sum_k,t,b λ[k,t,b] * Δt[t,b]

Therefore:

    d integral / d λ[k,t,b] = Δt[t,b]
"""
function _integral_lambda_bar(λ, Δt, ybar; normalize::Bool)
    K, T_seq, B = size(λ)

    size(Δt) == (T_seq, B) ||
        throw(DimensionMismatch("Δt must have shape (T, B)"))

    S = promote_type(eltype(λ), eltype(Δt), typeof(ybar))
    λbar = zeros(S, size(λ))

    norm = normalize ? S(T_seq * B) : one(S)
    scale = S(ybar) / norm

    @inbounds for b in 1:B
        for t in 1:T_seq
            dt = S(Δt[t, b])

            for k in 1:K
                λbar[k, t, b] = scale * dt
            end
        end
    end

    return λbar
end

function ChainRulesCore.rrule(
    ::typeof(observed_nll),
    λ,
    event_ids;
    normalize::Bool = true,
)
    y = observed_nll(λ, event_ids; normalize = normalize)

    project_λ = ChainRulesCore.ProjectTo(λ)

    function observed_nll_pullback(ybar)
        ybar_unthunked = ChainRulesCore.unthunk(ybar)

        λbar = _observed_nll_lambda_bar(
            λ,
            event_ids,
            ybar_unthunked;
            normalize = normalize,
        )

        return (
            ChainRulesCore.NoTangent(),  # function object
            project_λ(λbar),
            ChainRulesCore.NoTangent(),  # event_ids are discrete observed data
        )
    end

    return y, observed_nll_pullback
end

function ChainRulesCore.rrule(
    ::typeof(total_intensity_integral),
    λ,
    Δt;
    normalize::Bool = true,
)
    y = total_intensity_integral(λ, Δt; normalize = normalize)

    project_λ = ChainRulesCore.ProjectTo(λ)

    function total_intensity_integral_pullback(ybar)
        ybar_unthunked = ChainRulesCore.unthunk(ybar)

        λbar = _integral_lambda_bar(
            λ,
            Δt,
            ybar_unthunked;
            normalize = normalize,
        )

        return (
            ChainRulesCore.NoTangent(),  # function object
            project_λ(λbar),
            ChainRulesCore.NoTangent(),  # Δt treated as observed data
        )
    end

    return y, total_intensity_integral_pullback
end

function ChainRulesCore.rrule(
    ::typeof(full_hawkes_nll),
    λ,
    event_ids,
    Δt;
    normalize::Bool = true,
)
    y = full_hawkes_nll(λ, event_ids, Δt; normalize = normalize)

    project_λ = ChainRulesCore.ProjectTo(λ)

    function full_hawkes_nll_pullback(ybar)
        ybar_unthunked = ChainRulesCore.unthunk(ybar)

        observed_bar = _observed_nll_lambda_bar(
            λ,
            event_ids,
            ybar_unthunked;
            normalize = normalize,
        )

        integral_bar = _integral_lambda_bar(
            λ,
            Δt,
            ybar_unthunked;
            normalize = normalize,
        )

        λbar = observed_bar .+ integral_bar

        return (
            ChainRulesCore.NoTangent(),  # function object
            project_λ(λbar),
            ChainRulesCore.NoTangent(),  # event_ids are discrete observed data
            ChainRulesCore.NoTangent(),  # Δt treated as observed data
        )
    end

    return y, full_hawkes_nll_pullback
end
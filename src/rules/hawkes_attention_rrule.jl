# Custom reverse-mode rule scaffold for `_hawkes_attention`.
# See `docs/hawkes_attention_backward.md` for the full backward derivation.
#
#
# This file is intentionally disabled for now.
#
# Why this exists:
# - Zygote can differentiate `_hawkes_attention`, but medium-scale gradients are
#   extremely memory-heavy.
# - The long-term production path is a custom `rrule` for:
#
#       _hawkes_attention(qh, kh, vh, times, decay)
#
# - Before enabling that rule, we need:
#   1. scalar reference output tests
#   2. finite-difference gradient checks
#   3. a manually derived backward pass for attention + softmax + Hawkes bias
#
# Do NOT uncomment this until the custom pullback is fully implemented and tested.


"""
    _hawkes_attention_backward(qh, kh, vh, times, decay, ybar)

Manual reverse pass for `_hawkes_attention`.

Inputs:
- `qh`, `kh`, `vh`: `(D, T, H, B)`
- `times`: `(T, B)`
- `decay`: `(H,)`
- `ybar`: upstream sensitivity with shape `(D, T, H, B)`

Returns:
- `qbar`
- `kbar`
- `vbar`
- `decaybar`

This helper is not yet registered as a ChainRulesCore `rrule`.
It exists so we can validate the backward math against finite differences
before overriding Zygote's default reverse pass.
"""
function _hawkes_attention_backward(qh, kh, vh, times, decay, ybar)
    _validate_attention_inputs(qh, kh, vh, times, decay)

    size(ybar) == size(qh) ||
        throw(DimensionMismatch("ybar must have the same shape as qh/kh/vh"))

    D, T_seq, H, B = size(qh)

    S = promote_type(
        eltype(qh),
        eltype(kh),
        eltype(vh),
        eltype(times),
        eltype(decay),
        eltype(ybar),
    )

    qbar = zeros(S, size(qh))
    kbar = zeros(S, size(kh))
    vbar = zeros(S, size(vh))
    decaybar = zeros(S, size(decay))

    scale = inv(sqrt(S(D)))

    @inbounds for b in 1:B
        for h in 1:H
            decay_h = S(decay[h])
            abs_decay_h = abs(decay_h)
            decay_sign = sign(decay_h)

            # Forward recomputation for this (head, batch) block.
            #
            # W[s, t] is the attention weight from source s to target t.
            # Only s <= t is valid because of causal masking.
            W = zeros(S, T_seq, T_seq)

            for t in 1:T_seq
                scores = Vector{S}(undef, t)

                for s in 1:t
                    dot_qk = zero(S)

                    for d in 1:D
                        dot_qk += S(kh[d, s, h, b]) * S(qh[d, t, h, b])
                    end

                    Δ = max(S(times[t, b] - times[s, b]), zero(S))

                    scores[s] = dot_qk * scale - abs_decay_h * Δ
                end

                max_score = maximum(scores)

                denom = zero(S)
                for s in 1:t
                    scores[s] = exp(scores[s] - max_score)
                    denom += scores[s]
                end

                for s in 1:t
                    W[s, t] = scores[s] / denom
                end
            end

            # Backprop through:
            #
            #     Y = V * W
            #
            # Reverse:
            #
            #     Vbar = Ybar * W'
            #     Wbar = V' * Ybar

            Wbar = zeros(S, T_seq, T_seq)

            for d in 1:D
                for s in 1:T_seq
                    acc_v = zero(S)

                    for t in 1:T_seq
                        acc_v += S(ybar[d, t, h, b]) * W[s, t]
                    end

                    vbar[d, s, h, b] += acc_v
                end
            end

            for s in 1:T_seq
                for t in 1:T_seq
                    acc_w = zero(S)

                    for d in 1:D
                        acc_w += S(vh[d, s, h, b]) * S(ybar[d, t, h, b])
                    end

                    Wbar[s, t] = acc_w
                end
            end

            # Backprop through softmax over source positions for each target t:
            #
            #     Lbar[:, t] = W[:, t] .* (Wbar[:, t] .- dot(W[:, t], Wbar[:, t]))
            #
            # Only causal positions s <= t participate.

            Lbar = zeros(S, T_seq, T_seq)

            for t in 1:T_seq
                dot_term = zero(S)

                for s in 1:t
                    dot_term += W[s, t] * Wbar[s, t]
                end

                for s in 1:t
                    Lbar[s, t] = W[s, t] * (Wbar[s, t] - dot_term)
                end
            end

            # Backprop through:
            #
            #     L = K'Q / sqrt(D) - abs(decay[h]) * Δ + causal
            #
            # Cbar = Lbar
            #
            # Qbar = K * Cbar / sqrt(D)
            # Kbar = Q * Cbar' / sqrt(D)

            for d in 1:D
                for t in 1:T_seq
                    acc_q = zero(S)

                    for s in 1:T_seq
                        acc_q += S(kh[d, s, h, b]) * Lbar[s, t]
                    end

                    qbar[d, t, h, b] += acc_q * scale
                end
            end

            for d in 1:D
                for s in 1:T_seq
                    acc_k = zero(S)

                    for t in 1:T_seq
                        acc_k += S(qh[d, t, h, b]) * Lbar[s, t]
                    end

                    kbar[d, s, h, b] += acc_k * scale
                end
            end

            # Backprop through Hawkes log-domain decay:
            #
            #     L[s, t] = content[s, t] - abs(decay[h]) * Δ[s, t]
            #
            # dL / d abs_decay = -Δ
            # d abs_decay / d decay = sign(decay), away from zero

            acc_decay = zero(S)

            for t in 1:T_seq
                for s in 1:t
                    Δ = max(S(times[t, b] - times[s, b]), zero(S))
                    acc_decay += -Lbar[s, t] * Δ
                end
            end

            decaybar[h] += acc_decay * decay_sign
        end
    end

    return qbar, kbar, vbar, decaybar
end









#=
function ChainRulesCore.rrule(
    ::typeof(_hawkes_attention),
    qh,
    kh,
    vh,
    times,
    decay,
)
    # Forward pass.
    yh = _hawkes_attention(qh, kh, vh, times, decay)

    # Capture projection helpers outside the pullback.
    # ChainRulesCore.ProjectTo is important because AD systems may pass tangent
    # objects that need to be projected back into the primal array structure.
    project_qh = ChainRulesCore.ProjectTo(qh)
    project_kh = ChainRulesCore.ProjectTo(kh)
    project_vh = ChainRulesCore.ProjectTo(vh)
    project_times = ChainRulesCore.ProjectTo(times)
    project_decay = ChainRulesCore.ProjectTo(decay)

    function hawkes_attention_pullback(ȳh)
        # TODO:
        # Derive and implement:
        #
        # Forward:
        #   content = KᵀQ / sqrt(D)
        #   logits = content - abs(decay[h]) * Δt + causal
        #   weights = softmax(logits; dims=source_time)
        #   yh = V * weights
        #
        # Backward:
        #   ȳh       -> gradients wrt V and weights
        #   weights̄  -> gradients wrt logits via softmax Jacobian-vector product
        #   logits̄   -> gradients wrt Q, K, decay, and possibly times
        #
        # Return structure:
        #   NoTangent() for the function itself
        #   q̄h
        #   k̄h
        #   v̄h
        #   t̄imes
        #   d̄ecay
        #
        # For first production version, we may return NoTangent() for `times`
        # if we choose not to train through event timestamps.
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.@not_implemented("custom qh tangent not implemented"),
            ChainRulesCore.@not_implemented("custom kh tangent not implemented"),
            ChainRulesCore.@not_implemented("custom vh tangent not implemented"),
            ChainRulesCore.@not_implemented("custom times tangent not implemented"),
            ChainRulesCore.@not_implemented("custom decay tangent not implemented"),
        )
    end

    return yh, hawkes_attention_pullback
end
=#
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

This helper is registered through the `_hawkes_attention` ChainRulesCore `rrule`.

This version uses matrix multiplications per `(head, batch)` block for the
main linear algebra steps while keeping the causal softmax and decay gradient
explicit for clarity.
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

            Q = @view qh[:, :, h, b]
            Kmat = @view kh[:, :, h, b]
            V = @view vh[:, :, h, b]
            Ybar = @view ybar[:, :, h, b]

            Qbar = @view qbar[:, :, h, b]
            Kbar = @view kbar[:, :, h, b]
            Vbar = @view vbar[:, :, h, b]

            # ------------------------------------------------------------
            # Forward recomputation for this (head, batch) block.
            #
            # C[s, t] = dot(K[:, s], Q[:, t]) / sqrt(D)
            # W[s, t] = softmax(C[s, t] - abs(decay[h]) * Δ[s, t])
            # over causal source positions s <= t.
            # ------------------------------------------------------------

            C = Matrix{S}(undef, T_seq, T_seq)
            LinearAlgebra.mul!(C, transpose(Kmat), Q)
            C .*= scale

            W = zeros(S, T_seq, T_seq)

            for t in 1:T_seq
                max_score = typemin(S)

                for s in 1:t
                    Δ = max(S(times[t, b] - times[s, b]), zero(S))
                    score = C[s, t] - abs_decay_h * Δ

                    if score > max_score
                        max_score = score
                    end
                end

                denom = zero(S)

                for s in 1:t
                    Δ = max(S(times[t, b] - times[s, b]), zero(S))
                    weight_num = exp(C[s, t] - abs_decay_h * Δ - max_score)

                    W[s, t] = weight_num
                    denom += weight_num
                end

                inv_denom = inv(denom)

                for s in 1:t
                    W[s, t] *= inv_denom
                end
            end

            # ------------------------------------------------------------
            # Backprop through:
            #
            #     Y = V * W
            #
            # Reverse:
            #
            #     Vbar = Ybar * W'
            #     Wbar = V' * Ybar
            # ------------------------------------------------------------

            LinearAlgebra.mul!(Vbar, Ybar, transpose(W))

            Wbar = Matrix{S}(undef, T_seq, T_seq)
            LinearAlgebra.mul!(Wbar, transpose(V), Ybar)

            # ------------------------------------------------------------
            # Backprop through softmax over source positions.
            #
            # For each target t:
            #
            #     Lbar[:, t] = W[:, t] .* (Wbar[:, t] .- dot(W[:, t], Wbar[:, t]))
            #
            # Only causal positions s <= t participate.
            # ------------------------------------------------------------

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

            # ------------------------------------------------------------
            # Backprop through:
            #
            #     C = K'Q / sqrt(D)
            #
            # Reverse:
            #
            #     Qbar = K * Lbar / sqrt(D)
            #     Kbar = Q * Lbar' / sqrt(D)
            # ------------------------------------------------------------

            LinearAlgebra.mul!(Qbar, Kmat, Lbar)
            Qbar .*= scale

            LinearAlgebra.mul!(Kbar, Q, transpose(Lbar))
            Kbar .*= scale

            # ------------------------------------------------------------
            # Backprop through:
            #
            #     L[s, t] = C[s, t] - abs(decay[h]) * Δ[s, t]
            #
            # dL / d abs_decay = -Δ
            # d abs_decay / d decay = sign(decay), away from zero.
            # ------------------------------------------------------------

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

    project_qh = ChainRulesCore.ProjectTo(qh)
    project_kh = ChainRulesCore.ProjectTo(kh)
    project_vh = ChainRulesCore.ProjectTo(vh)
    project_decay = ChainRulesCore.ProjectTo(decay)

    function hawkes_attention_pullback(ybar)
        ybar_unthunked = ChainRulesCore.unthunk(ybar)

        qbar, kbar, vbar, decaybar =
            _hawkes_attention_backward(qh, kh, vh, times, decay, ybar_unthunked)

        return (
            ChainRulesCore.NoTangent(),      # function object
            project_qh(qbar),
            project_kh(kbar),
            project_vh(vbar),
            ChainRulesCore.NoTangent(),      # times treated as observed data
            project_decay(decaybar),
        )
    end

    return yh, hawkes_attention_pullback
end
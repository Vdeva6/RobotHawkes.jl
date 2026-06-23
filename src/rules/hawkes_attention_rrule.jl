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
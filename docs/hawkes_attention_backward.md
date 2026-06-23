# Hawkes Attention Backward Derivation

This document derives the manual reverse pass for the internal primitive:

```julia
_hawkes_attention(qh, kh, vh, times, decay)


# Hawkes Attention Backward Derivation

This document derives the manual reverse pass for the internal primitive:

```julia
_hawkes_attention(qh, kh, vh, times, decay)

The goal is to eventually replace Zygote's generic reverse pass through the attention computation with a custom ChainRulesCore.rrule.

This custom rule should reduce memory pressure for medium/large sequence lengths.

Shape Convention

The attention primitive receives:

qh     :: (D, T, H, B)
kh     :: (D, T, H, B)
vh     :: (D, T, H, B)
times  :: (T, B)
decay  :: (H,)

where:

D = head dimension
T = sequence length
H = number of heads
B = batch size

Internally, heads and batches are often collapsed:

N = H * B

so:

Q, K, V :: (D, T, N)

For each collapsed head-batch index n, attention is computed independently.

Forward Pass

For one head-batch block:

Q :: (D, T)
K :: (D, T)
V :: (D, T)

Define:

C = KᵀQ / sqrt(D)

where:

C :: (T, T)
C[s, t] = dot(K[:, s], Q[:, t]) / sqrt(D)

Here:

s = source/history index
t = target/query index

The Hawkes log-domain decay bias is:

R[s, t] = γ * Δ[s, t]

where:

γ = abs(decay[h])
Δ[s, t] = max(times[t, b] - times[s, b], 0)

The causal mask is:

M[s, t] = 0      if s ≤ t
M[s, t] = -Inf   if s > t

The attention logits are:

L = C - R + M

The attention weights are:

W = softmax(L; dims = source_time)

That means for each target time t:

W[:, t] = softmax(L[:, t])

The output is:

Y = V W

where:

Y :: (D, T)
Reverse Pass Overview

Given upstream sensitivity:

Ȳ = ∂loss/∂Y

we need:

Q̄ = ∂loss/∂Q
K̄ = ∂loss/∂K
V̄ = ∂loss/∂V
γ̄ = ∂loss/∂γ

Eventually we may also compute:

times̄ = ∂loss/∂times

but the first production rule may intentionally return NoTangent() for times, because event timestamps are observed inputs rather than trainable parameters.

Step 1: Backprop Through Y = V W

Forward:

Y = V W

Reverse:

V̄ = Ȳ Wᵀ
W̄ = Vᵀ Ȳ

Shapes:

V̄ :: (D, T)
W̄ :: (T, T)
Step 2: Backprop Through W = softmax(L)

Softmax is applied over source positions s for each fixed target t.

For each column t:

w = W[:, t]
w̄ = W̄[:, t]

The softmax vector-Jacobian product is:

L̄[:, t] = w .* (w̄ - dot(w, w̄))

So in code-like notation:

for t in 1:T
    dot_term = sum(W[:, t] .* Wbar[:, t])
    Lbar[:, t] = W[:, t] .* (Wbar[:, t] .- dot_term)
end

This is the key softmax backward identity.

Step 3: Backprop Through L = C - γΔ + M

The causal mask is constant, so it has no gradient.

C̄ = L̄

For decay:

L[s, t] = C[s, t] - abs(decay[h]) * Δ[s, t] + M[s, t]

Let:

γ = abs(decay[h])

Then:

∂L[s, t]/∂γ = -Δ[s, t]

So:

γ̄ = -ΣₛΣₜ L̄[s, t] * Δ[s, t]

Since:

γ = abs(decay[h])

we have:

decaȳ[h] = γ̄ * sign(decay[h])

This is nondifferentiable at decay[h] = 0, so tests should keep decay away from zero.

For the first custom rule, we may choose:

times̄ = NoTangent()

because event times are treated as observed data.

Step 4: Backprop Through C = KᵀQ / sqrt(D)

Forward:

C = KᵀQ / sqrt(D)

Reverse:

Q̄ = K C̄ᵀ / sqrt(D)
K̄ = Q C̄ᵀ?  careful with orientation

Let's derive carefully.

Elementwise:

C[s, t] = Σ_d K[d, s] * Q[d, t] / sqrt(D)

For Q:

Q̄[d, t] = Σ_s C̄[s, t] * K[d, s] / sqrt(D)

Matrix form:

Q̄ = K C̄ / sqrt(D)

because:

K       :: (D, T)
C̄      :: (T, T)
Q̄      :: (D, T)

For K:

K̄[d, s] = Σ_t C̄[s, t] * Q[d, t] / sqrt(D)

Matrix form:

K̄ = Q C̄ᵀ / sqrt(D)

because:

Q       :: (D, T)
C̄ᵀ     :: (T, T)
K̄      :: (D, T)

So:

Q̄ = K C̄ / sqrt(D)
K̄ = Q C̄ᵀ / sqrt(D)
Full Per-Head/Batch Backward Summary

For each head h and batch b:

Forward intermediates:

C = KᵀQ / sqrt(D)
Δ[s, t] = max(times[t, b] - times[s, b], 0)
L = C - abs(decay[h]) * Δ + causal
W = softmax(L; dims = 1)
Y = V W

Backward from upstream Ȳ:

V̄ = Ȳ Wᵀ
W̄ = Vᵀ Ȳ

Softmax backward:

for each target t:
    L̄[:, t] = W[:, t] .* (W̄[:, t] .- dot(W[:, t], W̄[:, t]))

Content score backward:

C̄ = L̄
Q̄ = K C̄ / sqrt(D)
K̄ = Q C̄ᵀ / sqrt(D)

Decay backward:

γ̄ = -ΣₛΣₜ L̄[s, t] * Δ[s, t]
decaȳ[h] += γ̄ * sign(decay[h])

Optional time backward:

not implemented initially
Implementation Notes for Future rrule

The future custom rule should likely:

Recompute forward intermediates inside the pullback instead of storing all of them.
Avoid materializing unnecessary T × T × H × B tensors when possible.
Accumulate gradients per (h, b) block.
Return NoTangent() for times initially unless training through timestamps becomes necessary.
Use ProjectTo for array tangent projection.

Expected rrule return shape:

return yh, pullback

where:

function pullback(ybar)
    return (
        NoTangent(),  # function object
        qbar,
        kbar,
        vbar,
        NoTangent(), # times, initially
        decaybar,
    )
end
Validation Plan

Before enabling the custom rule:

Compare _hawkes_attention output to reference_hawkes_attention.
Compare Zygote gradient to finite-difference gradient.
Implement manual backward as a separate helper.
Compare manual backward directional derivatives to finite differences.
Only then wrap the manual backward in ChainRulesCore.rrule.

The current benchmark target is medium:

T = 100
B = 32
H = 4
D = 16

Current bottleneck:

cell_gradient_params ≈ 1.3 seconds
memory ≈ 8.3 GiB

The custom rule is successful only if it significantly reduces the medium backward pass time and memory.



What we accomplished

We now have the mathematical map for the custom adjoint:

Y = V W
W = softmax(L)
L = KᵀQ / sqrt(D) - abs(decay) * Δ + causal

with backward:

V̄ = Ȳ Wᵀ
W̄ = VᵀȲ
L̄ = softmax_backward(W, W̄)
Q̄ = K L̄ / sqrt(D)
K̄ = Q L̄ᵀ / sqrt(D)
decaȳ = -Σ L̄ * Δ * sign(decay)
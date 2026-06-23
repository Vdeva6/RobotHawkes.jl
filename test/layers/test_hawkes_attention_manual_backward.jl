using Test
using Random
using Zygote
using RobotHawkes

function manual_backward_scalar_loss(qh, kh, vh, times, decay)
    y = RobotHawkes._hawkes_attention(qh, kh, vh, times, decay)
    return sum(abs2, y)
end

@testset "Hawkes attention manual backward" begin
    rng = Xoshiro(2468)

    # Tiny size by design. This test validates math, not speed.
    D = 2
    T_seq = 4
    H = 2
    B = 1

    qh = randn(rng, Float64, D, T_seq, H, B)
    kh = randn(rng, Float64, D, T_seq, H, B)
    vh = randn(rng, Float64, D, T_seq, H, B)

    Δt = rand(rng, Float64, T_seq, B) .+ 0.1
    times = cumsum(Δt; dims = 1)

    # Keep away from zero because abs(decay) is nondifferentiable at zero.
    decay = Float64[0.4, 0.9]

    y = RobotHawkes._hawkes_attention(qh, kh, vh, times, decay)
    ybar = 2 .* y

    qbar_manual, kbar_manual, vbar_manual, decaybar_manual =
        RobotHawkes._hawkes_attention_backward(qh, kh, vh, times, decay, ybar)

    loss_q(q) = manual_backward_scalar_loss(q, kh, vh, times, decay)
    loss_k(k) = manual_backward_scalar_loss(qh, k, vh, times, decay)
    loss_v(v) = manual_backward_scalar_loss(qh, kh, v, times, decay)
    loss_d(d) = manual_backward_scalar_loss(qh, kh, vh, times, d)

    qbar_zygote = first(Zygote.gradient(loss_q, qh))
    kbar_zygote = first(Zygote.gradient(loss_k, kh))
    vbar_zygote = first(Zygote.gradient(loss_v, vh))
    decaybar_zygote = first(Zygote.gradient(loss_d, decay))

    @test qbar_manual ≈ qbar_zygote rtol = 1.0e-8 atol = 1.0e-8
    @test kbar_manual ≈ kbar_zygote rtol = 1.0e-8 atol = 1.0e-8
    @test vbar_manual ≈ vbar_zygote rtol = 1.0e-8 atol = 1.0e-8
    @test decaybar_manual ≈ decaybar_zygote rtol = 1.0e-8 atol = 1.0e-8
end
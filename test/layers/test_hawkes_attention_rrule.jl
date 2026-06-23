using Test
using Random
using Zygote
using RobotHawkes

@testset "Hawkes attention custom rrule" begin
    rng = Xoshiro(13579)

    D = 2
    T_seq = 4
    H = 2
    B = 1

    qh = randn(rng, Float64, D, T_seq, H, B)
    kh = randn(rng, Float64, D, T_seq, H, B)
    vh = randn(rng, Float64, D, T_seq, H, B)

    Δt = rand(rng, Float64, T_seq, B) .+ 0.1
    times = cumsum(Δt; dims = 1)

    decay = Float64[0.25, 0.75]

    y = RobotHawkes._hawkes_attention(qh, kh, vh, times, decay)
    ybar = 2 .* y

    qbar_manual, kbar_manual, vbar_manual, decaybar_manual =
        RobotHawkes._hawkes_attention_backward(qh, kh, vh, times, decay, ybar)

    loss_q(q) = sum(abs2, RobotHawkes._hawkes_attention(q, kh, vh, times, decay))
    loss_k(k) = sum(abs2, RobotHawkes._hawkes_attention(qh, k, vh, times, decay))
    loss_v(v) = sum(abs2, RobotHawkes._hawkes_attention(qh, kh, v, times, decay))
    loss_d(d) = sum(abs2, RobotHawkes._hawkes_attention(qh, kh, vh, times, d))

    qbar_rule = first(Zygote.gradient(loss_q, qh))
    kbar_rule = first(Zygote.gradient(loss_k, kh))
    vbar_rule = first(Zygote.gradient(loss_v, vh))
    decaybar_rule = first(Zygote.gradient(loss_d, decay))

    @test qbar_rule ≈ qbar_manual rtol = 1.0e-8 atol = 1.0e-8
    @test kbar_rule ≈ kbar_manual rtol = 1.0e-8 atol = 1.0e-8
    @test vbar_rule ≈ vbar_manual rtol = 1.0e-8 atol = 1.0e-8
    @test decaybar_rule ≈ decaybar_manual rtol = 1.0e-8 atol = 1.0e-8
end
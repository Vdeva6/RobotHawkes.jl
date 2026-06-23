using Test
using Random
using Zygote
using RobotHawkes

function attention_scalar_loss(qh, kh, vh, times, decay)
    y = RobotHawkes._hawkes_attention(qh, kh, vh, times, decay)
    return sum(abs2, y)
end

function directional_finite_difference(f, x, dx; ϵ = 1.0e-6)
    return (f(x .+ ϵ .* dx) - f(x .- ϵ .* dx)) / (2ϵ)
end

function directional_ad(g, dx)
    return sum(g .* dx)
end

@testset "Hawkes attention finite-difference gradients" begin
    rng = Xoshiro(12345)

    # Keep this intentionally tiny. This test is for correctness, not speed.
    D = 2
    T_seq = 4
    H = 2
    B = 1

    qh = randn(rng, Float64, D, T_seq, H, B)
    kh = randn(rng, Float64, D, T_seq, H, B)
    vh = randn(rng, Float64, D, T_seq, H, B)

    Δt = rand(rng, Float64, T_seq, B) .+ 0.1
    times = cumsum(Δt; dims = 1)

    # Keep decay away from zero because abs(decay) is nondifferentiable at zero.
    decay = Float64[0.3, 0.7]

    dq = randn(rng, Float64, size(qh))
    dk = randn(rng, Float64, size(kh))
    dv = randn(rng, Float64, size(vh))
    dd = randn(rng, Float64, size(decay))

    loss_q(q) = attention_scalar_loss(q, kh, vh, times, decay)
    loss_k(k) = attention_scalar_loss(qh, k, vh, times, decay)
    loss_v(v) = attention_scalar_loss(qh, kh, v, times, decay)
    loss_d(d) = attention_scalar_loss(qh, kh, vh, times, d)

    gq = first(Zygote.gradient(loss_q, qh))
    gk = first(Zygote.gradient(loss_k, kh))
    gv = first(Zygote.gradient(loss_v, vh))
    gd = first(Zygote.gradient(loss_d, decay))

    fd_q = directional_finite_difference(loss_q, qh, dq)
    fd_k = directional_finite_difference(loss_k, kh, dk)
    fd_v = directional_finite_difference(loss_v, vh, dv)
    fd_d = directional_finite_difference(loss_d, decay, dd)

    ad_q = directional_ad(gq, dq)
    ad_k = directional_ad(gk, dk)
    ad_v = directional_ad(gv, dv)
    ad_d = directional_ad(gd, dd)

    @test ad_q ≈ fd_q rtol = 1.0e-4 atol = 1.0e-4
    @test ad_k ≈ fd_k rtol = 1.0e-4 atol = 1.0e-4
    @test ad_v ≈ fd_v rtol = 1.0e-4 atol = 1.0e-4
    @test ad_d ≈ fd_d rtol = 1.0e-4 atol = 1.0e-4
end
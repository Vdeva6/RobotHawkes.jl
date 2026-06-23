using Test
using Random
using Zygote
using RobotHawkes

function directional_finite_difference_loss(f, x, dx; ϵ = 1.0e-6)
    return (f(x .+ ϵ .* dx) - f(x .- ϵ .* dx)) / (2ϵ)
end

@testset "Loss custom rrules" begin
    rng = Xoshiro(112233)

    K = 5
    T_seq = 7
    B = 3

    λ = rand(rng, Float64, K, T_seq, B) .+ 0.5
    event_ids = rand(rng, 1:K, T_seq, B)
    Δt = rand(rng, Float64, T_seq, B) .+ 0.1

    dλ = randn(rng, Float64, size(λ))

    observed_loss(x) = observed_nll(x, event_ids; normalize = true)
    integral_loss(x) = total_intensity_integral(x, Δt; normalize = true)
    full_loss(x) = full_hawkes_nll(x, event_ids, Δt; normalize = true)

    observed_grad = first(Zygote.gradient(observed_loss, λ))
    integral_grad = first(Zygote.gradient(integral_loss, λ))
    full_grad = first(Zygote.gradient(full_loss, λ))

    observed_fd = directional_finite_difference_loss(observed_loss, λ, dλ)
    integral_fd = directional_finite_difference_loss(integral_loss, λ, dλ)
    full_fd = directional_finite_difference_loss(full_loss, λ, dλ)

    observed_ad = sum(observed_grad .* dλ)
    integral_ad = sum(integral_grad .* dλ)
    full_ad = sum(full_grad .* dλ)

    @test observed_ad ≈ observed_fd rtol = 1.0e-5 atol = 1.0e-5
    @test integral_ad ≈ integral_fd rtol = 1.0e-5 atol = 1.0e-5
    @test full_ad ≈ full_fd rtol = 1.0e-5 atol = 1.0e-5

    @test full_grad ≈ observed_grad .+ integral_grad rtol = 1.0e-12 atol = 1.0e-12
end
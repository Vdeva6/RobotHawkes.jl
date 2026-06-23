using Test
using Random
using Lux
using Zygote
using RobotHawkes

@testset "RobotHawkes.jl" begin
    @testset "TemporalEmbedding" begin
        rng = Xoshiro(42)

        layer = TemporalEmbedding(16)
        ps, st = Lux.setup(rng, layer)

        Δt = rand(Float32, 10, 4)

        y, st_new = layer(Δt, ps, st)

        @test size(y) == (16, 10, 4)
        @test st_new == st
        @test haskey(ps, :logfreq)
        @test haskey(ps, :phase)
        @test haskey(ps, :scale)
        @test eltype(y) == Float32
    end

    @testset "EventEmbedding" begin
        rng = Xoshiro(123)

        layer = EventEmbedding(7, 16)
        ps, st = Lux.setup(rng, layer)

        event_ids = [
            1 2 3 4;
            2 3 4 5;
            3 4 5 6;
            4 5 6 7;
            5 6 7 1;
        ]

        y, st_new = layer(event_ids, ps, st)

        @test size(y) == (16, 5, 4)
        @test st_new == st
        @test haskey(ps, :table)
        @test eltype(y) == Float32

        @test y[:, 1, 1] == ps.table[:, 1]
        @test y[:, 5, 4] == ps.table[:, 1]
    end

    @testset "TransformerHawkesCell" begin
        rng = Xoshiro(999)

        embed_dim = 16
        num_heads = 4
        T_seq = 8
        B = 3

        layer = TransformerHawkesCell(embed_dim, num_heads)
        ps, st = Lux.setup(rng, layer)

        x = rand(Float32, embed_dim, T_seq, B)
        Δt = rand(Float32, T_seq, B)
        times = cumsum(Δt; dims=1)

        y, st_new = layer(x, times, ps, st)

        @test size(y) == (embed_dim, T_seq, B)
        @test st_new == st
        @test eltype(y) == Float32

        @test haskey(ps, :Wq)
        @test haskey(ps, :Wk)
        @test haskey(ps, :Wv)
        @test haskey(ps, :Wo)
        @test haskey(ps, :decay)

        @test_throws ArgumentError TransformerHawkesCell(15, 4)
    end

    @testset "TransformerHawkesModel" begin
        rng = Xoshiro(2024)

        num_events = 6
        embed_dim = 16
        num_heads = 4
        T_seq = 9
        B = 2

        model = TransformerHawkesModel(num_events, embed_dim, num_heads)
        ps, st = Lux.setup(rng, model)

        event_ids = rand(1:num_events, T_seq, B)
        Δt = rand(Float32, T_seq, B)

        λ, st_new = model(event_ids, Δt, ps, st)

        @test size(λ) == (num_events, T_seq, B)
        @test st_new == st
        @test eltype(λ) == Float32
        @test all(λ .> 0)

        @test haskey(ps, :event)
        @test haskey(ps, :time)
        @test haskey(ps, :cell)
        @test haskey(ps, :Wλ)
        @test haskey(ps, :bλ)

        @test_throws ArgumentError TransformerHawkesModel(num_events, 15, 4)
    end

    @testset "Observed Log-Likelihood" begin
        λ = Float32[
            0.5 0.6 0.7;
            1.0 1.1 1.2
        ]

        # reshape into (K=2, T=3, B=1)
        λ = reshape(λ, 2, 3, 1)

        event_ids = reshape([1, 2, 1], 3, 1)

        ll = observed_loglikelihood(λ, event_ids)

        expected = log(Float32(0.5)) + log(Float32(1.1)) + log(Float32(0.7))

        @test ll ≈ expected
        @test observed_nll(λ, event_ids; normalize=false) ≈ -expected
        @test observed_nll(λ, event_ids; normalize=true) ≈ -expected / 3

        bad_event_ids = reshape([1, 3, 1], 3, 1)
        @test_throws ArgumentError observed_loglikelihood(λ, bad_event_ids)
    end

    @testset "Model Observed NLL + Zygote" begin
        rng = Xoshiro(7)

        num_events = 4
        embed_dim = 16
        num_heads = 4
        T_seq = 6
        B = 2

        model = TransformerHawkesModel(num_events, embed_dim, num_heads)
        ps, st = Lux.setup(rng, model)

        event_ids = rand(1:num_events, T_seq, B)
        Δt = rand(Float32, T_seq, B)

        loss, st_new = model_observed_nll(model, event_ids, Δt, ps, st)

        @test loss isa Real
        @test loss > 0
        @test st_new == st

        grad = Zygote.gradient(ps) do p
            first(model_observed_nll(model, event_ids, Δt, p, st))
        end

        @test grad !== nothing
        @test grad[1] !== nothing
    end

        @testset "Total Intensity Integral" begin
        λ = Float32[
            1.0 2.0 3.0;
            0.5 0.5 0.5
        ]

        λ = reshape(λ, 2, 3, 1)

        Δt = reshape(Float32[0.1, 0.2, 0.3], 3, 1)

        integral = total_intensity_integral(λ, Δt; normalize=false)

        expected =
            Float32(0.1) * (Float32(1.0) + Float32(0.5)) +
            Float32(0.2) * (Float32(2.0) + Float32(0.5)) +
            Float32(0.3) * (Float32(3.0) + Float32(0.5))

        @test integral ≈ expected
        @test total_intensity_integral(λ, Δt; normalize=true) ≈ expected / 3

        bad_Δt = rand(Float32, 4, 1)
        @test_throws DimensionMismatch total_intensity_integral(λ, bad_Δt)
    end

    @testset "Full Hawkes NLL" begin
        λ = Float32[
            0.5 0.6 0.7;
            1.0 1.1 1.2
        ]

        λ = reshape(λ, 2, 3, 1)

        event_ids = reshape([1, 2, 1], 3, 1)
        Δt = reshape(Float32[0.1, 0.2, 0.3], 3, 1)

        event_term = observed_nll(λ, event_ids; normalize=false)
        integral_term = total_intensity_integral(λ, Δt; normalize=false)

        @test full_hawkes_nll(λ, event_ids, Δt; normalize=false) ≈ event_term + integral_term
    end

    @testset "Model Full Hawkes NLL + Zygote" begin
        rng = Xoshiro(77)

        num_events = 4
        embed_dim = 16
        num_heads = 4
        T_seq = 6
        B = 2

        model = TransformerHawkesModel(num_events, embed_dim, num_heads)
        ps, st = Lux.setup(rng, model)

        event_ids = rand(1:num_events, T_seq, B)
        Δt = rand(Float32, T_seq, B)

        loss, st_new = model_full_hawkes_nll(model, event_ids, Δt, ps, st)

        @test loss isa Real
        @test loss > 0
        @test st_new == st

        grad = Zygote.gradient(ps) do p
            first(model_full_hawkes_nll(model, event_ids, Δt, p, st))
        end

        @test grad !== nothing
        @test grad[1] !== nothing
    end

    @testset "QuadGK Integral Utility" begin
        val = quadgk_integral(t -> t^2, 0.0, 1.0)

        @test val ≈ 1 / 3 atol = 1e-5
    end
    include("layers/test_transformer_hawkes_cell.jl")
    include("layers/test_hawkes_attention_reference.jl")
end
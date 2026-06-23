using Test
using Random
using Lux
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
end
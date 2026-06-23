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

        # Check that the first output vector equals the table column
        @test y[:, 1, 1] == ps.table[:, 1]

        # Check another event id lookup
        @test y[:, 5, 4] == ps.table[:, 1]
    end
end
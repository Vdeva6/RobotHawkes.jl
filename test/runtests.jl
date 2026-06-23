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
end
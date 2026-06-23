using Test
using Random
using Lux
using Zygote
using RobotHawkes

@testset "TransformerHawkesCell isolated attention" begin
    rng = Xoshiro(123)

    embed_dim = 16
    num_heads = 4
    T_seq = 6
    B = 2

    cell = TransformerHawkesCell(embed_dim, num_heads)
    ps, st = Lux.setup(rng, cell)

    x = rand(rng, Float32, embed_dim, T_seq, B)
    Δt = rand(rng, Float32, T_seq, B)
    times = cumsum(Δt; dims = 1)

    y, st_new = cell(x, times, ps, st)

    @test size(y) == (embed_dim, T_seq, B)
    @test st_new == st
    @test eltype(y) == Float32
    @test all(isfinite, y)

    grad = Zygote.gradient(ps) do p
        y, _ = cell(x, times, p, st)
        sum(abs2, y)
    end

    @test grad !== nothing
    @test grad[1] !== nothing
    @test haskey(grad[1], :Wq)
    @test haskey(grad[1], :Wk)
    @test haskey(grad[1], :Wv)
    @test haskey(grad[1], :Wo)
    @test haskey(grad[1], :decay)
end
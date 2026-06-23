using BenchmarkTools
using Lux
using Random
using RobotHawkes

println("RobotHawkes benchmark harness is ready.")

rng = Xoshiro(42)

layer = TemporalEmbedding(16)
ps, st = Lux.setup(rng, layer)

Δt = rand(Float32, 100, 32)

println("Benchmarking TemporalEmbedding forward pass...")

@btime $layer($Δt, $ps, $st)
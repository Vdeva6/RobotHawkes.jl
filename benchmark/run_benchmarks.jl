using BenchmarkTools
using Lux
using Random
using RobotHawkes

println("RobotHawkes benchmark harness is ready.")

rng = Xoshiro(42)

T_seq = 100
B = 32
embed_dim = 64
num_events = 16

time_layer = TemporalEmbedding(embed_dim)
time_ps, time_st = Lux.setup(rng, time_layer)

event_layer = EventEmbedding(num_events, embed_dim)
event_ps, event_st = Lux.setup(rng, event_layer)

Δt = rand(Float32, T_seq, B)
event_ids = rand(1:num_events, T_seq, B)

println()
println("Benchmarking TemporalEmbedding forward pass...")
@btime $time_layer($Δt, $time_ps, $time_st)

println()
println("Benchmarking EventEmbedding forward pass...")
@btime $event_layer($event_ids, $event_ps, $event_st)

println()
println("Benchmarking combined input representation...")
@btime begin
    x_time, _ = $time_layer($Δt, $time_ps, $time_st)
    x_event, _ = $event_layer($event_ids, $event_ps, $event_st)
    x_event .+ x_time
end
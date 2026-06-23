using BenchmarkTools
using Lux
using Random
using Zygote
using RobotHawkes

const PRESETS = Dict(
    "small" => (
        T_seq = 20,
        B = 8,
        embed_dim = 32,
        num_heads = 4,
    ),
    "medium" => (
        T_seq = 100,
        B = 32,
        embed_dim = 64,
        num_heads = 4,
    ),
)

function make_attention_problem(;
    seed::Integer = 42,
    T_seq::Integer,
    B::Integer,
    embed_dim::Integer,
    num_heads::Integer,
)
    rng = Xoshiro(seed)

    cell = TransformerHawkesCell(embed_dim, num_heads)
    ps, st = Lux.setup(rng, cell)

    x = rand(rng, Float32, embed_dim, T_seq, B)
    Δt = rand(rng, Float32, T_seq, B)
    times = cumsum(Δt; dims = 1)

    return cell, ps, st, x, times
end

function build_suite!(preset::String)
    haskey(PRESETS, preset) ||
        throw(ArgumentError("unknown preset: $preset. Use small or medium."))

    cfg = PRESETS[preset]

    cell, ps, st, x, times = make_attention_problem(;
        T_seq = cfg.T_seq,
        B = cfg.B,
        embed_dim = cfg.embed_dim,
        num_heads = cfg.num_heads,
    )

    suite = BenchmarkGroup()

    suite["cell_forward"] = @benchmarkable begin
        $cell($x, $times, $ps, $st)
    end

    suite["cell_gradient_params"] = @benchmarkable begin
        Zygote.gradient($ps) do p
            y, _ = $cell($x, $times, p, $st)
            sum(y)
        end
    end

    return suite, cfg
end

function print_trial_summary(name, trial)
    println()
    println("== $name ==")
    println("  minimum time:   ", round(minimum(trial).time / 1e6; digits = 4), " ms")
    println("  median time:    ", round(median(trial).time / 1e6; digits = 4), " ms")
    println("  mean time:      ", round(mean(trial).time / 1e6; digits = 4), " ms")
    println("  allocations:    ", minimum(trial).allocs)
    println("  memory:         ", round(minimum(trial).memory / 1024; digits = 2), " KiB")
    println("  gc time:        ", round(minimum(trial).gctime / 1e6; digits = 4), " ms")
end

function print_config(preset, cfg)
    println("RobotHawkes isolated attention benchmark")
    println("=======================================")
    println("Preset: $preset")
    println()
    println("Problem configuration:")
    println("  T_seq:     ", cfg.T_seq)
    println("  B:         ", cfg.B)
    println("  embed_dim: ", cfg.embed_dim)
    println("  num_heads: ", cfg.num_heads)
end

function main()
    preset = get(ARGS, 1, "small")

    suite, cfg = build_suite!(preset)
    print_config(preset, cfg)

    println()
    println("Tuning benchmarks...")
    tune!(suite)

    println()
    println("Running benchmarks...")
    results = run(suite; seconds = 1, samples = 3, evals = 1, verbose = true)

    for name in sort(collect(keys(results)))
        print_trial_summary(name, results[name])
    end

    return results
end

main()
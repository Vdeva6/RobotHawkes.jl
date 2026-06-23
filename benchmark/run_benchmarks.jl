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
        num_events = 8,
        num_heads = 4,
    ),
    "medium" => (
        T_seq = 100,
        B = 32,
        embed_dim = 64,
        num_events = 16,
        num_heads = 4,
    ),
    "large" => (
        T_seq = 200,
        B = 64,
        embed_dim = 128,
        num_events = 32,
        num_heads = 8,
    ),
)

function make_problem(;
    seed::Integer = 42,
    T_seq::Integer,
    B::Integer,
    embed_dim::Integer,
    num_events::Integer,
    num_heads::Integer,
)
    rng = Xoshiro(seed)

    model = TransformerHawkesModel(num_events, embed_dim, num_heads)
    ps, st = Lux.setup(rng, model)

    event_ids = rand(rng, 1:num_events, T_seq, B)
    Δt = rand(rng, Float32, T_seq, B)

    return model, ps, st, event_ids, Δt
end

function build_suite!(mode::String, preset::String)
    haskey(PRESETS, preset) ||
        throw(ArgumentError("unknown preset: $preset. Use small, medium, or large."))

    cfg = PRESETS[preset]

    model, ps, st, event_ids, Δt = make_problem(;
        T_seq = cfg.T_seq,
        B = cfg.B,
        embed_dim = cfg.embed_dim,
        num_events = cfg.num_events,
        num_heads = cfg.num_heads,
    )

    suite = BenchmarkGroup()

    suite["forward"] = @benchmarkable $model($event_ids, $Δt, $ps, $st)

    suite["observed_nll"] = @benchmarkable begin
        model_observed_nll($model, $event_ids, $Δt, $ps, $st)
    end

    suite["full_hawkes_nll"] = @benchmarkable begin
        model_full_hawkes_nll($model, $event_ids, $Δt, $ps, $st)
    end

    if mode == "grad"
        suite["observed_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_observed_nll($model, $event_ids, $Δt, p, $st))
            end
        end
    elseif mode == "fullgrad"
        if preset == "large"
            throw(ArgumentError("fullgrad is disabled for the large preset. Use small or medium."))
        end

        suite["observed_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_observed_nll($model, $event_ids, $Δt, p, $st))
            end
        end

        suite["full_hawkes_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_full_hawkes_nll($model, $event_ids, $Δt, p, $st))
            end
        end
    elseif mode != "quick"
        throw(ArgumentError("unknown benchmark mode: $mode. Use quick, grad, or fullgrad."))
    end

    return suite, cfg
end

function print_trial_summary(name, trial)
    println()
    println("== $name ==")
    println("  minimum time:   ", round(minimum(trial).time / 1e6; digits=4), " ms")
    println("  median time:    ", round(median(trial).time / 1e6; digits=4), " ms")
    println("  mean time:      ", round(mean(trial).time / 1e6; digits=4), " ms")
    println("  allocations:    ", minimum(trial).allocs)
    println("  memory:         ", round(minimum(trial).memory / 1024; digits=2), " KiB")
    println("  gc time:        ", round(minimum(trial).gctime / 1e6; digits=4), " ms")
end

function print_config(mode, preset, cfg)
    println("RobotHawkes benchmark suite")
    println("===========================")
    println("Mode:   $mode")
    println("Preset: $preset")
    println()
    println("Problem configuration:")
    println("  T_seq:      ", cfg.T_seq)
    println("  B:          ", cfg.B)
    println("  embed_dim:  ", cfg.embed_dim)
    println("  num_events: ", cfg.num_events)
    println("  num_heads:  ", cfg.num_heads)
end

function main()
    mode = get(ARGS, 1, "quick")
    preset = get(ARGS, 2, "small")

    suite, cfg = build_suite!(mode, preset)
    print_config(mode, preset, cfg)

    println()
    println("Tuning benchmarks...")
    tune!(suite)

    println()
    println("Running benchmarks...")
    results = run(suite; seconds=1, samples=3, evals=1, verbose=true)

    for name in sort(collect(keys(results)))
        print_trial_summary(name, results[name])
    end

    return results
end

main()
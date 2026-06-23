using BenchmarkTools
using Lux
using Random
using Zygote
using RobotHawkes

const SUITE = BenchmarkGroup()

function make_problem(;
    seed::Integer=42,
    T_seq::Integer=20,
    B::Integer=8,
    embed_dim::Integer=32,
    num_events::Integer=8,
    num_heads::Integer=4,
)
    rng = Xoshiro(seed)

    model = TransformerHawkesModel(num_events, embed_dim, num_heads)
    ps, st = Lux.setup(rng, model)

    event_ids = rand(rng, 1:num_events, T_seq, B)
    Δt = rand(rng, Float32, T_seq, B)

    return model, ps, st, event_ids, Δt
end

function build_suite!(mode::String)
    model, ps, st, event_ids, Δt = make_problem()

    SUITE["forward"] = @benchmarkable $model($event_ids, $Δt, $ps, $st)

    SUITE["observed_nll"] = @benchmarkable begin
        model_observed_nll($model, $event_ids, $Δt, $ps, $st)
    end

    SUITE["full_hawkes_nll"] = @benchmarkable begin
        model_full_hawkes_nll($model, $event_ids, $Δt, $ps, $st)
    end

    if mode == "grad"
        SUITE["observed_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_observed_nll($model, $event_ids, $Δt, p, $st))
            end
        end
    elseif mode == "fullgrad"
        SUITE["observed_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_observed_nll($model, $event_ids, $Δt, p, $st))
            end
        end

        SUITE["full_hawkes_nll_gradient"] = @benchmarkable begin
            Zygote.gradient($ps) do p
                first(model_full_hawkes_nll($model, $event_ids, $Δt, p, $st))
            end
        end
    elseif mode != "quick"
        throw(ArgumentError("unknown benchmark mode: $mode. Use quick, grad, or fullgrad."))
    end

    return SUITE
end

function print_trial_summary(name, trial)
    println()
    println("== $name ==")
    println("  minimum time:   ", minimum(trial).time / 1e6, " ms")
    println("  median time:    ", median(trial).time / 1e6, " ms")
    println("  mean time:      ", mean(trial).time / 1e6, " ms")
    println("  allocations:    ", minimum(trial).allocs)
    println("  memory:         ", minimum(trial).memory / 1024, " KiB")
    println("  gc time:        ", minimum(trial).gctime / 1e6, " ms")
end

function main()
    mode = get(ARGS, 1, "quick")

    println("RobotHawkes benchmark suite")
    println("===========================")
    println("Mode: $mode")

    suite = build_suite!(mode)

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
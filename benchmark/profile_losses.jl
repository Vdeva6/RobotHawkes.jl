using BenchmarkTools
using Random
using Zygote
using RobotHawkes

const PRESETS = Dict(
    "small" => (T_seq = 20, B = 8, num_events = 8),
    "medium" => (T_seq = 100, B = 32, num_events = 16),
    "large" => (T_seq = 200, B = 64, num_events = 32),
)

function make_problem(preset_name::String)
    haskey(PRESETS, preset_name) ||
        error("Unknown preset: $preset_name. Valid presets: $(collect(keys(PRESETS)))")

    cfg = PRESETS[preset_name]

    rng = Xoshiro(2026)

    λ = rand(rng, Float32, cfg.num_events, cfg.T_seq, cfg.B) .+ 0.1f0
    event_ids = rand(rng, 1:cfg.num_events, cfg.T_seq, cfg.B)
    Δt = rand(rng, Float32, cfg.T_seq, cfg.B) .+ 0.01f0

    return cfg, λ, event_ids, Δt
end

observed_lambda_grad(λ, event_ids) =
    Zygote.gradient(x -> observed_nll(x, event_ids; normalize = true), λ)[1]

integral_lambda_grad(λ, Δt) =
    Zygote.gradient(x -> total_intensity_integral(x, Δt; normalize = true), λ)[1]

full_lambda_grad(λ, event_ids, Δt) =
    Zygote.gradient(x -> full_hawkes_nll(x, event_ids, Δt; normalize = true), λ)[1]

function summarize_trial(name, trial)
    min_trial = minimum(trial)
    med_trial = median(trial)
    mean_trial = mean(trial)

    println()
    println("== $name ==")
    println("  minimum time:   ", round(min_trial.time / 1e6; digits = 4), " ms")
    println("  median time:    ", round(med_trial.time / 1e6; digits = 4), " ms")
    println("  mean time:      ", round(mean_trial.time / 1e6; digits = 4), " ms")
    println("  allocations:    ", min_trial.allocs)
    println("  memory:         ", round(min_trial.memory / 1024; digits = 2), " KiB")
    println("  gc time:        ", round(min_trial.gctime / 1e6; digits = 4), " ms")
end

function main()
    preset_name = length(ARGS) >= 1 ? ARGS[1] : "medium"

    cfg, λ, event_ids, Δt = make_problem(preset_name)

    println("RobotHawkes loss-gradient benchmark")
    println("===================================")
    println("Preset: $preset_name")
    println()
    println("Problem configuration:")
    println("  T_seq:      ", cfg.T_seq)
    println("  B:          ", cfg.B)
    println("  num_events: ", cfg.num_events)

    benchmarks = Dict(
        "observed_nll_lambda_gradient" =>
            (@benchmarkable observed_lambda_grad($λ, $event_ids)),
        "total_intensity_integral_lambda_gradient" =>
            (@benchmarkable integral_lambda_grad($λ, $Δt)),
        "full_hawkes_nll_lambda_gradient" =>
            (@benchmarkable full_lambda_grad($λ, $event_ids, $Δt)),
    )

    println()
    println("Tuning benchmarks...")
    for benchmark in values(benchmarks)
        tune!(benchmark)
    end

    println()
    println("Running benchmarks...")

    results = Dict{String, Any}()

    for (i, name) in enumerate(sort(collect(keys(benchmarks))))
        println("($i/$(length(benchmarks))) benchmarking \"$name\"...")
        results[name] = run(benchmarks[name]; seconds = 1, samples = 3, evals = 1, verbose = true)
        println("done")
    end

    for name in sort(collect(keys(results)))
        summarize_trial(name, results[name])
    end
end

main()

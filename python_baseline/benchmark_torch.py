import argparse
import statistics
import time
import tracemalloc

import torch

from transformer_hawkes_torch import (
    TransformerHawkesConfig,
    TransformerHawkesModel,
    model_full_hawkes_nll,
    model_observed_nll,
)


PRESETS = {
    "small": {
        "T_seq": 20,
        "B": 8,
        "embed_dim": 32,
        "num_events": 8,
        "num_heads": 4,
    },
    "medium": {
        "T_seq": 100,
        "B": 32,
        "embed_dim": 64,
        "num_events": 16,
        "num_heads": 4,
    },
    "large": {
        "T_seq": 200,
        "B": 64,
        "embed_dim": 128,
        "num_events": 32,
        "num_heads": 8,
    },
}


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def make_problem(preset: str, device: torch.device):
    cfg = PRESETS[preset]

    torch.manual_seed(42)

    config = TransformerHawkesConfig(
        num_events=cfg["num_events"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
    )

    model = TransformerHawkesModel(config).to(device)

    event_ids = torch.randint(
        low=1,
        high=cfg["num_events"] + 1,
        size=(cfg["T_seq"], cfg["B"]),
        device=device,
    )

    delta_t = torch.rand(
        cfg["T_seq"],
        cfg["B"],
        dtype=torch.float32,
        device=device,
    )

    return model, event_ids, delta_t, cfg


def clear_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def benchmark_callable(fn, device: torch.device, repeats: int = 10, warmup: int = 3):
    for _ in range(warmup):
        fn()
        synchronize(device)

    times_ms = []

    tracemalloc.start()

    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "min_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "python_current_kib": current / 1024,
        "python_peak_kib": peak / 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["quick", "grad", "fullgrad"], nargs="?", default="quick")
    parser.add_argument("preset", choices=["small", "medium", "large"], nargs="?", default="small")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.mode == "fullgrad" and args.preset == "large":
        raise RuntimeError("fullgrad is disabled for the large preset.")

    model, event_ids, delta_t, cfg = make_problem(args.preset, device)

    print("PyTorch Transformer Hawkes benchmark")
    print("===================================")
    print(f"Mode:   {args.mode}")
    print(f"Preset: {args.preset}")
    print(f"Device: {device}")
    print()
    print("Problem configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    benchmarks = {}

    model.eval()

    with torch.no_grad():
        benchmarks["forward"] = lambda: model(event_ids, delta_t)
        benchmarks["observed_nll"] = lambda: model_observed_nll(model, event_ids, delta_t)
        benchmarks["full_hawkes_nll"] = lambda: model_full_hawkes_nll(model, event_ids, delta_t)

        print()
        for name, fn in benchmarks.items():
            result = benchmark_callable(fn, device, repeats=args.repeats)
            print_result(name, result)

    if args.mode in {"grad", "fullgrad"}:
        model.train()

        def observed_grad():
            clear_grads(model)
            loss = model_observed_nll(model, event_ids, delta_t)
            loss.backward()
            return loss

        result = benchmark_callable(observed_grad, device, repeats=max(3, args.repeats // 2))
        print_result("observed_nll_gradient", result)

    if args.mode == "fullgrad":
        def full_grad():
            clear_grads(model)
            loss = model_full_hawkes_nll(model, event_ids, delta_t)
            loss.backward()
            return loss

        result = benchmark_callable(full_grad, device, repeats=max(3, args.repeats // 2))
        print_result("full_hawkes_nll_gradient", result)


def print_result(name: str, result: dict) -> None:
    print()
    print(f"== {name} ==")
    print(f"  minimum time:        {result['min_ms']:.4f} ms")
    print(f"  median time:         {result['median_ms']:.4f} ms")
    print(f"  mean time:           {result['mean_ms']:.4f} ms")
    print(f"  python current mem:  {result['python_current_kib']:.2f} KiB")
    print(f"  python peak mem:     {result['python_peak_kib']:.2f} KiB")


if __name__ == "__main__":
    main()
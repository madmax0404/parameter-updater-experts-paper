"""Microbenchmark the gated-lerp sequential loop vs parallel scan.

This benchmark isolates the recurrent state update used by DeltaNet-LoRA. It
does not load OLMoE and does not reproduce full end-to-end training time. The
`--exp28-shape` flag uses the LoRA factor dimensions from experiment 28.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from parallel_scan import exp28_lora_factor_sizes, parallel_gated_lerp_scan


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sequential_layer_scan(
    beta: torch.Tensor,
    factor_values: list[torch.Tensor],
    initial_factors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Sequential loop matching the four-factor update structure."""
    batch, seq_len = beta.shape
    states = list(initial_factors)
    snapshots: list[list[torch.Tensor]] = [[] for _ in factor_values]

    for pos in range(seq_len):
        for factor_idx, values in enumerate(factor_values):
            beta_t = beta[:, pos].reshape(batch, *([1] * (values.ndim - 2)))
            states[factor_idx] = (
                (1 - beta_t) * states[factor_idx] + beta_t * values[:, pos]
            )
            snapshots[factor_idx].append(states[factor_idx])

    return [torch.stack(factor_snapshots, dim=1) for factor_snapshots in snapshots]


def build_inputs(args: argparse.Namespace, device: torch.device):
    factor_sizes = exp28_lora_factor_sizes(
        d_model=args.d_model, d_ff=args.d_ff, rank=args.rank
    )
    beta_logits_by_layer = []
    values_by_layer = []
    initial_by_layer = []

    for _ in range(args.layers):
        beta_logits = torch.randn(
            args.batch_size, args.seq_len, device=device
        ).requires_grad_()
        factor_values = [
            (torch.randn(args.batch_size, args.seq_len, size, device=device) * 0.02)
            .requires_grad_()
            for size in factor_sizes
        ]
        initial_factors = [
            torch.zeros(args.batch_size, size, device=device) for size in factor_sizes
        ]
        beta_logits_by_layer.append(beta_logits)
        values_by_layer.append(factor_values)
        initial_by_layer.append(initial_factors)

    return factor_sizes, beta_logits_by_layer, values_by_layer, initial_by_layer


def clear_grads(beta_logits_by_layer, values_by_layer) -> None:
    for beta_logits in beta_logits_by_layer:
        beta_logits.grad = None
    for layer_values in values_by_layer:
        for values in layer_values:
            values.grad = None


def run_sequential(beta_logits_by_layer, values_by_layer, initial_by_layer) -> torch.Tensor:
    loss = values_by_layer[0][0].new_zeros(())
    for beta_logits, factor_values, initial_factors in zip(
        beta_logits_by_layer, values_by_layer, initial_by_layer
    ):
        beta = torch.sigmoid(beta_logits)
        snapshots = sequential_layer_scan(beta, factor_values, initial_factors)
        for states in snapshots:
            loss = loss + states.square().mean()
    return loss / (len(values_by_layer) * len(values_by_layer[0]))


def run_parallel(beta_logits_by_layer, values_by_layer, initial_by_layer) -> torch.Tensor:
    loss = values_by_layer[0][0].new_zeros(())
    for beta_logits, factor_values, initial_factors in zip(
        beta_logits_by_layer, values_by_layer, initial_by_layer
    ):
        beta = torch.sigmoid(beta_logits)
        for values, initial_state in zip(factor_values, initial_factors):
            states = parallel_gated_lerp_scan(beta, values, initial_state)
            loss = loss + states.square().mean()
    return loss / (len(values_by_layer) * len(values_by_layer[0]))


def time_runner(
    label: str,
    fn,
    beta_logits_by_layer,
    values_by_layer,
    initial_by_layer,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    include_backward: bool,
) -> dict[str, float | str]:
    for _ in range(warmup):
        clear_grads(beta_logits_by_layer, values_by_layer)
        loss = fn(beta_logits_by_layer, values_by_layer, initial_by_layer)
        if include_backward:
            loss.backward()
        sync(device)

    times = []
    for _ in range(iters):
        clear_grads(beta_logits_by_layer, values_by_layer)
        sync(device)
        start = time.perf_counter()
        loss = fn(beta_logits_by_layer, values_by_layer, initial_by_layer)
        if include_backward:
            loss.backward()
        sync(device)
        times.append(time.perf_counter() - start)

    return {
        "label": label,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
    }


def max_correctness_diff(beta_logits_by_layer, values_by_layer, initial_by_layer) -> float:
    with torch.no_grad():
        seq_loss = run_sequential(
            beta_logits_by_layer, values_by_layer, initial_by_layer
        )
        par_loss = run_parallel(beta_logits_by_layer, values_by_layer, initial_by_layer)
    return float((seq_loss - par_loss).abs().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument(
        "--exp28-shape",
        action="store_true",
        help="Use B=8, S=120, layers=16, D=2048, d_ff=1024, rank=4.",
    )
    args = parser.parse_args()

    if args.exp28_shape:
        args.batch_size = 8
        args.seq_len = 120
        args.layers = 16
        args.d_model = 2048
        args.d_ff = 1024
        args.rank = 4

    torch.manual_seed(0)
    device = resolve_device(args.device)
    factor_sizes, beta_logits_by_layer, values_by_layer, initial_by_layer = build_inputs(
        args, device
    )

    include_backward = not args.no_backward
    diff = max_correctness_diff(beta_logits_by_layer, values_by_layer, initial_by_layer)
    sequential = time_runner(
        "sequential",
        run_sequential,
        beta_logits_by_layer,
        values_by_layer,
        initial_by_layer,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        include_backward=include_backward,
    )
    parallel = time_runner(
        "parallel",
        run_parallel,
        beta_logits_by_layer,
        values_by_layer,
        initial_by_layer,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        include_backward=include_backward,
    )

    result = {
        "scope": "DeltaNet-LoRA recurrence microbenchmark only; not full OLMoE training.",
        "device": str(device),
        "include_backward": include_backward,
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "layers": args.layers,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "rank": args.rank,
            "factor_sizes": factor_sizes,
        },
        "loss_abs_diff": diff,
        "sequential": sequential,
        "parallel": parallel,
        "speedup_mean": sequential["mean_seconds"] / parallel["mean_seconds"],
        "speedup_median": sequential["median_seconds"] / parallel["median_seconds"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


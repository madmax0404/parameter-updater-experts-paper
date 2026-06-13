"""Check sequential vs parallel gated-lerp scan equivalence.

This script is intentionally standalone and does not load OLMoE. It verifies
the recurrence algebra that underlies the experiment-28 parallel scan claim,
including a nonzero initial state and gradients.
"""

from __future__ import annotations

import argparse
import json

import torch

from parallel_scan import parallel_gated_lerp_scan, sequential_gated_lerp_scan


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


def run_check(args: argparse.Namespace) -> dict[str, float | str | int | bool]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    beta_logits = torch.randn(
        args.batch_size, args.seq_len, device=device, requires_grad=True
    )
    values = torch.randn(
        args.batch_size, args.seq_len, args.features, device=device, requires_grad=True
    )
    initial_state = torch.randn(
        args.batch_size, args.features, device=device, requires_grad=True
    )

    beta = torch.sigmoid(beta_logits)
    sequential = sequential_gated_lerp_scan(beta, values, initial_state)
    sequential_loss = sequential.square().mean()
    sequential_grads = torch.autograd.grad(
        sequential_loss, (beta_logits, values, initial_state), retain_graph=True
    )

    parallel = parallel_gated_lerp_scan(beta, values, initial_state)
    parallel_loss = parallel.square().mean()
    parallel_grads = torch.autograd.grad(
        parallel_loss, (beta_logits, values, initial_state)
    )

    metrics = {
        "device": str(device),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "features": args.features,
        "forward_max_abs_diff": max_abs_diff(sequential, parallel),
        "loss_abs_diff": abs(float(sequential_loss.detach().cpu()) - float(parallel_loss.detach().cpu())),
        "grad_beta_logits_max_abs_diff": max_abs_diff(
            sequential_grads[0], parallel_grads[0]
        ),
        "grad_values_max_abs_diff": max_abs_diff(sequential_grads[1], parallel_grads[1]),
        "grad_initial_state_max_abs_diff": max_abs_diff(
            sequential_grads[2], parallel_grads[2]
        ),
    }
    metrics["passed"] = all(
        float(metrics[key]) <= args.tolerance
        for key in (
            "forward_max_abs_diff",
            "loss_abs_diff",
            "grad_beta_logits_max_abs_diff",
            "grad_values_max_abs_diff",
            "grad_initial_state_max_abs_diff",
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=11)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    metrics = run_check(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


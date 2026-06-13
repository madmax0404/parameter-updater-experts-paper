"""Standalone gated-lerp scan utilities.

These functions isolate the recurrence used by experiment 28:

    S_t = (1 - beta_t) * S_{t-1} + beta_t * V_t

The parallel version computes the same sequence of states with a lower
triangular decay matrix and batched matrix multiplication. It intentionally
does not implement TBPTT detach semantics; compare it to the sequential version
with detach_interval=0 for mathematical equivalence.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _check_inputs(
    beta: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    if beta.ndim != 2:
        raise ValueError(f"beta must have shape [B, S], got {tuple(beta.shape)}")
    if values.ndim < 3:
        raise ValueError(
            f"values must have shape [B, S, ...], got {tuple(values.shape)}"
        )
    if beta.shape[:2] != values.shape[:2]:
        raise ValueError(
            "beta and values must share batch and sequence dimensions: "
            f"{tuple(beta.shape[:2])} vs {tuple(values.shape[:2])}"
        )
    if initial_state is not None and initial_state.shape != values.shape[:1] + values.shape[2:]:
        raise ValueError(
            "initial_state must have shape [B, ...] matching values[:, 0]: "
            f"expected {tuple(values.shape[:1] + values.shape[2:])}, "
            f"got {tuple(initial_state.shape)}"
        )


def sequential_gated_lerp_scan(
    beta: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    detach_interval: int = 0,
) -> torch.Tensor:
    """Run the gated-lerp recurrence with an explicit Python time loop.

    Args:
        beta: Update gates with shape [B, S].
        values: Candidate states with shape [B, S, ...].
        initial_state: Optional starting state with shape [B, ...].
        detach_interval: If positive, detach the recurrent state every K tokens.
            This matches the training convenience used by the older sequential
            experiment, but it intentionally changes gradients.

    Returns:
        All recurrent states with shape [B, S, ...].
    """
    _check_inputs(beta, values, initial_state)
    batch, seq_len = beta.shape
    state = (
        values.new_zeros(values.shape[:1] + values.shape[2:])
        if initial_state is None
        else initial_state
    )

    expand_shape = (batch,) + (1,) * (values.ndim - 2)
    states = []
    for pos in range(seq_len):
        if detach_interval > 0 and pos > 0 and pos % detach_interval == 0:
            state = state.detach()
        beta_t = beta[:, pos].reshape(expand_shape)
        state = (1 - beta_t) * state + beta_t * values[:, pos]
        states.append(state)

    return torch.stack(states, dim=1)


def parallel_gated_lerp_scan(
    beta: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the gated-lerp recurrence with a closed-form parallel scan.

    For S_t = a_t S_{t-1} + b_t, where a_t = 1 - beta_t and
    b_t = beta_t * V_t:

        S_t = sum_{j=0}^t W[t, j] * b_j
        W[t, j] = prod_{s=j+1}^t a_s

    The W matrix is O(S^2), which is acceptable for the short-sequence
    mechanism-isolation benchmark but not a long-context production scan.
    """
    _check_inputs(beta, values, initial_state)
    batch, seq_len = beta.shape
    original_shape = values.shape

    beta_work = beta.float()
    values_work = values.float()
    flat_values = values_work.reshape(batch, seq_len, -1)

    a = 1 - beta_work
    log_a = torch.log(a.clamp(min=eps))
    log_cum = torch.cumsum(log_a, dim=1)

    log_diff = log_cum.unsqueeze(2) - log_cum.unsqueeze(1)
    weights = torch.exp(log_diff).tril()

    b = beta_work.unsqueeze(-1) * flat_values
    flat_states = torch.bmm(weights, b)

    if initial_state is not None:
        decay_init = torch.exp(log_cum).unsqueeze(-1)
        flat_init = initial_state.float().reshape(batch, 1, -1)
        flat_states = flat_states + decay_init * flat_init

    return flat_states.reshape(original_shape)


def exp28_lora_factor_sizes(
    *,
    d_model: int = 2048,
    d_ff: int = 1024,
    rank: int = 4,
) -> tuple[int, int, int, int]:
    """Return flattened factor sizes used by the exp-28 DeltaNet-LoRA state."""
    return (
        d_model * rank,  # A1: [D, r]
        rank * d_model,  # B1: [r, 2*d_ff], equal to D for OLMoE-1B-7B
        d_ff * rank,  # A2: [d_ff, r]
        rank * d_model,  # B2: [r, D]
    )


def total_feature_count(factor_sizes: Sequence[int]) -> int:
    """Total flattened state width across factor groups."""
    return int(sum(factor_sizes))


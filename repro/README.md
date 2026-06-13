# DeltaNet-LoRA Parallel Scan Reproduction

This directory contains a small reproduction bundle for the 2.77x parallel-scan
claim in the preprint. It is intentionally narrower than the full private
research workspace.

## What This Reproduces

- The gated-lerp recurrence algebra used by DeltaNet-LoRA:
  `S_t = (1 - beta_t) * S_{t-1} + beta_t * V_t`.
- Numerical equivalence between the sequential recurrence and the closed-form
  parallel scan, including gradients and a restored initial state.
- A recurrence-only microbenchmark comparing a Python time loop against the
  PyTorch `cumsum`/`exp`/`bmm` scan.
- The reported exp-27/exp-28 end-to-end numbers as a static JSON record.

## What This Does Not Reproduce

- It does not train or load OLMoE-1B-7B.
- It does not reproduce the full experiment-28 end-to-end training run.
- It does not publish the full private research repo.
- It does not claim a universal 2.77x speedup for all sequence lengths or model
  settings. The reported training speedup is for the short-sequence
  sliding-window mechanism-isolation benchmark at `T=120`.

## Files

| File | Purpose |
|---|---|
| `parallel_scan.py` | Standalone sequential and parallel gated-lerp scan functions. |
| `parallel_scan_equivalence.py` | Forward/gradient equivalence check. |
| `benchmark_parallel_scan.py` | Recurrence-only microbenchmark. |
| `exp27_exp28_reported_results.json` | Static record of the reported private exp-27/exp-28 results. |

## Quick Check

From the repository root:

```bash
python repro/parallel_scan_equivalence.py
```

Expected result: JSON with `"passed": true`. Typical CPU differences are around
`1e-7` for forward values and `1e-8` for gradients.

## Microbenchmark

Small default benchmark, suitable as a smoke test:

```bash
python repro/benchmark_parallel_scan.py
```

GPU benchmark using the experiment-28 LoRA factor dimensions:

```bash
python repro/benchmark_parallel_scan.py --device cuda --exp28-shape --warmup 3 --iters 10
```

The benchmark reports recurrence-only speed. It is useful for inspecting the
parallel scan bottleneck directly, but it is not the same measurement as the
full OLMoE training time in the paper.

## Reported End-to-End Result

The saved private experiment summaries report:

- Sequential exp 27: `6967.61 s` for 3000 steps, `2.3225 s/step`.
- Parallel-scan exp 28: `2517.77 s` for 3000 steps, `0.8393 s/step`.
- Derived ratio: `2.767x`, rounded to `2.77x`.

See `exp27_exp28_reported_results.json` for the exact values and caveats.


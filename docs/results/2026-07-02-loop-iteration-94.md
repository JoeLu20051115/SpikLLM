# Loop Iteration 94 - Residual Zero MLP Projector Screen160 In Progress

Date: 2026-07-02

## Purpose

Observed 160-step extension of the residual zero-initialized MLP projector screen from loops92-93.

This is a small-batch candidate gate run only. It is not the requested three-GPU best-baseline run.

## Setup

- tmux session: `loop94_residual_zero_mlp_screen160_gpu2`
- Script: `logs/loop94-residual-zero-mlp-screen160-gpu2-20260702-093510/runner.py`
- Log: `logs/loop94-residual-zero-mlp-screen160-gpu2-20260702-093510/train.log`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `160`
- W&B: disabled

Candidate:

```text
hidden_projector = tensor + 0.1 * Linear2(GELU(Linear1(tensor)))
```

with `Linear2.weight` initialized to zero.

## Current State At Record Time

The run is still in progress. The base half has started and had reached step35 in the local log:

- base step25 hard/soft: `7.7194 / 4.2881`;
- base step30 hard/soft: `7.8054 / 4.4754`;
- base step35 hard/soft: `7.7613 / 4.2908`.

No candidate result or final summary has been written yet.

Decision at record time: continue observing. Loops92-93 did not show a clear small-batch win for this candidate, so loop94 must show a materially stronger result before any baseline update or full run is justified.

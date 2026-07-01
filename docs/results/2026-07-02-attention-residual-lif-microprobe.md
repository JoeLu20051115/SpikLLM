# Attention Residual LIF Microprobe - No Loop Promotion

Date: 2026-07-02

## Purpose

The current block computes:

`attention_residual = hidden_state + attended`

and feeds `LayerNorm(attention_residual)` into the SFFN. Since `hidden_state` and `attended` are spike outputs after the first block, the sum can contain non-binary values such as `2`. This is a possible mismatch with the paper's fully binary spike-driven framing.

This runtime-only probe inserts an extra LIF after the attention residual before the SFFN:

`attention_residual = LIF(hidden_state + attended)`

No repository code was changed for the probe.

## Setup

- Code baseline: `main` at `4082798`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming with prefetched micro-batches reused across variants
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## 20-Step Probe

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| attention residual LIF | 7.9405 | 4.4827 | 7.8832 | 4.4696 | -0.0159 / -0.0201 | -0.0119 / -0.0077 |

The 20-step screen was positive but small, so it was extended to 40 steps.

## 40-Step Probe

The 40-step probe used 640 prefetched micro-batches. Warmup differs from the 20-step probe because the scheduler uses the configured max step count, so compare only within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6659 | 4.3908 | 7.7851 | 4.3733 | 7.7640 | 4.3830 |
| attention residual LIF | 7.6711 | 4.3883 | 7.7912 | 4.3666 | 7.7617 | 4.3741 |

Deltas for attention residual LIF vs base:

- Step 40 hard/soft: +0.0052 / -0.0025
- Last10 hard/soft: +0.0062 / -0.0067
- Last20 hard/soft: -0.0023 / -0.0089

## Decision

Do not create loop36 from this candidate. The 20-step screen was slightly positive, and the 40-step screen lowered soft loss, but hard loss did not clearly beat base at the final step or over the last-10 window. This fails the current rule that a candidate must be clearly better on the fixed small screen before an official 80-step gate.

Keep loop16 as the current best baseline.

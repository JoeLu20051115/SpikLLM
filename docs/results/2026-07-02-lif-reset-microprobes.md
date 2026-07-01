# LIF Reset Microprobes - No Loop Promotion

Date: 2026-07-02

## Purpose

The paper's LIF equation uses subtractive reset:

`U_t = I_t + beta U_{t-1} - S_{t-1} U_thr`

The current SpikingJelly `LIFNode` construction does not set `v_reset=None`, so it uses the default hard reset `v_reset=0.0`. This differs from the paper equation and from the local teacher rate encoder used in SpAD, which subtracts the threshold after a spike.

Two runtime-only probes tested whether this should become loop36:

1. set all model LIF nodes to soft/subtractive reset via `v_reset=None`;
2. set all model LIF nodes to `v_reset=None` and `detach_reset=False`.

No repository code was changed for these probes.

## Setup

- Code baseline: `main` at `b76ae35`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming with prefetched micro-batches reused across variants
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Soft Reset 20-Step Probe

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| soft reset | 7.9367 | 4.4892 | 7.8797 | 4.4735 | -0.0197 / -0.0136 | -0.0155 / -0.0037 |

The 20-step screen was positive but small, so it was extended to 40 steps.

## Soft Reset 40-Step Probe

The 40-step probe used 640 prefetched micro-batches. Warmup differs from the 20-step probe because the scheduler uses the configured max step count, so compare only within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6659 | 4.3908 | 7.7851 | 4.3733 | 7.7640 | 4.3830 |
| soft reset | 7.6748 | 4.3583 | 7.7846 | 4.3438 | 7.7675 | 4.3645 |

Deltas for soft reset vs base:

- Step 40 hard/soft: +0.0089 / -0.0324
- Last10 hard/soft: -0.0004 / -0.0295
- Last20 hard/soft: +0.0035 / -0.0185

Decision: mixed. Soft loss improved, but hard loss did not clearly improve at step 40 or over the last 20 steps.

## Soft Reset + Reset-Gradient 20-Step Probe

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| soft reset + `detach_reset=False` | 7.9217 | 4.4846 | 7.8576 | 4.4835 | -0.0348 / -0.0182 | -0.0375 / +0.0062 |

Decision: fail the micro-screen. Final hard/soft improved, but last10 soft was worse than base, so this is not a clear small-screen win and should not be extended to 40 or 80 steps.

## Overall Decision

Do not create loop36 from the reset variants. Soft reset is paper-motivated and improves soft loss, but it does not clearly improve both primary hard and soft metrics under the fixed streaming micro-screen.

Keep loop16 as the current best baseline.

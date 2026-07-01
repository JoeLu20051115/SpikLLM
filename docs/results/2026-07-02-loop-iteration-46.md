# Loop Iteration 46 - Soft Reset plus Constant Mean Input Screen

Date: 2026-07-02

## Hypothesis

The paper's LIF dynamics use a constant input current with subtractive reset:

`U_t = I_t + beta U_{t-1} - S_{t-1} U_thr`

Current `main` still uses a manual temporal input ramp and SpikingJelly's default hard reset. Loop32 tested mean-preserving constant input alone and failed the 80-step gate. The reset microprobe showed soft/subtractive reset alone was mixed. This screen tests the coherent LIF-dynamics combination:

- replace the input ramp with mean-preserving constant temporal input, `(T + 1) / (2T)`;
- set model LIF nodes to soft/subtractive reset via `v_reset=None`.

This was run as a runtime monkeypatch only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, prefetched micro-batches reused by base and candidate
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## 20-Step Screen

The first screen used 320 prefetched micro-batches.

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% |
| soft reset + constant mean input | 7.9121 | 4.4815 | 7.8643 | 4.4757 | 6.07% | 7.34% |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0334 / -0.0263
- Last10 hard/soft: -0.0358 / -0.0043

The step-20 result was positive, but the last10 soft gain was small, so the candidate was extended to 40 steps before considering an official 80-step gate.

## 40-Step Confirmation

The 40-step confirmation used 640 prefetched micro-batches. Warmup differs from the 20-step screen because scheduler horizon equals the probe length, so compare within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6693 | 4.3722 | 7.7956 | 4.3550 | 7.7669 | 4.3695 |
| soft reset + constant mean input | 7.6713 | 4.3718 | 7.7808 | 4.3491 | 7.7688 | 4.3674 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0020 / -0.0004
- Last10 hard/soft: -0.0148 / -0.0059
- Last20 hard/soft: +0.0019 / -0.0021

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 2.05% | 3.23% | 4146.8 | -4.9881 | 36.00% |
| soft reset + constant mean input | 2.15% | 3.42% | 4088.1 | -4.9122 | 38.55% |

## Decision

Do not promote loop46. The candidate improves last10 hard/soft and several secondary diagnostics, but it does not clearly beat the matched base at step 40 or over the last20 hard window. This is below the required bar for an 80-step gate.

Keep loop16 as the official small-gate metric baseline.

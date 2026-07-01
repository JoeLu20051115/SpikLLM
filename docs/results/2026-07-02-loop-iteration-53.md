# Loop Iteration 53 - Fixed Readout Scale Screen

Date: 2026-07-02

## Hypothesis

`readout_log_scale` is an extra trainable output-scale parameter introduced by earlier engineering loops. The paper does not specify a trainable logit-scale parameter after the final LN/head. This screen tests whether freezing readout scale at exactly 1.0 improves the hard/soft output path.

This was run as a runtime monkeypatch only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- Candidate: set `student.readout_log_scale = 0` and `requires_grad=False`.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, first 640 prefetched micro-batches reused by base and candidate
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 0.9885 |
| fixed readout scale | 7.6836 | 4.4124 | 7.8102 | 4.4098 | 7.7904 | 4.4232 | 1.0000 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0126 / -0.0169
- Last10 hard/soft: -0.0247 / -0.0376
- Last20 hard/soft: -0.0027 / -0.0069

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% |
| fixed readout scale | 4.50% | 5.48% | 4093.7 | -5.1691 | 33.64% |

## Decision

Do not promote loop53. Fixing readout scale improves recent-window losses, soft loss, token accuracy, teacher agreement, and target rank, but step40 hard loss and target margin regress. This is not a clear small-screen win and does not justify an 80-step gate.

Keep loop16 as the official small-gate metric baseline. The trainable readout scale is not removed from source based on this result.

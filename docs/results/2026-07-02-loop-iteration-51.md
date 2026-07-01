# Loop Iteration 51 - Official-Horizon Recheck for Strong Short Screens

Date: 2026-07-02

## Hypothesis

Several recent 20-step screens used a scheduler horizon equal to the screen length. That differs from the official 80-step small gate, whose warmup/cosine horizon is 80 steps. Loop47 in particular showed a strong 20-step win but reversed at 40 steps.

This loop rechecks the strongest recent candidates using a 40-step run with an 80-step scheduler horizon:

- `no_internal_pre_ln`
- `soft_reset_const_mean_input`

Both candidates were runtime monkeypatches only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, first 640 prefetched micro-batches reused by all variants
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 |
| no internal pre-LN | 7.6802 | 4.4166 | 7.8543 | 4.4056 | 7.8023 | 4.4065 |
| soft reset + constant mean input | 7.6777 | 4.4010 | 7.8421 | 4.3996 | 7.8045 | 4.4036 |

Deltas for candidates vs base:

| Variant | Step40 hard | Step40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no internal pre-LN | +0.0091 | -0.0127 | +0.0194 | -0.0418 | +0.0092 | -0.0235 |
| soft reset + constant mean input | +0.0067 | -0.0284 | +0.0072 | -0.0478 | +0.0114 | -0.0265 |

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% |
| no internal pre-LN | 3.52% | 5.19% | 3944.5 | -5.1930 | 31.65% |
| soft reset + constant mean input | 4.50% | 5.48% | 4235.2 | -5.1233 | 37.28% |

## Decision

Do not promote either candidate to an 80-step gate. With the official 80-step scheduler horizon, both candidates consistently improve soft loss but regress hard loss at step 40 and across the recent hard-loss windows.

This confirms that short-horizon 20-step wins are not sufficient evidence for promotion. Future pre-screens should use `scheduler_max_steps=80` when the target decision is an 80-step official small gate.

Keep loop16 as the official small-gate metric baseline.

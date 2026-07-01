# Loop Iteration 54 - Fixed Readout plus SFSA Output LIF Bypass

Date: 2026-07-02

## Hypothesis

Loop52 and loop53 each improved recent-window losses but failed the step-level hard/soft gate. Both candidates are output-path changes that move closer to the paper:

- bypass the extra SFSA pre-projection `attn_out_lif`;
- freeze the extra trainable readout scale at 1.0.

This screen tests whether combining them converts recent-window improvements into a clear hard/soft step-level gain.

This was run as a runtime monkeypatch only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- Candidate:
  - replace each `block.attention.attn_out_lif` with `Identity`;
  - set `student.readout_log_scale = 0` and `requires_grad=False`.
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

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 |
| fixed readout + bypass `attn_out_lif` | 7.6783 | 4.4059 | 7.8159 | 4.4099 | 7.8006 | 4.4334 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0073 / -0.0235
- Last10 hard/soft: -0.0190 / -0.0375
- Last20 hard/soft: +0.0075 / +0.0033

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Readout | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 0.9885 | 33.16% |
| fixed readout + bypass `attn_out_lif` | 4.70% | 5.38% | 4121.0 | -5.0537 | 1.0000 | 37.19% |

## Decision

Do not promote loop54. The combined output-path candidate improves soft loss, last10 hard/soft, token accuracy, teacher agreement, target rank, and target margin, but step40 hard and last20 hard/soft regress. This is not a clear small-screen win and does not justify an 80-step gate.

Keep loop16 as the official small-gate metric baseline.

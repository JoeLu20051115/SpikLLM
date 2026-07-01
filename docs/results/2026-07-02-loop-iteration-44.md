# Loop Iteration 44 - Soft Reset plus Attention Residual LIF Micro-Screen

Date: 2026-07-02

## Hypothesis

Two prior paper-motivated dynamics probes were individually weak but directionally interesting:

- soft/subtractive LIF reset (`v_reset=None`) improved soft loss but did not clearly improve hard loss;
- inserting a LIF after the attention residual improved soft loss slightly but did not clearly improve hard loss.

This loop tested their coherent combination as a runtime monkeypatch only:

- set model LIF nodes to soft/subtractive reset;
- insert an attention-residual LIF before the SFFN input.

No source code was changed for the run.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, first 640 prefetched micro-batches reused by base and candidate
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6693 | 4.3722 | 7.7956 | 4.3550 | 7.7669 | 4.3695 |
| soft reset + attention residual LIF | 7.6805 | 4.3840 | 7.8007 | 4.3641 | 7.7922 | 4.3962 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0112 / +0.0117
- Last10 hard/soft: +0.0051 / +0.0090
- Last20 hard/soft: +0.0254 / +0.0266

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 2.05% | 3.23% | 4146.8 | -4.9881 | 36.00% |
| soft reset + attention residual LIF | 3.72% | 3.62% | 4096.3 | -4.9614 | 28.80% |

## Decision

Do not promote loop44. The candidate improves some secondary output diagnostics, but all primary hard/soft metrics are worse than the matched base over the final step and recent windows. This fails the small-screen rule, so no 80-step gate or full run is justified.

Keep loop16 as the official small-gate metric baseline.

# Loop Iteration 52 - Bypass SFSA Pre-Projection Output LIF

Date: 2026-07-02

## Hypothesis

The SFSA figure in the paper shows:

`SpikeAttention @ SpikeV -> IntegerOut -> Linear -> SN`

Current `main` applies an extra LIF before the output projection:

`SpikeAttention @ SpikeV -> attn_out_lif -> out_proj -> out_lif`

This screen tests whether bypassing `attn_out_lif` preserves integer attention-output magnitude and improves output learning, while keeping the projection-after-SN path (`out_lif`) unchanged.

This was run as a runtime monkeypatch only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- Candidate: replace each `block.attention.attn_out_lif` with `Identity`.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, prefetched micro-batches reused by base and candidate
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## 40-Step Screen

The first screen used 640 prefetched micro-batches.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 |
| bypass `attn_out_lif` | 7.6723 | 4.4010 | 7.8164 | 4.4105 | 7.7866 | 4.4218 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0013 / -0.0284
- Last10 hard/soft: -0.0185 / -0.0369
- Last20 hard/soft: -0.0065 / -0.0083

Because recent hard/soft windows and soft loss improved, and step40 hard was effectively tied, the candidate was extended to a matched 80-step official small gate.

## 80-Step Gate

The 80-step gate used 1280 prefetched micro-batches.

| Variant | Step 80 hard | Step 80 soft | Last25 hard | Last25 soft | Last40 hard | Last40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.4572 | 4.5187 | 7.7819 | 4.2997 | 7.7818 | 4.3152 |
| bypass `attn_out_lif` | 7.4633 | 4.5450 | 7.7741 | 4.2951 | 7.7787 | 4.3145 |

Deltas for candidate vs base:

- Step 80 hard/soft: +0.0061 / +0.0263
- Last25 hard/soft: -0.0079 / -0.0046
- Last40 hard/soft: -0.0030 / -0.0007

Secondary diagnostics at step 80:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 5.38% | 8.02% | 4138.2 | -4.6461 | 27.34% |
| bypass `attn_out_lif` | 4.70% | 6.95% | 4184.6 | -4.6557 | 31.32% |

## Decision

Do not promote loop52. Bypassing `attn_out_lif` improves recent-window hard/soft means slightly, but the official step80 hard/soft losses, token accuracy, teacher agreement, target rank, and target margin all regress. This is not a clear small-gate win and does not satisfy the continuation rule.

Keep loop16 as the official small-gate metric baseline. Do not change the source implementation from this result alone.

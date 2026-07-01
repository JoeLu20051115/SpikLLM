# Loop Iteration 47 - Remove Internal Pre-LayerNorm Screen

Date: 2026-07-02

## Hypothesis

Figure 2 in the paper depicts the causal spiking decoder block as a spike-driven stack:

`SN -> SFSA -> residual -> SN -> FC -> SN -> FC -> SN -> residual -> time accumulation -> LN/head`

Current `main` follows an OPT-style pre-LN implementation inside each block:

`LayerNorm -> SFSA -> residual -> LayerNorm -> SFFN -> final LIF`

This screen tests whether the internal pre-LayerNorms are suppressing the paper-style spike-driven path. The candidate sets `attention_norm` and `mlp_norm` to `Identity` in every block while leaving SFSA, SFFN, SpAD losses, optimizer, data, labels, teacher logits, and readout unchanged.

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

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 45.11% |
| no internal pre-LN | 7.9247 | 4.4672 | 7.8572 | 4.4320 | 6.26% | 7.14% | 52.00% |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0207 / -0.0406
- Last10 hard/soft: -0.0429 / -0.0480

This was a clear positive 20-step screen, so the candidate was extended to 40 steps before considering an official 80-step gate.

## 40-Step Confirmation

The 40-step confirmation used 640 prefetched micro-batches. Warmup differs from the 20-step screen because scheduler horizon equals the probe length, so compare within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6693 | 4.3722 | 7.7956 | 4.3550 | 7.7669 | 4.3695 |
| no internal pre-LN | 7.6926 | 4.3823 | 7.7956 | 4.3576 | 7.7598 | 4.3718 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0233 / +0.0101
- Last10 hard/soft: +0.0000 / +0.0026
- Last20 hard/soft: -0.0070 / +0.0023

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 2.05% | 3.23% | 4146.8 | -4.9881 | 36.00% |
| no internal pre-LN | 3.23% | 4.60% | 3877.2 | -5.0168 | 33.51% |

## Decision

Do not promote loop47. Removing the internal pre-LayerNorms produced the strongest positive 20-step screen so far, but it failed the 40-step confirmation on step40 hard/soft and last10 soft. The last20 hard and target-rank improvements are not enough to justify an 80-step gate.

Keep loop16 as the official small-gate metric baseline. Future block-order candidates should account for the 20-step-to-40-step reversal rather than simply removing both norms.

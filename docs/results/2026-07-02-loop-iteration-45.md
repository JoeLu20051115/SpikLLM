# Loop Iteration 45 - Same-Dimension SFA MLP+LayerNorm Screen

Date: 2026-07-02

## Hypothesis

The paper's SFA appendix maps student features as:

`H'_SNN = LayerNorm(MLP(mean_t H_SNN(t)))`

Current loop16 code uses an identity same-dimension hidden projector. Loop30 tested a hidden-only same-dimension LayerNorm projector, but not the full MLP+LayerNorm mapping described by the paper. This loop screens the narrower paper-fidelity variant:

- keep same-dimension EA projector as identity;
- replace only the hidden/SFA projector with a same-dimension MLP+LayerNorm module.

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

The first 20-step screen used 320 prefetched micro-batches.

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 feature |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 0.5397 |
| same-dim SFA MLP+LN | 7.9246 | 4.4802 | 7.8928 | 4.4796 | 0.8213 |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0209 / -0.0276
- Last10 hard/soft: -0.0072 / -0.0005

The final-step result was positive, but the last10 soft gain was tiny, so the candidate was extended to 40 steps before considering an official 80-step gate.

## 40-Step Confirmation

The 40-step confirmation used 640 prefetched micro-batches. Warmup differs from the 20-step screen because scheduler horizon equals the probe length, so compare within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Step40 feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6693 | 4.3722 | 7.7956 | 4.3550 | 7.7669 | 4.3695 | 0.4815 |
| same-dim SFA MLP+LN | 7.6716 | 4.3807 | 7.7989 | 4.3577 | 7.7651 | 4.3665 | 0.7907 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0023 / +0.0085
- Last10 hard/soft: +0.0034 / +0.0027
- Last20 hard/soft: -0.0017 / -0.0030

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 2.05% | 3.23% | 4146.8 | -4.9881 | 36.00% |
| same-dim SFA MLP+LN | 2.05% | 3.23% | 4064.1 | -4.9707 | 43.09% |

## Decision

Do not promote loop45. The full SFA MLP+LayerNorm projector is paper-motivated and showed a positive 20-step final-step signal, but it failed the 40-step confirmation on step40 hard/soft and last10 hard/soft. The small last20 gain and slightly better rank/margin are not enough to justify an 80-step gate.

Keep loop16 as the official small-gate metric baseline.

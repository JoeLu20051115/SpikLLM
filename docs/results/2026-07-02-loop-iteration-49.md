# Loop Iteration 49 - No Pre-LN plus Attention Residual LIF Screen

Date: 2026-07-02

## Hypothesis

Loop47 removed both internal pre-LayerNorms and produced a strong 20-step signal, but failed the 40-step confirmation. Figure 2 in the paper also shows a spiking neuron after the attention residual before the feed-forward path. This screen tests whether adding that residual spike stage stabilizes the no-pre-LN block path:

- set `attention_norm` and `mlp_norm` to `Identity`;
- insert `attention_residual_lif(hidden_state + attended)` before SFFN.

This was run as a runtime monkeypatch only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused by base and candidate
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 5081.1 | -5.0093 | 45.11% |
| no pre-LN + residual LIF | 7.9599 | 4.4733 | 7.8944 | 4.4417 | 4.70% | 6.75% | 5142.9 | -5.1439 | 45.61% |

Deltas for candidate vs base:

- Step 20 hard/soft: +0.0145 / -0.0345
- Last10 hard/soft: -0.0057 / -0.0383

## Decision

Do not promote loop49. The candidate substantially improves soft loss and last10 soft loss, but the primary hard loss is worse at step 20 and target rank/margin also regress. This is not a clear small-screen win, so no 40-step or 80-step gate is justified.

Keep loop16 as the official small-gate metric baseline.

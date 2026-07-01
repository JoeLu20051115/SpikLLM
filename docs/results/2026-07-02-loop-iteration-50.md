# Loop Iteration 50 - Input Spike plus No Pre-LN Screen

Date: 2026-07-02

## Hypothesis

Figure 2 shows time-repeated input embeddings passing through a spiking neuron before the causal spiking decoder block. Loop33 tested input spike encoding alone and failed the official gate, while loop47 showed an early signal from removing internal pre-LayerNorms. This screen tests the adjacent Figure-2-aligned combination:

- spike-encode the per-step input current before the block stack;
- set block `attention_norm` and `mlp_norm` to `Identity`;
- keep the existing input ramp and keep `embedding_states` as the analog current for EA.

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
| input spike + no pre-LN | 7.9975 | 4.4865 | 7.9479 | 4.4460 | 6.26% | 7.73% | 5174.4 | -5.3157 | 54.49% |

Deltas for candidate vs base:

- Step 20 hard/soft: +0.0520 / -0.0213
- Last10 hard/soft: +0.0479 / -0.0340

## Decision

Do not promote loop50. The candidate improves soft loss, token accuracy, and teacher agreement, but hard loss, last10 hard, target rank, and target margin regress sharply. This is not a clear small-screen win, so no 40-step or 80-step gate is justified.

Keep loop16 as the official small-gate metric baseline.

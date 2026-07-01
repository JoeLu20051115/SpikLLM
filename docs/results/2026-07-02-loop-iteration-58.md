# Loop Iteration 58 - Unscaled EA Dynamics Screen

Date: 2026-07-02

## Hypothesis

Loop57 showed that a broad paper-dynamics bundle recovered at the final step but had a worse early regime. Loop58 tests a narrower interaction:

- use raw teacher-space token-plus-position embeddings for the embedding-alignment target instead of the scaled current embedding states;
- set model LIF nodes to soft/subtractive reset via `v_reset=None`;
- replace the temporal input ramp with a mean-preserving constant input scale, `(T + 1) / (2T)`;
- keep the existing internal block pre-LN path, unlike loop57.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming; base and candidate use matched seed and streaming order
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

- Base run: `loop58-unscaled-ea-soft-reset-const-screen20-base`
- Base W&B: `wandb/offline-run-20260702_065049-q5p8z6qh/run-q5p8z6qh.wandb`
- Candidate run: `loop58-unscaled-ea-soft-reset-const-screen20-candidate`
- Candidate W&B: `wandb/offline-run-20260702_065137-8w441x5l/run-8w441x5l.wandb`

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 5081.1 | -5.0093 | 45.11% |
| candidate | 7.9220 | 4.4878 | 7.8680 | 4.4901 | 5.09% | 5.68% | 5184.6 | -4.9265 | 43.05% |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0235 / -0.0200
- Last10 hard/soft: -0.0321 / +0.0100
- Step 20 token accuracy / teacher agreement: +1.08 pp / -0.39 pp
- Step 20 target rank / margin: +103.5 / +0.0828
- Step 20 spike rate: -2.05 pp

Full 20-step means:

| Variant | Mean hard | Mean soft | Mean token acc | Mean teacher agreement | Mean target rank | Mean target margin | Mean spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 8.5097 | 4.9765 | 3.62% | 6.21% | 6422.7 | -5.4513 | 51.28% |
| candidate | 8.4222 | 4.9550 | 3.81% | 5.38% | 6269.3 | -5.2931 | 51.46% |

Full-window candidate deltas:

- Mean hard/soft: -0.0875 / -0.0216
- Mean token accuracy / teacher agreement: +0.19 pp / -0.83 pp
- Mean target rank / margin: -153.4 / +0.1582
- Mean spike rate: +0.18 pp

## Decision

Fail loop58. Do not extend to 40 steps:

- The candidate improves final-step hard/soft and full-20 hard/soft means, but it is not clearly better on the recent window.
- Last10 soft loss is worse than base by +0.0100.
- Teacher agreement is worse at both the final step and across the full 20-step window.
- Final-step target rank is worse despite a better margin.
- Final-step spike rate remains lower than base, so this does not address the under-firing seen in the loop55 matched-geometry comparison.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. The unscaled-EA/soft-reset/constant-input interaction is not a promotion candidate under the small-screen gate.

# Loop Iteration 59 - Temporal Logit Fusion Screen

Date: 2026-07-02

## Hypothesis

Recent loops repeatedly showed small gains in auxiliary losses or one output loss without a stable hard/soft win. Loop59 tests a distinct output-supervision hypothesis from the BPTT/time-step side:

- keep all five SpAD losses and paper weights unchanged;
- keep SFSA, SFA, EA, optimizer, data, labels, teacher logits, and scheduler unchanged;
- replace the output-logit fusion from `LN(mean_t hidden_t) -> lm_head` with `mean_t(lm_head(LN(hidden_t)))`.

This tests whether supervising logits after each time-step representation is more stable than taking logits from the averaged hidden state. This was a runtime monkeypatch only. No source code was changed.

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
- Scheduler horizon equals the 20-step screen length

## Result

- Base run: `loop59-temporal-logit-fusion-screen20-base`
- Base W&B: `wandb/offline-run-20260702_065913-4w33mojq/run-4w33mojq.wandb`
- Candidate run: `loop59-temporal-logit-fusion-screen20-temporal-logits`
- Candidate W&B: `wandb/offline-run-20260702_070033-0a0n3lmp/run-0a0n3lmp.wandb`

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 5081.1 | -5.0093 | 45.11% |
| temporal logits | 7.9381 | 4.5235 | 7.8876 | 4.4958 | 2.94% | 4.89% | 5063.7 | -4.9610 | 43.23% |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0073 / +0.0158
- Last10 hard/soft: -0.0125 / +0.0158
- Step 20 token accuracy / teacher agreement: -1.08 pp / -1.17 pp
- Step 20 target rank / margin: -17.4 / +0.0483
- Step 20 spike rate: -1.88 pp

Full 20-step means:

| Variant | Mean hard | Mean soft | Mean token acc | Mean teacher agreement | Mean target rank | Mean target margin | Mean spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 8.5097 | 4.9765 | 3.62% | 6.21% | 6422.7 | -5.4513 | 51.28% |
| temporal logits | 8.4554 | 4.9037 | 3.36% | 5.60% | 6398.3 | -5.1779 | 50.19% |

Full-window candidate deltas:

- Mean hard/soft: -0.0543 / -0.0729
- Mean token accuracy / teacher agreement: -0.26 pp / -0.61 pp
- Mean target rank / margin: -24.4 / +0.2734
- Mean spike rate: -1.09 pp

## Decision

Fail loop59. Do not extend to 40 steps:

- The candidate improves full-20 hard/soft means, but the final step and recent window are not clearly better.
- Step-20 soft loss regresses by +0.0158.
- Last10 soft loss also regresses by +0.0158.
- Token accuracy and teacher agreement both regress at the final step and over the full 20-step window.
- Spike rate decreases, so this does not address the loop55 under-firing relative to loop14.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Time-step logit fusion is not a promotion candidate under the small-screen gate.

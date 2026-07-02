# Loop Iteration 70 - Teacher-Std Random Embedding Screen

Date: 2026-07-02

## Hypothesis

Loop63's random embedding run was competitive but under-fired relative to the teacher-copy baseline. Loop66 showed that increasing `input_scale` alone did not fix the random branch and caused EA loss to jump. A different scale hypothesis is that the tied LM head itself remains at random `std=0.02`, while the OPT teacher token embedding has a larger `std=0.0554`.

Loop70 tests random initialization with teacher-matched embedding scale, without copying teacher embedding content:

- disable teacher token/position embedding copy, as in loop63;
- reinitialize the student token embedding / tied LM head from `Normal(0, teacher_token_std)`;
- reinitialize the student position embedding from `Normal(0, teacher_position_std)`;
- keep `input_scale=None`, so the model still uses the existing implicit scale `1 / initializer_range = 50`;
- keep SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, and readout unchanged.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before the run
- Teacher token embedding std: 0.0554028
- Teacher position embedding std: 0.0168632

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Random-only reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-only W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Candidate run: `loop70-random-emb-teacher-std-screen40`
- Candidate W&B: `wandb/offline-run-20260702_075742-hhyokk0d/run-hhyokk0d.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 | 8.5686 | 5.1342 |
| random + teacher std | 7.6813 | 4.4115 | 7.8478 | 4.4164 | 7.8407 | 4.4560 | 7.9117 | 4.5666 | 8.7190 | 5.4107 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0103 / -0.0179
- Last10 hard/soft: +0.0129 / -0.0310
- Last20 hard/soft: +0.0476 / +0.0259
- Last25 hard/soft: +0.0967 / +0.1097
- Full40 hard/soft: +0.4632 / +0.6033

Deltas for candidate vs random-only:

- Step 40 hard/soft: +0.0337 / +0.0255
- Last10 hard/soft: +0.0453 / +0.0186
- Last20 hard/soft: +0.0618 / +0.0415
- Last25 hard/soft: +0.0645 / +0.0664
- Full40 hard/soft: +0.1503 / +0.2765

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 1.5517 | 0.4328 | 0.9885 |
| random embedding | 4.50% | 5.48% | 13.60% | 4033.3 | -5.1189 | 25.28% | 1.5378 | 2.2442 | 1.0062 |
| random + teacher std | 4.70% | 5.38% | 13.60% | 4404.1 | -5.1185 | 37.22% | 1.5809 | 1.3067 | 0.9887 |

Auxiliary losses at step 40:

| Variant | EA loss | Attention loss | Feature loss |
| --- | ---: | ---: | ---: |
| base | 2.9220 | 0.5914 | 0.4751 |
| random embedding | 0.5227 | 0.5860 | 0.3801 |
| random + teacher std | 2.9372 | 0.6312 | 0.4929 |

## Decision

Fail loop70. Do not extend to an 80-step gate:

- It is worse than loop63 random-only on every hard/soft metric window.
- It is not a clear win over base: step40 and last10 soft improve, but hard loss regresses, and last20/last25/full40 hard and soft all regress.
- Teacher-std random initialization removes the low EA-loss advantage from loop63 and returns EA loss to the base level.
- Higher spike rate and logit std do not improve target rank, target margin, or recent hard/soft windows.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Random embedding scale is not the bottleneck in this form.


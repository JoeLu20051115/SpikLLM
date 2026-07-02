# Loop Iteration 74 - Token-Random plus Soft Reset Screen

Date: 2026-07-02

## Hypothesis

Loop72 token-random improved step80 hard loss and last25 hard/soft means, but failed promotion because step80 soft remained worse than loop16 and spike rate was low. Soft/subtractive reset is paper-motivated and can increase or reshape firing behavior, but loop69 showed it destabilized random token+position.

Loop74 tests the narrower interaction:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- set all student model LIF nodes to soft/subtractive reset via `v_reset=None`;
- keep SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, readout, and temporal input ramp unchanged.

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
- Candidate changed modules: 96 LIF nodes with `v_reset=None`

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Token-random reference run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Random-both soft-reset reference: `loop69-random-emb-soft-reset-screen40`
- Random-both soft-reset W&B: `wandb/offline-run-20260702_075213-fuu9l4ym/run-fuu9l4ym.wandb`
- Candidate run: `loop74-token-random-soft-reset-screen40`
- Candidate W&B: `wandb/offline-run-20260702_081445-h9d6xo5p/run-h9d6xo5p.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| token-random | 7.6580 | 4.3853 | 7.8217 | 4.4034 | 7.7895 | 4.4061 | 7.8470 | 4.4870 | 8.5789 | 5.1308 |
| random token+position + soft reset | 7.6643 | 4.3798 | 7.8436 | 4.4021 | 7.8188 | 4.4357 | 7.8804 | 4.5153 | 8.5796 | 5.1412 |
| token-random + soft reset | 7.7227 | 4.4233 | 7.8772 | 4.3999 | 7.8312 | 4.4433 | 7.8992 | 4.5337 | 8.6123 | 5.1650 |

Deltas for candidate vs token-random:

- Step 40 hard/soft: +0.0648 / +0.0380
- Last10 hard/soft: +0.0555 / -0.0035
- Last20 hard/soft: +0.0417 / +0.0372
- Last25 hard/soft: +0.0522 / +0.0466
- Full40 hard/soft: +0.0334 / +0.0342

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0517 / -0.0061
- Last10 hard/soft: +0.0423 / -0.0475
- Last20 hard/soft: +0.0382 / +0.0132
- Last25 hard/soft: +0.0842 / +0.0768
- Full40 hard/soft: +0.3566 / +0.3576

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 1.5517 | 0.4328 | 0.9885 |
| token-random | 4.50% | 5.48% | 13.60% | 4228.8 | -5.0888 | 26.32% | 1.5849 | 5.0594 | 1.0052 |
| random token+position + soft reset | 3.62% | 5.48% | 13.41% | 4173.8 | -5.0059 | 40.57% | 1.5317 | 1.8412 | 1.0062 |
| token-random + soft reset | 4.50% | 5.48% | 13.60% | 4545.3 | -5.2351 | 33.55% | 1.4791 | 0.2914 | 1.0054 |

Recent-window diagnostics:

| Variant | Last10 grad norm | Last20 grad norm | Last25 grad norm | Last10 spike rate | Last20 spike rate | Last25 spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token-random | 3.1698 | 2.2583 | 2.2159 | 25.82% | 25.65% | 26.65% |
| token-random + soft reset | 2199.2141 | 1108.5840 | 949.5986 | 33.58% | 33.59% | 32.98% |

## Decision

Fail loop74. Do not extend to an 80-step gate:

- Soft reset increases spike rate, but it worsens token-random on every primary hard/soft window except a tiny last10 soft improvement.
- Step40 hard and soft both regress vs token-random.
- Last20, last25, and full40 hard/soft all regress vs token-random.
- It is not a clear win over base: hard loss regresses in all windows, and last20/last25/full40 soft also regress.
- The recent-window grad norms show severe instability, repeating the random+soft-reset issue seen in loop69.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Soft reset does not solve token-random's 80-step soft-loss gap.


# Loop Iteration 69 - Random Embedding plus Soft Reset Screen

Date: 2026-07-02

## Hypothesis

Loop63 showed that disabling teacher token/position embedding copy is more paper-faithful and competitive, but its 80-step soft loss failed to beat loop16. Prior reset probes showed that soft/subtractive LIF reset (`v_reset=None`) tends to improve soft loss while leaving hard loss mixed.

Loop69 tests the smallest paper-dynamics addition on top of loop63:

- disable teacher token/position embedding copy, as in loop63;
- set all student model LIF nodes to soft/subtractive reset via `v_reset=None`;
- keep the existing temporal input ramp, tied LM head, SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged.

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
- Random-only reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-only W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Candidate run: `loop69-random-emb-soft-reset-screen40`
- Candidate W&B: `wandb/offline-run-20260702_075213-fuu9l4ym/run-fuu9l4ym.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 | 8.5686 | 5.1342 |
| random + soft reset | 7.6643 | 4.3798 | 7.8436 | 4.4021 | 7.8188 | 4.4357 | 7.8804 | 4.5153 | 8.5796 | 5.1412 |

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0068 / -0.0495
- Last10 hard/soft: +0.0087 / -0.0453
- Last20 hard/soft: +0.0257 / +0.0056
- Last25 hard/soft: +0.0654 / +0.0584
- Full40 hard/soft: +0.3238 / +0.3338

Deltas for candidate vs random-only:

- Step 40 hard/soft: +0.0166 / -0.0062
- Last10 hard/soft: +0.0411 / +0.0043
- Last20 hard/soft: +0.0399 / +0.0213
- Last25 hard/soft: +0.0333 / +0.0151
- Full40 hard/soft: +0.0109 / +0.0070

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 0.4328 | 0.9885 |
| random embedding | 4.50% | 5.48% | 13.60% | 4033.3 | -5.1189 | 25.28% | 2.2442 | 1.0062 |
| random + soft reset | 3.62% | 5.48% | 13.41% | 4173.8 | -5.0059 | 40.57% | 1.8412 | 1.0062 |

Window diagnostics:

| Variant | Last10 grad norm | Last20 grad norm | Last25 grad norm | Last10 spike rate | Last20 spike rate | Last25 spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random embedding | 3.8274 | 2.6043 | 2.5942 | 26.67% | 29.97% | 32.26% |
| random + soft reset | 127.9237 | 74.8170 | 66.3208 | 41.09% | 40.15% | 40.46% |

## Decision

Fail loop69. Do not extend to an 80-step gate:

- It improves soft loss vs base and slightly improves step40 soft vs loop63 random-only, but it is worse than random-only on step40 hard.
- It is worse than random-only on all last10, last20, last25, and full40 hard/soft windows.
- Last20 and last25 are not clearly better than base either.
- Token accuracy regresses vs both base and random-only at step40.
- The higher spike rate does not improve output alignment and comes with very large gradient-norm spikes in recent windows.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Random embedding plus soft reset is not a promotion candidate.


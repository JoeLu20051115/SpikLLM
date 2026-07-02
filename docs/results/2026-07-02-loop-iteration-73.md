# Loop Iteration 73 - Token-Random plus Fixed Readout Screen

Date: 2026-07-02

## Hypothesis

Loop72 token-random improved step80 hard loss and last25 hard/soft means, but failed promotion because step80 soft loss remained worse than loop16. Prior fixed-readout screens sometimes improved soft-loss windows. Loop73 tests whether freezing the extra readout scale at 1.0 fixes the token-random soft-loss gap before spending another 80-step gate.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- set `readout_log_scale = 0` and freeze it, so readout scale stays exactly 1.0;
- keep SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, temporal input ramp, and reset behavior unchanged.

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

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Token-random reference run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Candidate run: `loop73-token-random-fixed-readout-screen40`
- Candidate W&B: `wandb/offline-run-20260702_081048-qqv2vu55/run-qqv2vu55.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| token-random | 7.6580 | 4.3853 | 7.8217 | 4.4034 | 7.7895 | 4.4061 | 7.8470 | 4.4870 | 8.5789 | 5.1308 |
| token-random + fixed readout | 7.6723 | 4.4130 | 7.8122 | 4.3886 | 7.7887 | 4.4107 | 7.8462 | 4.4911 | 8.5765 | 5.1325 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0013 / -0.0164
- Last10 hard/soft: -0.0227 / -0.0589
- Last20 hard/soft: -0.0044 / -0.0193
- Last25 hard/soft: +0.0312 / +0.0342
- Full40 hard/soft: +0.3207 / +0.3252

Deltas for candidate vs token-random:

- Step 40 hard/soft: +0.0144 / +0.0277
- Last10 hard/soft: -0.0095 / -0.0148
- Last20 hard/soft: -0.0008 / +0.0046
- Last25 hard/soft: -0.0008 / +0.0040
- Full40 hard/soft: -0.0024 / +0.0017

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 1.5517 | 0.4328 | 0.9885 |
| token-random | 4.50% | 5.48% | 13.60% | 4228.8 | -5.0888 | 26.32% | 1.5849 | 5.0594 | 1.0052 |
| token-random + fixed readout | 4.50% | 5.48% | 15.17% | 4282.3 | -5.1735 | 25.12% | 1.4972 | 1.5562 | 1.0000 |

## Decision

Fail loop73. Do not extend to an 80-step gate:

- It improves last10 hard/soft vs token-random, but step40 hard and soft both regress.
- Last20 soft, last25 soft, and full40 soft all regress vs token-random.
- It is not a clear win over base because step40 hard, last25 hard/soft, and full40 hard/soft regress.
- The lower logit std and fixed readout do not improve target rank or margin.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Fixed readout does not solve token-random's 80-step soft-loss gap.


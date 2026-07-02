# Loop Iteration 80 - Token-Random plus Teacher Final LayerNorm and Fixed Readout Screen

Date: 2026-07-02

## Hypothesis

Loop79 showed that teacher final LayerNorm initialization is better than freezing the final LayerNorm, but still does not beat token-random at step20. Loop80 tests whether pairing that trainable teacher final-LN initialization with a fixed readout scale removes the loop77 scalar readout conflict without losing the useful final-LN adaptation.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- copy the teacher final LayerNorm weights and bias into `student.final_layer_norm`;
- keep the final LayerNorm trainable;
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
- Optimizer steps: 20
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before the run

## Runs

- Base reference run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B first20: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Token-random reference run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B first20: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Loop79 reference: `loop79-token-random-teacher-finalnorm-screen20`
- Loop79 W&B: `wandb/offline-run-20260702_083626-d7mdn99h/run-d7mdn99h.wandb`
- Candidate run: `loop80-token-random-teacher-finalnorm-fixed-readout-screen20`
- Candidate W&B: `wandb/offline-run-20260702_083902-np5y5vx4/run-np5y5vx4.wandb`
- Candidate log: `logs/loop80-token-random-teacher-finalnorm-fixed-readout-screen20/train.log`

## Result

All reference rows below use the same first 20 optimizer steps.

| Variant | Step20 hard | Step20 soft | Last10 hard | Last10 soft | Full20 hard | Full20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 7.9879 | 4.5508 | 7.9978 | 4.6208 | 8.7184 | 5.1846 |
| token-random first20 | 7.9733 | 4.6164 | 8.4735 | 5.1399 | 9.3683 | 5.8556 |
| token-random + teacher final LN | 7.9741 | 4.6245 | 8.4227 | 5.1111 | 9.3324 | 5.8423 |
| token-random + teacher final LN + fixed readout | 7.9878 | 4.6467 | 8.4469 | 5.1335 | 9.3459 | 5.8541 |

Deltas for candidate vs loop79:

- Step20 hard/soft: +0.0138 / +0.0222
- Last10 hard/soft: +0.0242 / +0.0224
- Full20 hard/soft: +0.0135 / +0.0118

Deltas for candidate vs token-random first20:

- Step20 hard/soft: +0.0145 / +0.0303
- Last10 hard/soft: -0.0267 / -0.0064
- Full20 hard/soft: -0.0224 / -0.0015

Deltas for candidate vs base first20:

- Step20 hard/soft: -0.0000 / +0.0959
- Last10 hard/soft: +0.4491 / +0.5127
- Full20 hard/soft: +0.6275 / +0.6695

Secondary diagnostics at step20:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 5.77% | 7.53% | 19.37% | 4868.9 | -4.8015 | 42.80% | 1.5326 | 1.2597 | 0.9955 |
| token-random first20 | 6.26% | 7.14% | 16.83% | 4949.9 | -4.8626 | 29.10% | 1.3262 | 0.4907 | 1.0041 |
| token-random + teacher final LN | 6.16% | 7.14% | 17.12% | 5124.4 | -4.7154 | 23.95% | 1.3507 | 0.4916 | 1.0037 |
| token-random + teacher final LN + fixed readout | 3.33% | 5.58% | 16.83% | 4973.0 | -4.5902 | 25.45% | 1.3002 | 7.1760 | 1.0000 |

Recent-window trends:

| Variant | Last10 hard slope/100 | Last10 soft slope/100 | Full20 hard slope/100 | Full20 soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| base first20 | -2.1419 | -1.8595 | -15.8978 | -12.6514 |
| token-random first20 | -14.6075 | -13.2348 | -17.7979 | -13.9640 |
| token-random + teacher final LN | -13.7347 | -12.9099 | -18.0111 | -14.2034 |
| token-random + teacher final LN + fixed readout | -13.5662 | -12.5688 | -17.8218 | -13.9980 |

## Decision

Fail loop80. Do not extend to a 40-step or 80-step gate:

- Fixed readout regresses every primary window versus loop79.
- Candidate remains worse than token-random at step20 hard/soft.
- Candidate remains much worse than base on step20 soft and all 10/20-step means.
- Token accuracy and teacher agreement collapse relative to token-random and loop79.
- The large step20 grad norm suggests fixed readout is still causing output-path stress rather than solving it.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Do not continue readout-freeze combinations unless a later diagnostic shows a different readout mechanism is needed.

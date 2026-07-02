# Loop Iteration 79 - Token-Random plus Teacher Final LayerNorm Init Screen

Date: 2026-07-02

## Hypothesis

Loop78 showed that freezing the final LayerNorm suppresses useful adaptation and worsens token-random on every primary 20-step window. Loop79 tests a less restrictive output calibration: initialize the student final LayerNorm from the OPT teacher final LayerNorm, but keep it trainable.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- copy the teacher final LayerNorm weights and bias into `student.final_layer_norm`;
- keep the final LayerNorm trainable;
- keep readout scale trainable;
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
- Loop78 reference: `loop78-token-random-freeze-finalnorm-screen20`
- Loop78 W&B: `wandb/offline-run-20260702_083319-fxhes6s7/run-fxhes6s7.wandb`
- Candidate run: `loop79-token-random-teacher-finalnorm-screen20`
- Candidate W&B: `wandb/offline-run-20260702_083626-d7mdn99h/run-d7mdn99h.wandb`
- Candidate log: `logs/loop79-token-random-teacher-finalnorm-screen20/train.log`

## Result

All reference rows below use the same first 20 optimizer steps.

| Variant | Step20 hard | Step20 soft | Last10 hard | Last10 soft | Full20 hard | Full20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 7.9879 | 4.5508 | 7.9978 | 4.6208 | 8.7184 | 5.1846 |
| token-random first20 | 7.9733 | 4.6164 | 8.4735 | 5.1399 | 9.3683 | 5.8556 |
| token-random + frozen final LN | 7.9880 | 4.6384 | 8.4994 | 5.1662 | 9.3816 | 5.8688 |
| token-random + teacher final LN | 7.9741 | 4.6245 | 8.4227 | 5.1111 | 9.3324 | 5.8423 |

Deltas for candidate vs token-random first20:

- Step20 hard/soft: +0.0008 / +0.0081
- Last10 hard/soft: -0.0509 / -0.0288
- Full20 hard/soft: -0.0359 / -0.0133

Deltas for candidate vs base first20:

- Step20 hard/soft: -0.0138 / +0.0737
- Last10 hard/soft: +0.4249 / +0.4903
- Full20 hard/soft: +0.6139 / +0.6577

Secondary diagnostics at step20:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 5.77% | 7.53% | 19.37% | 4868.9 | -4.8015 | 42.80% | 1.5326 | 1.2597 | 0.9955 |
| token-random first20 | 6.26% | 7.14% | 16.83% | 4949.9 | -4.8626 | 29.10% | 1.3262 | 0.4907 | 1.0041 |
| token-random + frozen final LN | 3.91% | 5.58% | 16.83% | 4949.5 | -4.6055 | 24.11% | 1.2664 | 0.5752 | 1.0038 |
| token-random + teacher final LN | 6.16% | 7.14% | 17.12% | 5124.4 | -4.7154 | 23.95% | 1.3507 | 0.4916 | 1.0037 |

Recent-window trends:

| Variant | Last10 hard slope/100 | Last10 soft slope/100 | Full20 hard slope/100 | Full20 soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| base first20 | -2.1419 | -1.8595 | -15.8978 | -12.6514 |
| token-random first20 | -14.6075 | -13.2348 | -17.7979 | -13.9640 |
| token-random + frozen final LN | -14.4218 | -12.9250 | -17.5769 | -13.7288 |
| token-random + teacher final LN | -13.7347 | -12.9099 | -18.0111 | -14.2034 |

## Decision

Fail loop79. Do not extend to a 40-step or 80-step gate:

- It improves token-random's last10 and full20 hard/soft means, but step20 hard and soft both remain worse than token-random.
- It improves step20 hard vs base, but step20 soft and all base-window means are worse than base.
- It recovers token accuracy and teacher agreement vs loop78, but top5 accuracy and target rank remain worse than base.
- The result is mixed rather than a clear small-screen win over the current best reference.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Teacher final-LN initialization alone is not a promotion candidate on top of token-random.

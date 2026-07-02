# Loop Iteration 81 - Token-Random plus Zero MLP Bias Screen

Date: 2026-07-02

## Hypothesis

Loop77 showed that token-random weakens MLP STA/HTA alignment: MLP cosine changes from `+0.1482` in the base checkpoint to `-0.1868` at token-random step40 and only `+0.0368` at token-random step80. Loop61 showed that zeroing MLP biases alone could improve short soft-loss windows, but did not pass the broader screen. Loop81 tests whether zero MLP biases are more useful on top of token-random.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- zero all 24 SFFN/MLP `fc1` and `fc2` bias tensors after student construction;
- keep readout scale trainable;
- keep SFSA/SFFN structure, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, temporal input ramp, and reset behavior unchanged.

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
- Candidate run: `loop81-token-random-zero-mlp-bias-screen20`
- Candidate W&B: `wandb/offline-run-20260702_084220-9plmiael/run-9plmiael.wandb`
- Candidate log: `logs/loop81-token-random-zero-mlp-bias-screen20/train.log`

## Result

All reference rows below use the same first 20 optimizer steps.

| Variant | Step20 hard | Step20 soft | Last10 hard | Last10 soft | Full20 hard | Full20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 7.9879 | 4.5508 | 7.9978 | 4.6208 | 8.7184 | 5.1846 |
| token-random first20 | 7.9733 | 4.6164 | 8.4735 | 5.1399 | 9.3683 | 5.8556 |
| token-random + zero MLP bias | 7.9980 | 4.6531 | 8.4803 | 5.1531 | 9.3694 | 5.8619 |

Deltas for candidate vs token-random first20:

- Step20 hard/soft: +0.0247 / +0.0367
- Last10 hard/soft: +0.0068 / +0.0132
- Full20 hard/soft: +0.0011 / +0.0063

Deltas for candidate vs base first20:

- Step20 hard/soft: +0.0101 / +0.1022
- Last10 hard/soft: +0.4825 / +0.5323
- Full20 hard/soft: +0.6510 / +0.6772

Secondary diagnostics at step20:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 5.77% | 7.53% | 19.37% | 4868.9 | -4.8015 | 42.80% | 1.5326 | 1.2597 | 0.9955 |
| token-random first20 | 6.26% | 7.14% | 16.83% | 4949.9 | -4.8626 | 29.10% | 1.3262 | 0.4907 | 1.0041 |
| token-random + zero MLP bias | 5.97% | 6.36% | 16.73% | 5003.1 | -4.6617 | 27.58% | 1.3037 | 3.5561 | 1.0038 |

Recent-window trends:

| Variant | Last10 hard slope/100 | Last10 soft slope/100 | Full20 hard slope/100 | Full20 soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| base first20 | -2.1419 | -1.8595 | -15.8978 | -12.6514 |
| token-random first20 | -14.6075 | -13.2348 | -17.7979 | -13.9640 |
| token-random + zero MLP bias | -14.3129 | -12.7827 | -17.6698 | -13.7956 |

## Decision

Fail loop81. Do not extend to a 40-step or 80-step gate:

- Candidate is worse than token-random on step20, last10, and full20 hard/soft.
- Candidate is worse than base on all primary 20-step comparisons.
- Teacher agreement, target rank, spike rate, and grad norm all regress versus token-random.
- Zeroing MLP biases does not address the token-random output conflict.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Do not continue token-random + MLP-bias variants without stronger diagnostic evidence.

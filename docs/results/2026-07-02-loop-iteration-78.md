# Loop Iteration 78 - Token-Random plus Frozen Final LayerNorm Screen

Date: 2026-07-02

## Hypothesis

Loop77 showed that token-random does not remove the STA/HTA output-path conflict: readout scale remains exactly opposite, final LayerNorm remains negatively aligned, and MLP alignment weakens. Loop73 already showed that freezing the readout scale alone is not enough, so loop78 tests the smaller final-normalization intervention first.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- freeze `final_layer_norm` at its default `weight=1`, `bias=0`;
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
- Candidate run: `loop78-token-random-freeze-finalnorm-screen20`
- Candidate W&B: `wandb/offline-run-20260702_083319-fxhes6s7/run-fxhes6s7.wandb`
- Candidate log: `logs/loop78-token-random-freeze-finalnorm-screen20/train.log`

## Result

All reference rows below use the same first 20 optimizer steps, not the last20 window from the 40-step runs.

| Variant | Step20 hard | Step20 soft | Last10 hard | Last10 soft | Full20 hard | Full20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 7.9879 | 4.5508 | 7.9978 | 4.6208 | 8.7184 | 5.1846 |
| token-random first20 | 7.9733 | 4.6164 | 8.4735 | 5.1399 | 9.3683 | 5.8556 |
| token-random + frozen final LN | 7.9880 | 4.6384 | 8.4994 | 5.1662 | 9.3816 | 5.8688 |

Deltas for candidate vs token-random first20:

- Step20 hard/soft: +0.0147 / +0.0220
- Last10 hard/soft: +0.0259 / +0.0263
- Full20 hard/soft: +0.0132 / +0.0132

Deltas for candidate vs base first20:

- Step20 hard/soft: +0.0001 / +0.0876
- Last10 hard/soft: +0.5017 / +0.5454
- Full20 hard/soft: +0.6631 / +0.6842

Secondary diagnostics at step20:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base first20 | 5.77% | 7.53% | 19.37% | 4868.9 | -4.8015 | 42.80% | 1.5326 | 1.2597 | 0.9955 |
| token-random first20 | 6.26% | 7.14% | 16.83% | 4949.9 | -4.8626 | 29.10% | 1.3262 | 0.4907 | 1.0041 |
| token-random + frozen final LN | 3.91% | 5.58% | 16.83% | 4949.5 | -4.6055 | 24.11% | 1.2664 | 0.5752 | 1.0038 |

Recent-window trends:

| Variant | Last10 hard slope/100 | Last10 soft slope/100 | Full20 hard slope/100 | Full20 soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| base first20 | -2.1419 | -1.8595 | -15.8978 | -12.6514 |
| token-random first20 | -14.6075 | -13.2348 | -17.7979 | -13.9640 |
| token-random + frozen final LN | -14.4218 | -12.9250 | -17.5769 | -13.7288 |

## Decision

Fail loop78. Do not extend to a 40-step or 80-step gate:

- Candidate is worse than token-random on step20, last10, and full20 hard/soft.
- Candidate is worse than base on soft loss at step20 and much worse than base over last10/full20.
- Token accuracy and teacher agreement regress vs both base and token-random.
- Freezing only the final LayerNorm reduces neither the early loss gap nor the token-random soft-loss problem.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. The next candidate should not be final-LN-only; the output conflict must be addressed without suppressing useful adaptation.

# Loop Iteration 71 - Embedding Initialization Split Screen

Date: 2026-07-02

## Hypothesis

Loop63 disabled teacher token and position embedding copy together and produced the best recent paper-random signal, but it failed the 80-step promotion gate. Loop70 showed that changing the random embedding scale was not the bottleneck.

Loop71 splits the initialization source to identify whether loop63's signal comes from the tied token/head path or the position path:

- `token-random`: reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy; keep position embeddings copied from the teacher.
- `position-random`: reinitialize only the student position embeddings from `Normal(0, initializer_range)` after the default teacher copy; keep token embedding / tied LM head copied from the teacher.
- Keep SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, readout, temporal input ramp, and reset behavior unchanged.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPUs: 2x H200, GPU0 for `token-random`, GPU1 for `position-random`
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each run

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Random-both reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-both W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Token-random run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Position-random run: `loop71-embedding-init-split-screen40-position-random`
- Position-random W&B: `wandb/offline-run-20260702_080220-08xpr5eu/run-08xpr5eu.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| random token+position | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 | 8.5686 | 5.1342 |
| token-random | 7.6580 | 4.3853 | 7.8217 | 4.4034 | 7.7895 | 4.4061 | 7.8470 | 4.4870 | 8.5789 | 5.1308 |
| position-random | 7.6946 | 4.4270 | 7.8308 | 4.4177 | 7.7977 | 4.4255 | 7.8261 | 4.4520 | 8.2782 | 4.8119 |

Deltas for `token-random` vs base:

- Step 40 hard/soft: -0.0131 / -0.0441
- Last10 hard/soft: -0.0132 / -0.0440
- Last20 hard/soft: -0.0036 / -0.0240
- Last25 hard/soft: +0.0320 / +0.0301
- Full40 hard/soft: +0.3232 / +0.3235

Deltas for `token-random` vs random token+position:

- Step 40 hard/soft: +0.0103 / -0.0007
- Last10 hard/soft: +0.0193 / +0.0056
- Last20 hard/soft: +0.0107 / -0.0083
- Last25 hard/soft: -0.0001 / -0.0132
- Full40 hard/soft: +0.0103 / -0.0033

Deltas for `position-random` vs base:

- Step 40 hard/soft: +0.0236 / -0.0024
- Last10 hard/soft: -0.0041 / -0.0297
- Last20 hard/soft: +0.0046 / -0.0046
- Last25 hard/soft: +0.0110 / -0.0049
- Full40 hard/soft: +0.0224 / +0.0046

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 1.5517 | 0.4328 | 0.9885 |
| random token+position | 4.50% | 5.48% | 13.60% | 4033.3 | -5.1189 | 25.28% | 1.5378 | 2.2442 | 1.0062 |
| token-random | 4.50% | 5.48% | 13.60% | 4228.8 | -5.0888 | 26.32% | 1.5849 | 5.0594 | 1.0052 |
| position-random | 3.13% | 4.21% | 13.60% | 4200.0 | -4.9774 | 38.18% | 1.5301 | 2.1710 | 0.9882 |

## Decision

Fail loop71. Do not extend either candidate to an 80-step gate:

- `position-random` is not a primary-metric win. It regresses step40 hard, last20 hard, last25 hard, full40 hard, and full40 soft vs base, and its token accuracy / teacher agreement are worse.
- `token-random` carries almost all of loop63's useful signal and beats base on step40, last10, and last20 hard/soft, but it does not clearly beat the stronger random-both reference:
  - step40 hard is worse than random-both;
  - last10 hard and soft are both worse than random-both;
  - last20 hard is worse than random-both;
  - full40 hard is worse than random-both.
- Since loop63 random-both already failed the 80-step official gate, a split variant that does not clearly beat loop63 at the 40-step screen should not spend another 80-step run.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. The loop63 signal is primarily tied-token/head randomization, not position randomization, but the split does not produce a stronger promotion candidate.


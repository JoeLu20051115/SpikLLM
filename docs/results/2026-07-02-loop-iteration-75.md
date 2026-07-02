# Loop Iteration 75 - Token-Random plus Attention Output Bypass Screen

Date: 2026-07-02

## Hypothesis

Loop72 token-random improved step80 hard loss and last25 hard/soft means, but failed promotion because step80 soft remained worse than loop16. Loop52 showed that bypassing the extra SFSA pre-projection `attn_out_lif` can improve recent soft windows, although it did not pass its 80-step gate. Loop64 showed the same bypass fails when both token and position embeddings are random.

Loop75 tests the narrower interaction:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- replace each `block.attention.attn_out_lif` with `Identity`;
- keep SFSA otherwise, SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, readout, temporal input ramp, and reset behavior unchanged.

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
- Candidate changed modules: 12 `attn_out_lif` modules bypassed

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Token-random reference run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Random-both bypass reference: `loop64-random-emb-bypass-attnout-screen40`
- Random-both bypass W&B: `wandb/offline-run-20260702_072552-g5jb4wh2/run-g5jb4wh2.wandb`
- Candidate run: `loop75-token-random-bypass-attnout-screen40`
- Candidate W&B: `wandb/offline-run-20260702_081847-bzr1ivbx/run-bzr1ivbx.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| token-random | 7.6580 | 4.3853 | 7.8217 | 4.4034 | 7.7895 | 4.4061 | 7.8470 | 4.4870 | 8.5789 | 5.1308 |
| random token+position + bypass | 7.6708 | 4.4010 | 7.8271 | 4.4145 | 7.7897 | 4.4190 | 7.8511 | 4.5025 | 8.5794 | 5.1418 |
| token-random + bypass | 7.6699 | 4.3981 | 7.8225 | 4.3914 | 7.7931 | 4.4131 | 7.8455 | 4.4903 | 8.5707 | 5.1345 |

Deltas for candidate vs token-random:

- Step 40 hard/soft: +0.0119 / +0.0129
- Last10 hard/soft: +0.0007 / -0.0119
- Last20 hard/soft: +0.0036 / +0.0070
- Last25 hard/soft: -0.0016 / +0.0032
- Full40 hard/soft: -0.0082 / +0.0036

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0011 / -0.0312
- Last10 hard/soft: -0.0125 / -0.0560
- Last20 hard/soft: +0.0000 / -0.0170
- Last25 hard/soft: +0.0304 / +0.0334
- Full40 hard/soft: +0.3149 / +0.3271

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 1.5517 | 0.4328 | 0.9885 |
| token-random | 4.50% | 5.48% | 13.60% | 4228.8 | -5.0888 | 26.32% | 1.5849 | 5.0594 | 1.0052 |
| random token+position + bypass | 4.50% | 5.48% | 13.70% | 4393.4 | -5.1732 | 27.54% | 1.5350 | 12.6479 | 1.0067 |
| token-random + bypass | 4.50% | 5.48% | 13.60% | 4098.3 | -5.0033 | 28.47% | 1.5135 | 3.1894 | 1.0061 |

Recent-window diagnostics:

| Variant | Last10 grad norm | Last20 grad norm | Last25 grad norm | Last10 spike rate | Last20 spike rate | Last25 spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token-random | 3.1698 | 2.2583 | 2.2159 | 25.82% | 25.65% | 26.65% |
| token-random + bypass | 1.9591 | 3.3108 | 20249330389801929346646016.0000 | 30.76% | 32.34% | 34.61% |

## Decision

Fail loop75. Do not extend to an 80-step gate:

- It improves last10 soft vs token-random, but step40 hard/soft and last20 hard/soft regress.
- It is not a clear win over base because last20 hard is tied and last25/full40 hard/soft regress.
- The bypass increases spike rate but does not produce stable output improvement.
- The last25 grad norm has the same overflow-scale instability pattern observed in loop64.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Bypassing `attn_out_lif` does not solve token-random's 80-step soft-loss gap.


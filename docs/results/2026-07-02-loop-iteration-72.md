# Loop Iteration 72 - Token-Random 80-Step Gate

Date: 2026-07-02

## Hypothesis

Loop71 showed that token/head randomization carries almost all of loop63's useful 40-step signal, while randomizing only position embeddings does not help. Although token-random did not clearly beat random token+position at 40 steps, it had stronger teacher-agreement windows and copied teacher position embeddings, which could plausibly address loop63's 80-step soft-loss miss.

Loop72 runs the token-random variant through an official 80-step gate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- keep SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, readout, temporal input ramp, and reset behavior unchanged.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 80
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before the run

## Runs

- Official best baseline: loop16 small gate (`cvxuw267` / refresh `1f8plmtv`)
- Current-base reference: `loop38-stableclip-small-seq512-bs2ga16-t4-80step-20260702-022151`
- Current-base W&B: `wandb/offline-run-20260702_022154-cwmvdi1s/run-cwmvdi1s.wandb`
- Random-both reference: `loop63-paper-random-embedding-gate80-random-emb`
- Random-both W&B: `wandb/offline-run-20260702_072056-vl15sdve/run-vl15sdve.wandb`
- Candidate run: `loop72-token-random-gate80`
- Candidate W&B: `wandb/offline-run-20260702_080608-9rru84k5/run-9rru84k5.wandb`

## Result

| Variant | Step 80 hard | Step 80 soft | Last10 hard | Last10 soft | Last25 hard | Last25 soft | Last40 hard | Last40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop16 official best | 7.4532 | 4.5196 | n/a | n/a | 7.7798 | 4.2932 | n/a | n/a |
| current base | 7.4572 | 4.5187 | 7.7706 | 4.2969 | 7.7819 | 4.2997 | 7.7818 | 4.3152 |
| random token+position | 7.4536 | 4.5357 | 7.7591 | 4.2966 | 7.7709 | 4.2944 | 7.7726 | 4.3125 |
| token-random | 7.4496 | 4.5256 | 7.7578 | 4.2859 | 7.7691 | 4.2879 | 7.7748 | 4.3112 |

Deltas for token-random vs loop16 official best:

- Step 80 hard/soft: -0.0036 / +0.0060
- Last25 hard/soft: -0.0107 / -0.0053

Deltas for token-random vs current base:

- Step 80 hard/soft: -0.0076 / +0.0069
- Last10 hard/soft: -0.0128 / -0.0111
- Last25 hard/soft: -0.0128 / -0.0119
- Last40 hard/soft: -0.0070 / -0.0040

Deltas for token-random vs random token+position:

- Step 80 hard/soft: -0.0040 / -0.0101
- Last10 hard/soft: -0.0013 / -0.0107
- Last25 hard/soft: -0.0018 / -0.0065
- Last40 hard/soft: +0.0021 / -0.0013

Secondary diagnostics at step 80:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current base | 5.38% | 8.02% | 19.57% | 4138.2 | -4.6461 | 27.34% | 1.5319 | 0.5223 | 0.9846 |
| random token+position | 5.48% | 7.24% | 19.57% | 4144.3 | -4.5314 | 19.78% | 1.4907 | 0.4553 | 1.0038 |
| token-random | 4.40% | 5.48% | 19.57% | 4114.5 | -4.6025 | 20.42% | 1.4960 | 0.4595 | 1.0020 |

Recent-window trends:

| Variant | Last25 hard slope/100 | Last25 soft slope/100 | Last40 hard slope/100 | Last40 soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| current base | -0.2835 | +0.3007 | -0.0674 | -0.0604 |
| random token+position | -0.2527 | +0.3777 | -0.0683 | -0.0541 |
| token-random | -0.2653 | +0.3237 | -0.1096 | -0.1121 |

## Decision

Do not promote loop72 to long or full training:

- It improves the official loop16 last25 hard/soft means and sets the best step80 hard loss among these local gates.
- It does not beat loop16 on step80 soft loss, which remains worse by +0.0060.
- Step80 token accuracy and teacher agreement are worse than both current base and random-both.
- The last25 soft slope is positive, so the recent soft-loss window is not a clear downward trend.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Loop72 is a useful secondary reference because it improves last25 hard/soft and step80 hard, but it is not a clean hard+soft promotion candidate under the user's small-screen rule.


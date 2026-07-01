# Loop Iteration 65 - Random Embedding plus Fixed Readout Screen

Date: 2026-07-02

## Hypothesis

Loop62 showed that, by the step40 baseline checkpoint, STA and HTA have opposite gradients on the scalar readout scale. Loop63's random-embedding run was more paper-faithful and competitive, but its readout scale moved above 1.0 and its 80-step soft loss failed to beat loop16.

Loop65 tests the minimal output-scale intervention on top of loop63:

- disable teacher token/position embedding copy, as in loop63;
- set `readout_log_scale = 0` and freeze it, so readout scale stays exactly 1.0;
- keep tied LM head, SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged.

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
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Random-only reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-only W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Candidate run: `loop65-random-emb-fixed-readout-screen40`
- Candidate W&B: `wandb/offline-run-20260702_073009-blsu0ko8/run-blsu0ko8.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 |
| random + fixed readout | 7.6593 | 4.3890 | 7.8131 | 4.3919 | 7.7846 | 4.4157 | 7.8464 | 4.4973 |

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0118 / -0.0403
- Last10 hard/soft: -0.0218 / -0.0555
- Last20 hard/soft: -0.0085 / -0.0144
- Last25 hard/soft: +0.0314 / +0.0404

Deltas for candidate vs random-only:

- Step 40 hard/soft: +0.0116 / +0.0030
- Last10 hard/soft: +0.0106 / -0.0059
- Last20 hard/soft: +0.0057 / +0.0012
- Last25 hard/soft: -0.0007 / -0.0029
- Full40 hard/soft: -0.0030 / -0.0040

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% | 0.4328 | 0.9885 |
| random embedding | 4.50% | 5.48% | 4033.3 | -5.1189 | 25.28% | 2.2442 | 1.0062 |
| random + fixed readout | 4.50% | 5.48% | 4284.6 | -5.0212 | 24.09% | 0.2348 | 1.0000 |

## Decision

Fail loop65. Do not extend to an 80-step gate:

- It improves over the current matched base, but loop63 random-only is the relevant promotion reference for this combination.
- Compared with random-only, step40 hard and soft both regress.
- Last10 hard regresses, and last20 hard/soft both regress.
- Last25 and full40 means improve only slightly, mostly because fixed readout smooths the early phase.
- Spike rate drops further below random-only, worsening the under-firing concern from loop55.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Fixing readout scale is not enough to turn loop63's paper-random initialization into a promotion candidate.

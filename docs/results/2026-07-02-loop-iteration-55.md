# Loop Iteration 55 - Current Code True 3GPU Loop14 Geometry

Date: 2026-07-02

## Hypothesis

Loop39 failed under a 2GPU approximation of loop14's medium geometry. GPU1 later became available, so this loop retests the current stable `main` code under the true loop14-matched 3GPU geometry before spending any full-training compute.

This was a geometry/status probe only. No source code was changed.

## Setup

- Code state: `main` after loop54 docs, stable clipping and scheduler-horizon support included.
- Run: `loop55-current-true3gpu-seq1024-bs4ga64-t4-160step-20260702-055136`
- Local W&B: `wandb/offline-run-20260702_055140-xnttrmg6/run-xnttrmg6.wandb`
- Output: `output/loop55-current-true3gpu-seq1024-bs4ga64-t4-160step-20260702-055136`
- GPUs: 3x H200, GPU0/GPU1/GPU2
- Sequence length: 1024
- Time steps: 4
- Per-GPU batch size: 4
- Gradient accumulation: 64
- Effective tokens per optimizer step: 786,432
- Planned max optimizer steps: 160
- Scheduler horizon: 2000 steps
- Precision: bf16
- W&B mode: offline

## Result

The run reached step 86 and was intentionally interrupted after the step-80 comparison showed it was clearly behind loop14. This saved the remaining GPU time instead of running to step 160.

Step 80 comparison, same tokens seen as loop14:

| Run | Hard | Soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop14 | 7.6577 | 4.3664 | 5.35% | 8.94% | 3027.9 | -4.6409 | 33.66% |
| loop55 | 7.7714 | 4.4120 | 3.91% | 6.21% | 3101.5 | -4.6877 | 24.83% |

Deltas for loop55 vs loop14 at step 80:

- Hard/soft: +0.1137 / +0.0457
- Token accuracy / teacher agreement: -1.44 pp / -2.74 pp
- Target rank / margin: +73.6 / -0.0468
- Spike rate: -8.83 pp

Last logged step:

- Step 86 hard/soft: 7.6883 / 4.4309
- Step 86 token accuracy / teacher agreement: 3.71% / 5.67%
- Step 86 target rank / margin: 3881.5 / -4.8647
- Step 86 spike rate: 23.92%

Recent-window diagnostics at step 86:

| Window | Decision | Hard mean | Soft mean | Hard slope / 100 | Soft slope / 100 |
| --- | --- | ---: | ---: | ---: | ---: |
| Last 10 | fail | 7.7619 | 4.4489 | -0.6267 | +0.5962 |
| Last 20 | fail | 7.7663 | 4.4442 | -0.1900 | +0.3292 |
| Last 25 | fail | 7.7668 | 4.4593 | -0.0542 | -0.1352 |
| Last 40 | fail | 7.7675 | 4.4987 | -0.0685 | -0.4039 |
| Last 80 | fail | 8.1494 | 4.8749 | -2.1438 | -2.0544 |

For the same step-86 prefix, loop14's last-25 and last-40 windows still produced `extend`, with better hard/soft means and better output metrics.

## Decision

Fail loop55. Do not continue this run to step 160 and do not launch full training from the current code state:

- It is worse than loop14 at the same 3GPU geometry and the same tokens seen by step 80.
- It is worse than loop14 on every primary output metric at step 80: hard loss, soft loss, token accuracy, teacher agreement, target rank, and target margin.
- Its recent step-86 windows do not meet the hard/soft continuation rule.
- Its spike rate is much lower than loop14 at matched steps, suggesting the current code path is under-firing relative to the best historical long run.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. The next loop should be a code-level hypothesis screened against the current best small baseline before any long or full run.

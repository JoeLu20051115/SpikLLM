# Loop Iteration 90 - Two-GPU Seq512 Baseline Failure

Date: 2026-07-02

## Purpose

Observed two-GPU baseline run that started while the requested three-GPU baseline was being retried.

This is not the requested three-GPU loop14 geometry.

## Setup

- Run name: `loop90-loop16-baseline-seq512-bs2-ga16-2xh200-1bt-20260702-091455`
- tmux session: `loop90_loop16_baseline_seq512_bs2_ga16_2xh200_1bt_20260702_091455`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/0eztd5vg
- Local W&B: `wandb/run-20260702_091458-0eztd5vg/run-0eztd5vg.wandb`
- Log: `logs/loop90-loop16-baseline-seq512-bs2-ga16-2xh200-1bt-20260702-091455/train.log`
- Output: `output/loop90-loop16-baseline-seq512-bs2-ga16-2xh200-1bt-20260702-091455`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- Sequence length: `512`
- Time steps: `4`
- Batch size per rank: `2`
- Gradient accumulation: `16`
- Precision: `bf16`

## Interim State At First Record

W&B local parse:

- rows: `31`;
- last step: `31`;
- latest hard/soft: `10.1526 / 6.4478`.

Decision at record time: in progress, but it is not the requested three-GPU baseline. It currently blocks GPUs 0 and 1. A clean three-GPU best-baseline run requires stopping or waiting for this run and the GPU2 screen160 run.

## Final State

The run later failed with the same distributed CUDA illegal-memory-access pattern seen in loop83 and loop87.

Failure excerpt:

```text
[rank1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Root Cause: rank 1 exitcode -6 (SIGABRT)
```

Final local W&B parse:

- rows: `277`;
- last step: `277`;
- latest hard/soft: `7.8301 / 4.4876`;
- latest total loss: `4.4390`;
- latest token accuracy: `7.24%`;
- latest teacher top-1 agreement: `12.92%`;
- latest top-5 accuracy: `20.35%`;
- latest target rank mean: `4029.8`;
- latest target margin mean: `-4.8651`;
- latest spike rate: `37.19%`;
- latest tokens seen: `9,076,736`.

Recent windows:

| Window | Hard mean | Soft mean | Hard slope/100 | Soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| last10 | 7.7905 | 4.4343 | -1.2410 | +0.3605 |
| last25 | 7.7195 | 4.3104 | +0.4049 | +1.2438 |
| last50 | 7.7396 | 4.3367 | -0.0881 | +0.0723 |
| last100 | 7.7723 | 4.3652 | -0.1523 | -0.1164 |

Decision: failed runtime attempt and not the requested geometry. It does not update the baseline and produced no checkpoint.

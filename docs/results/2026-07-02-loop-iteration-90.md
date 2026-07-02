# Loop Iteration 90 - Two-GPU Seq512 Baseline In Progress

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

## Current State At Record Time

W&B local parse:

- rows: `31`;
- last step: `31`;
- latest hard/soft: `10.1526 / 6.4478`.

Decision at record time: in progress, but it is not the requested three-GPU baseline. It currently blocks GPUs 0 and 1. A clean three-GPU best-baseline run requires stopping or waiting for this run and the GPU2 screen160 run.

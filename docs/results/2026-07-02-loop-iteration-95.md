# Loop Iteration 95 - Queued Clean Three-GPU Baseline Launcher

Date: 2026-07-02

## Purpose

The requested clean three-GPU best-baseline run cannot currently start because GPUs 0 and 1 are occupied by the observed single-GPU baseline runs `loop91a` and `loop91b`.

To avoid another contaminated launch, loop95 is a queued launcher. It waits without occupying GPUs, then starts the current best baseline only after all three GPUs are idle for multiple consecutive checks.

## Intended Training Run

- Source baseline: current `main` / loop16 code baseline
- Long-run geometry: loop14 three-GPU setup
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2`
- Sequence length: `1024`
- Time steps: `4`
- Batch size per rank: `4`
- Gradient accumulation: `64`
- Precision: `bf16`
- W&B project: `bispikclm`
- Paper-faithful losses unchanged: `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`

## Launch Guard

The launcher will poll `nvidia-smi` and require all three GPUs to report low memory use for three consecutive checks before starting `torchrun`.

This is intended to prevent the resource-conflict pattern seen in loops89 and 91, where a three-GPU run reached W&B initialization but had to be stopped because another run occupied one or more GPUs.

## Current State At Record Time

- `loop91a-loop16-baseline-seq512-bs2-ga16-gpu0-1bt-20260702-092515` is still running on GPU0.
- `loop91b-loop16-baseline-seq512-bs2-ga16-gpu1-1bt-20260702-092515` is still running on GPU1.
- GPU2 is free after loop94 completed.

Decision at record time: queue a guarded three-GPU launch rather than starting immediately. The baseline is unchanged.

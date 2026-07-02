# Loop Iteration 87 - Two-GPU Baseline Retry

Date: 2026-07-02

## Purpose

Observed retry of the current baseline on two GPUs after the loop86 smoke test.

## Setup

- Run name: `loop87-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090855`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/f4em75fb
- Local W&B: `wandb/run-20260702_090828-f4em75fb/run-f4em75fb.wandb`
- Log: `logs/loop87-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090855/train.log`
- Output: `output/loop87-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090855`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- Sequence length: `1024`
- Time steps: `4`
- Batch size per rank: `4`
- Gradient accumulation: `64`
- Precision: `bf16`

## Result

The run wrote six optimizer-step rows, then failed during rank-1 backward with the same CUBLAS/CUDA illegal-memory-access pattern as loop83.

Failure excerpt:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx(...)`
Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Root Cause: rank 1 exitcode -6 (SIGABRT)
```

Final W&B state:

- rows: `6`;
- last step: `6`;
- latest hard/soft: `11.4772 / 7.3300`;
- latest total loss: `6.4183`;
- latest token accuracy: `0.00%`;
- latest teacher top-1 agreement: `0.00%`;
- latest top-5 accuracy: `0.00%`;
- latest target rank mean: `21926.9`;
- latest target margin mean: `-6.6013`;
- latest spike rate: `67.10%`;
- latest readout scale: `0.999987`;
- latest tokens seen: `3,145,728`.

Recent windows:

| Window | Hard mean | Soft mean | Hard slope/100 | Soft slope/100 |
| --- | ---: | ---: | ---: | ---: |
| last3 | 11.7482 | 7.5138 | -25.2223 | -17.8769 |
| last5 | 11.9397 | 7.6931 | -19.9150 | -17.8612 |
| all6 | 11.9962 | 7.7385 | -16.2233 | -14.0983 |

Decision: failed runtime attempt. No checkpoint was produced and it does not update the baseline.

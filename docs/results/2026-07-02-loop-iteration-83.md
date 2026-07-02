# Loop Iteration 83 - Two-GPU Baseline Attempt

Date: 2026-07-02

## Purpose

Observed follow-up baseline artifact after the interrupted loop82 three-GPU run.

This used the same current best source baseline as loop82, but with two GPUs:

- current `main` source / loop16 small-gate best code;
- same-dimension SpAD projectors as identities;
- all five SpAD losses and paper weights unchanged: `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`;
- `seq=1024`, `T=4`, per-rank batch `4`, gradient accumulation `64`;
- `CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node=2`.

## Launch Artifact

- Run name: `loop83-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090030`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/h8jwufki
- Local W&B: `wandb/run-20260702_090037-h8jwufki/run-h8jwufki.wandb`
- Output: `output/loop83-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090030`
- Log: `logs/loop83-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090030/train.log`
- Launch command: `logs/loop83-loop16-baseline-seq1024-bs4-ga64-2xh200-1bt-20260702-090030/launch.cmd`

## Result

The run failed early during backward on rank 1:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx(...)`
Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Root Cause: rank 1 exitcode -6 (SIGABRT)
```

Only three optimizer-step rows were recorded in the local W&B file.

At step 3:

- hard/soft: `11.5212 / 7.6273`;
- total loss: `6.5197`;
- token accuracy: `0.00%`;
- teacher top-1 agreement: `0.024%`;
- top-5 accuracy: `0.024%`;
- target rank mean: `22645.8`;
- target margin mean: `-6.1021`;
- spike rate: `66.58%`;
- readout scale: `0.999999`;
- tokens seen: `1,572,864`.

Decision: failed infrastructure/runtime attempt. It does not update the baseline and does not provide a usable checkpoint.

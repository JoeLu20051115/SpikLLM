# Loop Iteration 89 - Screen160 Artifact And Cancelled Three-GPU Attempt

Date: 2026-07-02

## Purpose

Two loop89 artifacts were observed because a screen160 run on GPU2 appeared immediately before the three-GPU baseline relaunch.

The GPU2 screen was not the requested three-GPU baseline. A three-GPU baseline attempt was then started with a high random port, but it overlapped GPU2 with the already-running screen and was stopped immediately to avoid wasting resources.

## Screen160 Artifact

- tmux session: `loop89_sfa_mlp_ln_screen160_gpu2`
- Script: `logs/loop89-sfa-mlp-ln-screen160-gpu2-20260702-091400/runner.py`
- Log: `logs/loop89-sfa-mlp-ln-screen160-gpu2-20260702-091400/train.log`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Steps: `160`
- W&B: disabled

Candidate remains the same same-dimension SFA MLP+LayerNorm projector used in loops84-85.

Status at record time: still running. It does not supersede the current baseline unless it later shows a clear small-batch win over the baseline, which loops84-85 did not show.

## Cancelled Three-GPU Attempt

- Run name: `loop89-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091436`
- tmux session: `loop89_loop16_baseline_seq1024_bs4_ga64_3xh200_1bt_20260702_091436`
- Port: `55989`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/gy8szku1
- Local W&B: `wandb/run-20260702_091440-gy8szku1/run-gy8szku1.wandb`
- Log: `logs/loop89-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091436/train.log`

This attempt reached W&B initialization but had zero optimizer-step rows. It was stopped because GPU2 was already occupied by `loop89_sfa_mlp_ln_screen160_gpu2`, so the run would not have been a clean three-GPU baseline.

Decision: cancelled resource-conflict attempt. It does not update the baseline and should not be compared against loop14/loop16.

# 2026-07-02 loop14 rerun queue

## Status

- State: queued, waiting for GPUs 0/1/2 to become available.
- Queue PID: `1596548`.
- Reason for queueing: the current three-card run `loop105-loop16-baseline3x-seq1024-bs4-ga64-T4-3xh200-1bt-20260702-120754` is still occupying GPUs 0/1/2 at 100% utilization.
- No existing loop14 checkpoint was found around step 600/663, so this is a from-scratch rerun rather than a resume.

## Code And Run

- Code worktree: `.worktrees/loop14-baseline-5e4d6df`
- Code commit: `5e4d6df`
- Run name: `loop14-rerun-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260702-134621`
- Launch script: `.worktrees/loop14-baseline-5e4d6df/logs/loop14-rerun-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260702-134621/launch_wait_and_run.sh`
- Wait log: `.worktrees/loop14-baseline-5e4d6df/logs/loop14-rerun-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260702-134621/waiter.log`
- Train log after start: `.worktrees/loop14-baseline-5e4d6df/logs/loop14-rerun-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260702-134621/train.log`
- Output dir: `.worktrees/loop14-baseline-5e4d6df/output/loop14-rerun-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260702-134621`

## Original Loop14 Geometry

- Teacher/config: `configs/bispikclm_opt125m_spad.toml`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2`
- Distributed: `torchrun --nproc_per_node=3 --master_port=29658`
- Sequence length: `1024`
- Time steps: `4`
- Per-rank batch size: `4`
- Gradient accumulation: `64`
- Precision: `bf16`
- Max steps: `2000`
- W&B project: `bispikclm`

## Launch Policy

The launcher checks GPU memory every 300 seconds and starts training only when GPUs 0/1/2 are each below 2000 MiB used. This avoids interrupting the active loop105 run while preserving the requested loop14 three-card rerun.

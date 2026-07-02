# 2026-07-02 loop14 rerun restart

## Status

- State: running.
- Initial queue PID `1596548` was stopped after the user requested immediate launch.
- Previous three-card run `loop105-loop16-baseline3x-seq1024-bs4-ga64-T4-3xh200-1bt-20260702-120754` was stopped by explicit user instruction.
- No existing loop14 checkpoint was found around step 600/663, so this is a from-scratch rerun rather than a resume.

## Code And Run

- Code worktree: `.worktrees/loop14-baseline-5e4d6df`
- Code commit: `5e4d6df`
- Run name: `loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431`
- W&B run id: `qn6bddld`
- Launch PIDs at startup: parent bash `1598670`, torchrun `1598672`, ranks `1598680`, `1598681`, `1598682`
- Train log: `.worktrees/loop14-baseline-5e4d6df/logs/loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431/train.log`
- Output dir: `.worktrees/loop14-baseline-5e4d6df/output/loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431`

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

The waiting launcher was superseded by the immediate restart request. The actual launch uses the original loop14 run name, output path, W&B run name, code commit, and training arguments, but runs from the preserved loop14 worktree so the source tree is fixed at `5e4d6df`.

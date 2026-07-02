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

Final screen160 result:

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step160 hard | 7.9777 | 7.9779 | +0.0003 |
| step160 soft | 4.9140 | 4.9152 | +0.0012 |
| last10 hard | 7.8128 | 7.8112 | -0.0016 |
| last10 soft | 4.3154 | 4.3155 | +0.0001 |
| last20 hard | 7.8078 | 7.8084 | +0.0006 |
| last20 soft | 4.3340 | 4.3342 | +0.0002 |
| step160 total | 4.4824 | 4.5140 | +0.0317 |
| step160 feature | 0.3853 | 0.6906 | +0.3053 |
| step160 token accuracy | 1.76% | 1.76% | 0.00 pp |
| step160 teacher top-1 agreement | 2.25% | 2.25% | 0.00 pp |
| step160 target rank mean | 4608.8 | 4584.0 | -24.8 |
| step160 target margin mean | -4.9267 | -4.8656 | +0.0610 |
| step160 spike rate | 22.35% | 30.75% | +8.40 pp |
| step160 grad norm | 0.7221 | 0.7064 | -0.0157 |

Decision: reject as a baseline replacement. The longer screen confirms loops84-85: the trainable same-dim SFA MLP+LayerNorm projector does not clearly beat the current baseline, and step160 hard/soft/total are slightly worse.

## Cancelled Three-GPU Attempt

- Run name: `loop89-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091436`
- tmux session: `loop89_loop16_baseline_seq1024_bs4_ga64_3xh200_1bt_20260702_091436`
- Port: `55989`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/gy8szku1
- Local W&B: `wandb/run-20260702_091440-gy8szku1/run-gy8szku1.wandb`
- Log: `logs/loop89-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091436/train.log`

This attempt reached W&B initialization but had zero optimizer-step rows. It was stopped because GPU2 was already occupied by `loop89_sfa_mlp_ln_screen160_gpu2`, so the run would not have been a clean three-GPU baseline.

Decision: cancelled resource-conflict attempt. It does not update the baseline and should not be compared against loop14/loop16.

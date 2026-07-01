# Loop Iteration 20 - Readout Scale 0.9 Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Loop14's longer probe learned the trainable readout scale down to about 0.92, while the earlier 2.0 readout probe reached the same hard-loss plateau. This iteration tests a conservative config-only candidate: initialize `readout_scale = 0.9` instead of 1.0, keeping the model structure, SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, and data source unchanged.

## Gate Geometry

- GPU: 1x H200
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0 before entering the training CLI.

## Current Small-Batch Baseline

- Baseline: loop16 identity-projector candidate
- Commit: `9c0d64f`
- Run: `loop16-gate-candidate-small-seq512-bs2-ga16-1xh200-20260701-2134`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/cvxuw267
- Step 80 hard/soft: 7.4532 / 4.5195
- Last 25-step hard/soft means: 7.7798 / 4.2932
- Token accuracy at step 80: 4.79%
- Teacher top-1 agreement at step 80: 6.07%
- Target rank mean at step 80: 4162.4
- Target margin mean at step 80: -4.6015

## Candidate Result

- Candidate: `readout_scale = 0.9`
- Run: `loop20-readout09-small-seq512-bs2-ga16-1xh200-20260701-215603`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/07y1moj1
- Step 80 hard/soft: 7.4681 / 4.5348
- Last 25-step hard/soft means: 7.7779 / 4.2954
- Token accuracy at step 80: 4.31%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4141.7
- Target margin mean at step 80: -4.6237
- Spike rate mean at step 80: 29.86%
- Readout scale at step 80: 0.8857

## Decision

- Fail against the current small-batch best baseline.
- The candidate slightly improved step-80 target rank, but worsened step-80 hard loss, soft loss, token accuracy, teacher agreement, and target margin.
- Do not keep `readout_scale = 0.9`.
- Keep loop16 as the current small-batch best baseline.

# Loop Iteration 21 - Input Scale 25 Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The default implicit input scale is `1 / initializer_range = 50`, while loop17's `input_scale = 10.0` was slightly too weak and failed the current small-batch baseline. This iteration tests the midpoint `input_scale = 25.0` as a single-variable current-scaling candidate, keeping the model structure, SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, readout initialization, and data source unchanged.

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

- Candidate: `input_scale = 25.0`
- Run: `loop21-inputscale25-small-seq512-bs2-ga16-1xh200-20260701-220232`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/fusppyw3
- Step 80 hard/soft: 7.4746 / 4.5524
- Last 25-step hard/soft means: 7.7783 / 4.3015
- Token accuracy at step 80: 3.23%
- Teacher top-1 agreement at step 80: 5.28%
- Target rank mean at step 80: 4236.1
- Target margin mean at step 80: -4.6014
- Spike rate mean at step 80: 27.58%
- Readout scale at step 80: 0.9835

## Decision

- Fail against the current small-batch best baseline.
- Lowering input scale reduced spike rate, but worsened step-80 hard loss, soft loss, token accuracy, teacher agreement, and target rank.
- Do not keep `input_scale = 25.0`.
- Keep loop16 as the current small-batch best baseline.

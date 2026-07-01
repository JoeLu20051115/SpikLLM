# Loop Iteration 19 - Attention-Only Teacher Init Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Loop11 showed that initializing full student layers from the teacher degraded early hard-label behavior. This iteration tests a narrower variant: initialize only the BiSpik attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `out_proj`) from the corresponding OPT self-attention weights, without initializing MLP weights or whole transformer blocks.

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

- Candidate: attention-only teacher initialization
- Run: `loop19-gate-attninit-small-seq512-bs2-ga16-1xh200-20260701-2148`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/joc7jmun
- Step 80 hard/soft: 8.1839 / 5.2825
- Last 25-step hard/soft means: 8.6951 / 5.2572
- Token accuracy at step 80: 0.98%
- Teacher top-1 agreement at step 80: 1.27%
- Target rank mean at step 80: 5774.3
- Target margin mean at step 80: -5.1348

## Decision

- Fail against the current small-batch best baseline.
- Attention-only teacher initialization strongly worsened hard/soft loss, token accuracy, teacher agreement, target rank, and target margin.
- Do not keep attention-only teacher initialization.
- Keep loop16 as the current small-batch best baseline.

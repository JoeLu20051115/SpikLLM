# Loop Iteration 17 - Input Scale 10 Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Loop14/16 small-batch gates showed high spike rates early in training, with loop14 moving from 47.41% at step 20 to 37.15% at step 80. Because the student now initializes token and position embeddings from the OPT teacher, the default implicit input scale of `1 / initializer_range = 50` may overdrive the spiking blocks. This iteration tests a single config-only candidate: set `input_scale = 10.0`.

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
- Spike rate mean at step 80: 31.30%

## Candidate Result

- Candidate: `input_scale = 10.0`
- Run: `loop17-gate-inputscale10-small-seq512-bs2-ga16-1xh200-20260701-2140`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/jmm2cr57
- Step 80 hard/soft: 7.4586 / 4.5276
- Last 25-step hard/soft means: 7.7760 / 4.2949
- Token accuracy at step 80: 4.31%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4194.0
- Target margin mean at step 80: -4.6244
- Spike rate mean at step 80: 32.08%

## Decision

- Fail against the current small-batch best baseline.
- The candidate did not improve the primary step-80 hard/soft losses over loop16, and it also reduced token accuracy and teacher top-1 agreement while worsening target rank and margin.
- Do not keep `input_scale = 10.0`.
- Keep loop16 as the current small-batch best baseline for the next iteration.

# Loop Iteration 30 - Hidden-Only SFA LayerNorm Projector Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The SFA appendix defines student feature matching as `LayerNorm(MLP(mean_t H_SNN(t)))` before MSE against teacher hidden states. Loop16 improved the small-batch baseline by making same-dimension SpAD projectors identities, but that change affected both EA and SFA. This iteration tests a narrower paper-fidelity repair: keep the same-dimension embedding projector as an exact identity for EA, but use a trainable same-dimension LayerNorm projector only for hidden/SFA alignment. SpAD loss weights, attention alignment, embedding alignment input scale, temperature, optimizer, schedule, labels, teacher logits, data source, initialization, reset behavior, and output readout stay unchanged.

## Candidate Code

- Branch: `loop30-hidden-projector-ln`
- Commit: `a8dd6da`
- Remote branch: `origin/loop30-hidden-projector-ln`
- Diff summary:
  - Added optional `normalize_same_dim` support to `SpADProjector`.
  - `build_student_from_teacher` now constructs the hidden projector with same-dimension LayerNorm enabled.
  - The embedding projector remains identity and parameter-free when teacher/student dimensions match.
  - Added regression coverage for hidden-only same-dimension SFA LayerNorm behavior.

## Verification Before Gate

- Clean worktree baseline before edits:
  - `PATH=/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin:$PATH PYTHONPATH=$PWD /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest tests/smoke -q`
  - Result: 43 passed, 58 warnings.
- Red test before implementation:
  - `test_hidden_projector_uses_layer_norm_for_same_dim_sfa`
  - Result: failed as expected because the hidden projector was still an identity with no trainable parameters.
- Targeted tests after implementation:
  - `test_hidden_projector_uses_layer_norm_for_same_dim_sfa`: passed.
  - `test_same_dimension_spad_projector_is_identity`: passed.
  - `test_trainable_parameter_detection_skips_identity_projectors`: passed.
  - `test_identity_projector_checkpoint_round_trip`: passed.
- Full smoke:
  - `PATH=/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin:$PATH PYTHONPATH=$PWD /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest tests/smoke -q`
  - Result: 44 passed, 58 warnings.

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

- Candidate: same-dimension LayerNorm projector for hidden/SFA only.
- Run: `loop30-hiddenproj-ln-small-seq512-bs2-ga16-1xh200-20260701-231932`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/dx00sinl
- Step 80 hard/soft: 7.4779 / 4.5393
- Last 25-step hard/soft means: 7.7856 / 4.3028
- Step 80 embedding/attention/feature losses: 2.7303 / 0.5350 / 0.8002
- Step 80 feature rate/MSE losses: 0.2735 / 1.3270
- Token accuracy at step 80: 4.60%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4262.7
- Target margin mean at step 80: -4.7617
- Spike rate mean at step 80: 34.41%
- Readout scale at step 80: 0.9844
- Logit std at step 80: 1.5196
- Last 25-step token accuracy mean: 4.35%
- Last 25-step teacher top-1 agreement mean: 6.41%
- Last 25-step target rank mean: 3932.2
- Last 25-step target margin mean: -4.9202

## Decision

- Fail against the current small-batch best baseline.
- The hidden-only LayerNorm projector increased feature loss and worsened the primary hard/soft losses, target rank, and target margin.
- Do not launch a long probe or full training from this result.
- Do not merge `loop30-hidden-projector-ln` into `main`.
- Keep loop16 as the current small-batch best baseline.

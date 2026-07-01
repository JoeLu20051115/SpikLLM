# Loop Iteration 27 - Direct Teacher Attention Rate Drive Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The SAA Rate-MSE branch should feed teacher attention probabilities directly into the LIF rate encoder. The current implementation first scales each teacher-attention row so its maximum reaches the spike threshold. That anti-zero drive makes the teacher spike proxy denser than the teacher attention probabilities themselves and can overemphasize attention alignment relative to output hard/soft learning. This iteration tests the single-variable change of preserving the teacher attention probability scale before rate encoding, keeping the attention distribution MSE, SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, initialization, reset behavior, and output readout unchanged.

## Candidate Code

- Branch: `loop27-direct-attention-rate`
- Commit: `aea66d0`
- Remote branch: `origin/loop27-direct-attention-rate`
- Diff summary:
  - `_attention_rate_drive` now returns teacher attention probabilities unchanged.
  - Added a regression test that the teacher attention rate drive preserves probability scale instead of row-max scaling to the spike threshold.
  - Updated the collapsed-attention regression test to keep the nonzero-loss and backward-gradient checks without depending on the old row-max amplification threshold.

## Verification Before Gate

- Clean worktree baseline before edits:
  - `PATH=/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin:$PATH PYTHONPATH=$PWD /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest tests/smoke -q`
  - Result: 43 passed, 58 warnings.
- Red test before implementation:
  - `test_spad_teacher_attention_rate_drive_preserves_probability_scale`
  - Result: failed as expected because `[0.20, 0.10]` was amplified to `[0.70, 0.35]`.
- Targeted tests after implementation:
  - `test_spad_teacher_attention_rate_drive_preserves_probability_scale`: passed.
  - `test_spad_attention_and_feature_losses_include_rate_mse_branches`: passed.
  - `test_spad_attention_loss_penalizes_zero_student_attention_distribution`: passed.
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

- Candidate: direct teacher attention probabilities for SAA Rate-MSE drive.
- Run: `loop27-direct-attnrate-small-seq512-bs2-ga16-1xh200-20260701-225245`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/cpoojtyh
- Step 80 hard/soft: 7.4639 / 4.5346
- Last 25-step hard/soft means: 7.7729 / 4.2914
- Step 80 embedding/attention/feature losses: 2.7438 / 0.5753 / 0.4435
- Step 80 attention rate/MSE losses: 0.7693 / 0.3812
- Token accuracy at step 80: 6.65%
- Teacher top-1 agreement at step 80: 9.10%
- Target rank mean at step 80: 4225.1
- Target margin mean at step 80: -4.6129
- Spike rate mean at step 80: 29.27%
- Readout scale at step 80: 0.9844
- Logit std at step 80: 1.4995
- Last 25-step token accuracy mean: 4.22%
- Last 25-step teacher top-1 agreement mean: 6.97%
- Last 25-step target rank mean: 3882.0
- Last 25-step target margin mean: -4.8487

## Decision

- Mixed but fail against the current small-batch best baseline.
- The candidate slightly improved the last-25 hard/soft means and improved step-80 token accuracy and teacher agreement, but the primary step-80 hard and soft losses are worse than loop16, and the gains are not clearly above gate noise.
- Do not launch a long probe or full training from this result.
- Do not merge `loop27-direct-attention-rate` into `main`.
- Keep loop16 as the current small-batch best baseline.

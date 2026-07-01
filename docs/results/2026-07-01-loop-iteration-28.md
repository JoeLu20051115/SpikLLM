# Loop Iteration 28 - Unscaled Embedding Alignment Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The EA loss currently aligns the student `embedding_states` after `input_scale` and temporal ramping, while the teacher embedding target is the raw OPT token-plus-position embedding. With `input_scale = 50` and `time_steps = 4`, the temporal fusion of student embedding states is 31.25x the raw teacher-space embedding. Because the student token embedding is tied to the LM head, this can make EA gradients fight the hard/soft output losses by pulling the shared output head away from the teacher scale. This iteration tests the single-variable fix of returning unscaled token-plus-position embeddings for EA while preserving the scaled/ramped current as the actual SNN block input.

## Candidate Code

- Branch: `loop28-unscaled-embedding-alignment`
- Commit: `6fb1d26`
- Remote branch: `origin/loop28-unscaled-embedding-alignment`
- Diff summary:
  - `BiSpikModel` now keeps `embedding_states` in teacher embedding scale for SpAD EA.
  - The block input current still uses `input_scale` and temporal ramping.
  - Added regression coverage that temporal fusion of `embedding_states` equals raw token-plus-position embeddings while `hidden_states[0]` still exposes the scaled current.

## Verification Before Gate

- Clean worktree baseline before edits:
  - `PATH=/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin:$PATH PYTHONPATH=$PWD /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest tests/smoke -q`
  - Result: 43 passed, 58 warnings.
- Root-cause diagnostic before implementation:
  - Raw embedding first token: `[0.08, 0.09, 0.10, 0.11]`
  - Temporal-fused student `embedding_states`: `[2.50, 2.8125, 3.125, 3.4375]`
  - Ratio: 31.25x, matching `input_scale * mean([1/4, 2/4, 3/4, 4/4])`.
- Red test before implementation:
  - `test_embedding_alignment_states_keep_teacher_scale_while_input_current_is_scaled`
  - Result: failed as expected because `embedding_states` were scaled by 31.25x.
- Targeted tests after implementation:
  - `test_embedding_alignment_states_keep_teacher_scale_while_input_current_is_scaled`: passed.
  - `test_lm_forward_returns_tensor_features`: passed.
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

- Candidate: unscaled teacher-space embedding states for EA.
- Run: `loop28-unscaled-emb-small-seq512-bs2-ga16-1xh200-20260701-230105`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/pl9tpsmd
- Step 80 hard/soft: 7.4513 / 4.5219
- Last 25-step hard/soft means: 7.7803 / 4.2960
- Step 80 embedding/attention/feature losses: 0.00007 / 0.5524 / 0.4256
- Step 80 attention rate/MSE losses: 0.7331 / 0.3718
- Token accuracy at step 80: 4.60%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4089.2
- Target margin mean at step 80: -4.6302
- Spike rate mean at step 80: 27.46%
- Readout scale at step 80: 0.9839
- Logit std at step 80: 1.5116
- Last 25-step token accuracy mean: 4.40%
- Last 25-step teacher top-1 agreement mean: 6.95%
- Last 25-step target rank mean: 3922.7
- Last 25-step target margin mean: -4.8422

## Decision

- Mixed but fail against the current small-batch best baseline.
- The candidate fixed the EA scale mismatch and slightly improved step-80 hard loss and target rank, but step-80 soft loss, last-25 hard/soft means, token accuracy, teacher agreement, and target margin are worse than loop16.
- Do not launch a long probe or full training from this result.
- Do not merge `loop28-unscaled-embedding-alignment` into `main`.
- Keep loop16 as the current small-batch best baseline.

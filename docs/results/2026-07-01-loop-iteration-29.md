# Loop Iteration 29 - Paper-Scale SpAD Input Combo Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Loop27 and loop28 each tested one paper-faithful SpAD input-scale repair and produced mixed results. This iteration tests their interaction as a combined SpAD-definition candidate: SAA teacher Rate-MSE uses teacher attention probabilities directly as constant input current, and EA uses unscaled teacher-space token-plus-position embeddings rather than scaled SNN input current. The actual SNN block input remains scaled and temporally ramped. SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, initialization, reset behavior, and output readout stay unchanged.

## Candidate Code

- Branch: `loop29-paper-spad-combo`
- Commit: `f90ca8a`
- Remote branch: `origin/loop29-paper-spad-combo`
- Diff summary:
  - `_attention_rate_drive` returns teacher attention probabilities unchanged.
  - `BiSpikModel` returns teacher-scale `embedding_states` for EA while preserving scaled/ramped current in `hidden_states[0]`.
  - Added a combined regression test that both SAA and EA inputs match teacher scales.
  - Updated older smoke assertions that encoded the previous row-max SAA drive and ramped EA semantics.

## Verification Before Gate

- Clean worktree baseline before edits:
  - `PATH=/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin:$PATH PYTHONPATH=$PWD /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest tests/smoke -q`
  - Result: 43 passed, 58 warnings.
- Red test before implementation:
  - `test_spad_alignment_inputs_match_teacher_scales`
  - Result: failed as expected because teacher attention was amplified from `[0.20, 0.10]` to `[0.70, 0.35]`; the same test also guards the EA scale mismatch.
- Targeted tests after implementation:
  - `test_spad_alignment_inputs_match_teacher_scales`: passed.
  - `test_lm_forward_returns_tensor_features`: passed.
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

- Candidate: direct teacher attention Rate-MSE drive plus teacher-scale EA inputs.
- Run: `loop29-paper-spad-combo-small-seq512-bs2-ga16-1xh200-20260701-231028`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/s6etbhxq
- Step 80 hard/soft: 7.4694 / 4.5497
- Last 25-step hard/soft means: 7.7804 / 4.3022
- Step 80 embedding/attention/feature losses: 0.00007 / 0.5405 / 0.4256
- Step 80 attention rate/MSE losses: 0.7219 / 0.3592
- Token accuracy at step 80: 5.19%
- Teacher top-1 agreement at step 80: 9.78%
- Target rank mean at step 80: 4179.8
- Target margin mean at step 80: -4.5840
- Spike rate mean at step 80: 27.30%
- Readout scale at step 80: 0.9836
- Logit std at step 80: 1.5256
- Last 25-step token accuracy mean: 4.12%
- Last 25-step teacher top-1 agreement mean: 7.05%
- Last 25-step target rank mean: 3903.5
- Last 25-step target margin mean: -4.8394

## Decision

- Fail against the current small-batch best baseline.
- The candidate improved teacher agreement and produced the expected near-zero EA loss, but step-80 hard/soft losses and last-25 hard/soft means are worse than loop16.
- Do not launch a long probe or full training from this result.
- Do not merge `loop29-paper-spad-combo` into `main`.
- Keep loop16 as the current small-batch best baseline.

# Loop Iteration 22 - Mean Step Logits Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The current LM readout temporally averages the final hidden state first, then applies final LayerNorm and the tied LM head. Because the student is explicitly temporal, this may blur per-step spike states before the nonlinear LayerNorm. This iteration tests a narrow output-path candidate: apply final LayerNorm and the tied LM head to each time step, then average the per-step logits. The change keeps SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, readout scale, data source, and SFSA/SFA/SAA paths unchanged.

## Candidate Code

- Branch: `loop22-mean-step-logits`
- Commit: `0b4a575`
- Remote branch: `origin/loop22-mean-step-logits`
- Diff summary:
  - `BiSpikForCausalLM` now requests temporal hidden states internally for LM readout.
  - Temporal readout computes per-step `final_layer_norm -> lm_head` logits, averages logits across time, and keeps the public `output_hidden_states` return contract unchanged.
  - Added a regression test proving the LM head fuses per-step logits after LayerNorm rather than applying LayerNorm after hidden-state temporal averaging.

## Verification Before Gate

- Red test before implementation:
  - `PYTHONPATH=$PWD .../.venv/bin/python -m pytest tests/smoke/test_paper_faithful_pipeline.py::test_lm_head_temporally_fuses_step_logits_after_layer_norm -q`
  - Result: failed as expected because the baseline used mean-hidden readout.
- Targeted tests after implementation:
  - `test_lm_head_temporally_fuses_step_logits_after_layer_norm`: passed.
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

- Candidate: mean per-step logits after final LayerNorm and LM head.
- Run: `loop22-mean-step-logits-small-seq512-bs2-ga16-1xh200-20260701-221142`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/45l4byhn
- Step 80 hard/soft: 7.4907 / 4.5766
- Last 25-step hard/soft means: 7.7819 / 4.3090
- Token accuracy at step 80: 4.79%
- Teacher top-1 agreement at step 80: 6.95%
- Target rank mean at step 80: 4082.7
- Target margin mean at step 80: -4.5771
- Spike rate mean at step 80: 28.70%
- Readout scale at step 80: 0.9837

## Decision

- Fail against the current small-batch best baseline.
- The candidate improved teacher agreement, target rank, and target margin, but worsened the primary hard and soft losses at step 80 and over the last 25-step window.
- Do not merge `loop22-mean-step-logits` into `main`.
- Keep loop16 as the current small-batch best baseline.

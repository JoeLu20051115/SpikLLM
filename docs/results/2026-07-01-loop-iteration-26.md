# Loop Iteration 26 - Raw Attention MSE Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The paper's Spike-Aware Alignment Distillation describes attention alignment as a direct MSE between temporally fused student attention and teacher ANN attention, plus Rate-MSE on the attention spike rate. The current implementation normalized both tensors into attention distributions before MSE, which can change the loss geometry and hide amplitude errors. This iteration tests a paper-faithful single-variable change: use raw tensor `F.mse_loss` for both attention-rate and attention-MSE terms, keeping SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, initialization, reset behavior, and output readout unchanged.

## Candidate Code

- Branch: `loop26-raw-attention-mse`
- Commit: `172b323`
- Remote branch: `origin/loop26-raw-attention-mse`
- Diff summary:
  - Replace distribution-normalized attention Rate-MSE with raw `F.mse_loss(student_rate, teacher_rate)`.
  - Replace distribution-normalized attention MSE with raw `F.mse_loss(student_attention, teacher_attention)`.
  - Added a regression test that distinguishes raw attention MSE from the previous distribution-normalized MSE.

## Verification Before Gate

- Red test before implementation:
  - `PYTHONPATH=$PWD .../.venv/bin/python -m pytest tests/smoke/test_scaffold.py::test_spad_attention_alignment_uses_raw_mse_not_distribution_mse -q`
  - Result: failed as expected because baseline attention alignment returned distribution-normalized MSE instead of the raw MSE target.
- Targeted tests after implementation:
  - `test_spad_attention_alignment_uses_raw_mse_not_distribution_mse`: passed.
  - `test_spad_five_loss_backward_with_temporal_fusion`: passed.
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

- Candidate: raw attention MSE and raw attention Rate-MSE.
- Run: `loop26-raw-attnmse-small-seq512-bs2-ga16-1xh200-20260701-224030`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/2n5kk8f5
- Step 80 hard/soft: 7.4718 / 4.5548
- Last 25-step hard/soft means: 7.7884 / 4.3173
- Step 80 embedding/attention/feature losses: 2.8577 / 0.0254 / 0.3293
- Token accuracy at step 80: 4.70%
- Teacher top-1 agreement at step 80: 8.02%
- Target rank mean at step 80: 3999.7
- Target margin mean at step 80: -4.5412
- Spike rate mean at step 80: 15.18%
- Readout scale at step 80: 0.9861
- Logit std at step 80: 1.4982

## Decision

- Fail against the current small-batch best baseline.
- The candidate improved teacher top-1 agreement, target rank, and target margin, and made the attention loss much smaller, but it worsened the primary hard and soft losses at step 80 and over the last 25 steps.
- Do not merge `loop26-raw-attention-mse` into `main`.
- Keep loop16 as the current small-batch best baseline.

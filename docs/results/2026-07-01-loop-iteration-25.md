# Loop Iteration 25 - Attention Projection Bias Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Loop24 improved representation losses but not output hard/soft losses. This iteration tests a capacity and firing-threshold adjustment path: the current SFSA `q_proj`, `k_proj`, `v_proj`, and `out_proj` layers are bias-free, while OPT-family attention projections are bias-capable. A zero-initialized trainable bias does not change the initial forward pass, but can let training quickly shift channel firing behavior. The candidate adds zero-initialized trainable biases to SFSA projections, keeping SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, initialization, reset behavior, and output readout unchanged.

## Candidate Code

- Branch: `loop25-attention-bias`
- Commit: `86a22b4`
- Remote branch: `origin/loop25-attention-bias`
- Diff summary:
  - Make SFSA `q_proj`, `k_proj`, `v_proj`, and `out_proj` bias-capable.
  - Zero-initialize all added SFSA projection biases.
  - Added a regression test that the SFSA projection biases are trainable and zero-initialized.

## Verification Before Gate

- Red test before implementation:
  - `PYTHONPATH=$PWD .../.venv/bin/python -m pytest tests/smoke/test_paper_faithful_pipeline.py::test_sfsa_projection_biases_are_trainable_and_zero_initialized -q`
  - Result: failed as expected because baseline SFSA projections used `bias=False`.
- Targeted tests after implementation:
  - `test_sfsa_projection_biases_are_trainable_and_zero_initialized`: passed.
  - `test_attention_is_causal_and_respects_padding_mask`: passed.
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

- Candidate: zero-initialized trainable SFSA projection biases.
- Run: `loop25-attnbias-small-seq512-bs2-ga16-1xh200-20260701-223152`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/m2lafom7
- Step 80 hard/soft: 7.4813 / 4.5625
- Last 25-step hard/soft means: 7.7782 / 4.3041
- Token accuracy at step 80: 5.19%
- Teacher top-1 agreement at step 80: 6.16%
- Target rank mean at step 80: 4164.2
- Target margin mean at step 80: -4.6269
- Spike rate mean at step 80: 29.40%
- Readout scale at step 80: 0.9846

## Decision

- Fail against the current small-batch best baseline.
- The candidate improved step-80 token accuracy and teacher agreement, but worsened the primary hard/soft losses, target rank, and target margin.
- Do not merge `loop25-attention-bias` into `main`.
- Keep loop16 as the current small-batch best baseline.

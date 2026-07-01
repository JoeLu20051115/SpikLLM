# Loop Iteration 24 - Transformer-Scale Linear Init Small-Batch Gate

Date: 2026-07-01

## Hypothesis

The previous reset-gradient candidate stayed close to loop16 but did not improve the output losses. This iteration tests a different initialization root cause: only token and position embeddings currently use `initializer_range`, while SFSA and SFFN linear layers inherit PyTorch defaults. OPT-family initialization uses transformer-scale weights and zero biases; random MLP biases can inject input-independent current into spiking neurons. The candidate initializes SFSA/SFFN linear weights with `normal_(0, initializer_range)` and zeros SFFN biases, keeping SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, reset behavior, and output readout unchanged.

## Candidate Code

- Branch: `loop24-transformer-linear-init`
- Commit: `de77112`
- Remote branch: `origin/loop24-transformer-linear-init`
- Diff summary:
  - Initialize SFSA `q_proj`, `k_proj`, `v_proj`, and `out_proj` weights from `normal_(0, initializer_range)`.
  - Initialize SFFN `fc1` and `fc2` weights from `normal_(0, initializer_range)`.
  - Zero SFFN linear biases.
  - Added a regression test covering transformer-scale linear initialization and zero bias.

## Verification Before Gate

- Red test before implementation:
  - `PYTHONPATH=$PWD .../.venv/bin/python -m pytest tests/smoke/test_scaffold.py::test_spiking_linear_layers_use_transformer_scale_initialization -q`
  - Result: failed as expected because baseline linears used PyTorch default initialization.
- Targeted tests after implementation:
  - `test_spiking_linear_layers_use_transformer_scale_initialization`: passed.
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

- Candidate: transformer-scale SFSA/SFFN linear initialization.
- Run: `loop24-linearinit-small-seq512-bs2-ga16-1xh200-20260701-222540`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/u5ww7r9b
- Step 80 hard/soft: 7.4643 / 4.5444
- Last 25-step hard/soft means: 7.7733 / 4.2969
- Token accuracy at step 80: 4.50%
- Teacher top-1 agreement at step 80: 5.38%
- Target rank mean at step 80: 4190.1
- Target margin mean at step 80: -4.5828
- Spike rate mean at step 80: 21.32%
- Readout scale at step 80: 0.9828
- Step 80 representation losses improved versus loop16 qualitatively, but the output losses did not.

## Decision

- Fail against the current small-batch best baseline.
- The candidate improved auxiliary representation losses and step-80 target margin, but primary hard/soft losses, token accuracy, teacher agreement, and target rank did not beat loop16.
- Do not merge `loop24-transformer-linear-init` into `main`.
- Keep loop16 as the current small-batch best baseline.

# Loop Iteration 23 - LIF Reset Gradient Small-Batch Gate

Date: 2026-07-01

## Hypothesis

Several recent candidates only changed readout or current scaling and did not beat loop16. This iteration tests a different root-cause area: BPTT through spiking neuron reset. The paper's BPTT appendix explicitly discusses eligibility traces under reset, while the current implementation sets all SpikingJelly `LIFNode` instances to `detach_reset=True`, cutting the reset path from the gradient graph. The candidate changes all model LIF nodes to `detach_reset=False`, keeping SpAD loss weights, temperature, optimizer, schedule, labels, teacher logits, data source, and output readout unchanged.

## Candidate Code

- Branch: `loop23-reset-gradient`
- Commit: `f96659a`
- Remote branch: `origin/loop23-reset-gradient`
- Diff summary:
  - Set `detach_reset=False` for SFSA Q/K/V, attention, attention-output, and output LIF nodes.
  - Set `detach_reset=False` for SFFN LIF.
  - Set `detach_reset=False` for the block output LIF.
  - Added a regression test that all model LIF nodes keep the reset path in BPTT.

## Verification Before Gate

- Red test before implementation:
  - `PYTHONPATH=$PWD .../.venv/bin/python -m pytest tests/smoke/test_paper_faithful_pipeline.py::test_spiking_neurons_keep_reset_path_in_bptt -q`
  - Result: failed as expected because baseline LIF nodes used `detach_reset=True`.
- Targeted tests after implementation:
  - `test_spiking_neurons_keep_reset_path_in_bptt`: passed.
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

- Candidate: `detach_reset=False` for model LIF nodes.
- Run: `loop23-reset-gradient-small-seq512-bs2-ga16-1xh200-20260701-221846`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/t0b2da59
- Step 80 hard/soft: 7.4557 / 4.5228
- Last 25-step hard/soft means: 7.7912 / 4.3040
- Token accuracy at step 80: 4.50%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4125.5
- Target margin mean at step 80: -4.6893
- Spike rate mean at step 80: 35.62%
- Readout scale at step 80: 0.9864

## Decision

- Fail against the current small-batch best baseline.
- The candidate slightly improved step-80 target rank, but primary hard/soft losses did not beat loop16, and token accuracy, teacher agreement, and target margin worsened.
- Do not merge `loop23-reset-gradient` into `main`.
- Keep loop16 as the current small-batch best baseline.

# Loop Iteration 32 - Mean-Preserving Constant Temporal Input

Date: 2026-07-01

## Hypothesis

Loop18 considered constant temporal input but was aborted before training because an older smoke invariant expected embedding states to differ across time. The paper's LIF and Rate-MSE descriptions use a constant input current and let membrane dynamics produce temporal spike behavior. This loop retests that idea without changing the total average input drive: replace the manual ramp `1/T, 2/T, ..., T/T` with a constant scale equal to the ramp mean, `(T + 1) / (2T)`.

This keeps the temporal fusion of the embedding current unchanged while making the per-step input current constant.

## Code Change

- Branch: `loop32-constant-mean-input`
- Commit: `f6388a2` (`fix: use mean-preserving constant temporal input`)
- `BiSpikModel.forward` now feeds each time step with `base_embedding * ((T + 1) / (2T))`.
- Updated the smoke invariant so embedding current is constant across time while final spike hidden states still differ through LIF dynamics.

## Verification

- RED test: `test_lm_forward_returns_tensor_features` failed before implementation because `embedding_states[0]` and `embedding_states[1]` still differed under the ramp.
- Targeted GREEN: `test_lm_forward_returns_tensor_features` passed.
- Full smoke before gate: `43 passed, 58 warnings`.
- Fixed-batch pre-gate screen was positive:
  - Loop16 reference from the same script: final hard/soft 5.2881 / 1.0851, last-8 hard/soft 5.3367 / 1.1523.
  - Loop32 candidate: final hard/soft 4.8418 / 0.9931, last-8 hard/soft 4.9201 / 1.0362.

## Small-Batch Gate

Matched geometry:
- GPU: 1x H200
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0 before entering the training CLI.

Current baseline:
- Code state: loop16 identity-projector candidate, commit `9c0d64f`
- W&B: `cvxuw267`
- Step 80 hard/soft: 7.4532 / 4.5195
- Last 25-step hard/soft means: 7.7798 / 4.2932
- Token accuracy at step 80: 4.79%
- Teacher top-1 agreement at step 80: 6.07%
- Target rank mean at step 80: 4162.4
- Target margin mean at step 80: -4.6015

Loop32 candidate:
- Run: `loop32-constmean-small-seq512-bs2-ga16-1xh200-20260701-234640`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/irfnofhl
- Step 80 hard/soft: 7.4716 / 4.5489
- Last 25-step hard/soft means: 7.7771 / 4.2980
- Token accuracy at step 80: 4.50%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4214.9
- Target margin mean at step 80: -4.7069
- Spike rate mean at step 80: 26.16%
- Readout scale at step 80: 0.9841
- Logit std at step 80: 1.5125

## Decision

Fail the loop32 candidate. The fixed-batch screen did not transfer to the streaming small-batch gate. Step-80 hard and soft losses are worse than loop16, last-25 soft is worse, and token accuracy, teacher agreement, target rank, and target margin all regress. The slightly better last-25 hard mean is not sufficient.

Do not merge loop32 code into `main`, and do not launch a long probe. Keep loop16 as the current small-batch best baseline.

Next direction: do not keep tuning temporal input drive by simple scale/ramp variants. The next loop should inspect loss-gradient conflict, especially the teacher-initialized embedding / tied LM-head path, because output hard/soft gradients are much smaller than auxiliary embedding gradients on the shared token matrix.

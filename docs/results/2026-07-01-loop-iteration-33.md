# Loop Iteration 33 - Spike-Encoded Embedding Input

Date: 2026-07-01

## Hypothesis

The SFSA appendix defines the attention input as spike-based `X`, but the current first BiSpik block receives the scaled analog embedding current directly. This means the first block's residual path can mix analog embedding current with spike outputs before the first output LIF. Loop33 tests a narrower paper-fidelity repair: spike-encode the embedding input current before it enters the block stack, while preserving the pre-spike embedding current in `embedding_states` for EA.

## Code Change

- Branch: `loop33-spike-encoded-input`
- Commit: `e3ac87d` (`fix: spike encode embedding input current`)
- Added `BiSpikModel.input_lif` with the same threshold, decay, surrogate, and reset behavior as the other model LIF nodes.
- `hidden_states[0]` / first block input is now binary spike output from `input_lif(input_current)`.
- `embedding_states` still stores the analog embedding current used by EA, preserving the prior EA semantics for this loop.

## Verification

- RED test: `test_lm_forward_returns_tensor_features` failed before implementation because `hidden_states[0]` contained analog values.
- Targeted GREEN: `test_lm_forward_returns_tensor_features` passed.
- Full smoke before gate: `43 passed, 58 warnings`.
- Fixed-batch pre-gate screen was mixed:
  - Loop16 reference from the same script: final hard/soft 5.2881 / 1.0851, last-8 hard/soft 5.3367 / 1.1523.
  - Loop33 candidate: final hard/soft 5.2615 / 1.1542, last-8 hard/soft 5.2881 / 1.1783.
  - Input spike rate in the fixed-batch screen averaged 27.33%.

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

Loop33 candidate:
- Run: `loop33-inputspike-small-seq512-bs2-ga16-1xh200-20260701-235455`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/aheyou6l
- Step 80 hard/soft: 7.4615 / 4.5432
- Last 25-step hard/soft means: 7.7771 / 4.2983
- Token accuracy at step 80: 6.07%
- Teacher top-1 agreement at step 80: 8.12%
- Target rank mean at step 80: 4112.6
- Target margin mean at step 80: -4.5752
- Spike rate mean at step 80: 28.34%
- Readout scale at step 80: 0.9831
- Logit std at step 80: 1.5156

## Decision

Fail the loop33 candidate. It improves several secondary output diagnostics at step 80, including token accuracy, teacher top-1 agreement, target rank, and target margin, but the primary hard/soft losses are worse than loop16 and last-25 soft is also worse. This is not a clear small-batch win.

Do not merge loop33 code into `main`, and do not launch a long probe. Keep loop16 as the current small-batch best baseline.

Next direction: loop33 suggests making the first block more spike-faithful can improve agreement/margin but not the primary losses. The next loop should avoid another isolated input-spike variant and instead investigate why hard/soft remain insensitive despite better agreement, with attention to the tied embedding/output head and EA gradient conflict.

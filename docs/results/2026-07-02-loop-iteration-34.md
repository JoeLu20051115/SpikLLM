# Loop Iteration 34 - Spike-Encoded Input with Teacher-Scale EA

Date: 2026-07-02

## Hypothesis

Loop33 made the first block input spike-faithful and improved several secondary output diagnostics, but did not improve primary hard/soft losses. A remaining issue is that EA still aligns the scaled/ramped input current rather than the raw OPT token-plus-position embedding. With `input_scale = 50` and `T=4`, the temporal-fused EA source is 31.25x the raw teacher embedding, creating a large auxiliary gradient on the token matrix that is tied to the LM head.

Loop34 combines the paper-consistent input spike encoding from loop33 with teacher-scale embedding alignment from loop28:
- block input is spike-encoded before the first BiSpik block;
- `embedding_states` stays at raw teacher embedding scale for EA.

## Code Change

- Branch: `loop34-inputspike-teacherscale-ea`
- Commit: `9581a36` (`fix: spike encode input while keeping ea teacher scale`)
- Added `BiSpikModel.input_lif`.
- `hidden_states[0]` is now the binary spike-encoded input to the block stack.
- `embedding_states` stores masked raw token-plus-position embeddings, not the scaled/ramped input current.

## Verification

- RED test: `test_embedding_alignment_stays_teacher_scale_while_block_input_is_spiking` failed before implementation because EA was still 6.25x raw embedding in the test setup.
- Targeted GREEN: `test_embedding_alignment_stays_teacher_scale_while_block_input_is_spiking` passed.
- Full smoke before gate: `44 passed, 58 warnings`.
- Fixed-batch pre-gate screen was mixed:
  - Loop16 reference from the same script: final hard/soft 5.2881 / 1.0851, last-8 hard/soft 5.3367 / 1.1523.
  - Loop34 candidate: final hard/soft 5.0238 / 1.1592, last-8 hard/soft 5.1130 / 1.2431.
  - Embedding loss was effectively removed in the fixed-batch screen.

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

Loop34 candidate:
- Run: `loop34-inputspike-eascale-small-seq512-bs2-ga16-1xh200-20260702-000224`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/g1vzm8l1
- Step 80 hard/soft: 7.4590 / 4.5277
- Last 25-step hard/soft means: 7.7810 / 4.2994
- Token accuracy at step 80: 5.09%
- Teacher top-1 agreement at step 80: 8.41%
- Target rank mean at step 80: 4222.6
- Target margin mean at step 80: -4.6269
- Spike rate mean at step 80: 30.91%
- Readout scale at step 80: 0.9837
- Logit std at step 80: 1.5124

## Decision

Fail the loop34 candidate. Teacher agreement improves, but the primary hard/soft losses and last-25 hard/soft means are all worse than loop16. Target rank and margin also do not support promotion.

Do not merge loop34 code into `main`, and do not launch a long probe. Keep loop16 as the current small-batch best baseline.

Next direction: the input-spike/EA scale family is not enough. Further loops should avoid combining these variants again and instead inspect output loss leverage directly, such as module-wise gradient routing or a paper-consistent way to increase STA/HTA influence without changing the five-loss objective definitions.

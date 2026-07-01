# Loop Iteration 31 - Final Readout LayerNorm Teacher Init

Date: 2026-07-01

## Hypothesis

Loops 27-30 tested SpAD scale/fidelity variants and did not beat the current small-batch best baseline. Output diagnostics on a fixed initialization showed that the final LM readout only receives the time-averaged final spike state, with values limited to `{0, 0.25, 0.5, 0.75, 1.0}` at `T=4`. A fixed-batch prototype then compared OPT LayerNorm initialization variants and found that copying only the teacher final LayerNorm improved the short fixed-batch hard/soft losses, while copying internal OPT LayerNorms hurt hard loss.

This iteration retests the narrow output-path candidate on top of the loop16 identity-projector baseline: initialize only `student.final_layer_norm` from the OPT teacher final LayerNorm. It does not copy internal LayerNorms, attention weights, MLP weights, or change SFSA/SFFN/SpAD loss definitions.

## Code Change

- Branch: `loop31-output-readout-diagnostics`
- Commit: `7281de5` (`fix: initialize final readout norm from teacher`)
- Added `initialize_student_final_layer_norm_from_teacher`.
- Called it after teacher token/position embedding initialization in `build_student_from_teacher`.
- Added RED/GREEN coverage that final readout LayerNorm is copied while internal student LayerNorms remain randomly initialized.

## Verification

- RED test failed before implementation because `student.final_layer_norm.weight` remained the default all-ones vector.
- Targeted GREEN: `test_student_final_layer_norm_initializes_from_teacher_readout_only` passed.
- Paper-faithful smoke subset: `15 passed`.
- Full smoke before gate: `44 passed, 58 warnings`.

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

Loop31 candidate:
- Run: `loop31-finalnorm-small-seq512-bs2-ga16-1xh200-20260701-233716`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/cda5wo1l
- Step 80 hard/soft: 7.4691 / 4.5429
- Last 25-step hard/soft means: 7.7794 / 4.2987
- Token accuracy at step 80: 4.50%
- Teacher top-1 agreement at step 80: 5.48%
- Target rank mean at step 80: 4155.2
- Target margin mean at step 80: -4.6579
- Spike rate mean at step 80: 29.16%
- Readout scale at step 80: 0.9842
- Logit std at step 80: 1.5151

## Decision

Fail the loop31 candidate. Step-80 hard and soft losses are both worse than loop16, last-25 soft is worse, and token accuracy, teacher top-1 agreement, and target margin all regress. The slightly better target rank and essentially tied last-25 hard mean are not enough to promote the candidate.

Do not merge loop31 code into `main`, and do not launch a long probe. Keep loop16 as the current small-batch best baseline.

Next direction: avoid repeating final-readout calibration and SpAD-scale variants. Focus on a materially different root cause in the student computation path that can change the output hard/soft plateau without broad teacher-weight initialization.

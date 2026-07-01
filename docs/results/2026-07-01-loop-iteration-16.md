# Loop Iteration 16 - Same-Dimension SpAD Projector Identity

Date: 2026-07-01

## Hypothesis

Loop15 showed lower embedding/feature losses than loop14 but worse hard/soft losses and worse target rank at comparable steps. This suggests the representation-alignment path can improve without forcing the raw student hidden states into the teacher/output space used by the tied LM head.

The BiSpikCLM paper describes lightweight projections for SpAD only when teacher and student dimensions, heads, or layers differ. In the OPT-125M reproduction path, teacher and student hidden dimensions match, but the current `SpADProjector` still applies a trainable LayerNorm to the student side. This iteration tests the minimal paper-faithful fix: make same-dimension SpAD projectors exact identities, while keeping projection+normalization only for dimension-mismatch cases.

## Planned Change

- Revert loop15 final layer norm teacher initialization.
- Make `SpADProjector(student_dim, teacher_dim)` an exact no-op when `student_dim == teacher_dim`.
- Skip DDP wrapping for projector modules with no trainable parameters.
- Keep SpAD weights, temperature, labels, teacher logits, optimizer, schedule, data source, and SFSA semantics unchanged.

## Verification Plan

- RED/GREEN smoke test: same-dimension `SpADProjector` returns the exact input tensor and has no trainable parameters.
- Smoke test: SpAD still supports mismatched student/teacher dimensions through the existing projection path.
- Full smoke command before probe: `.venv/bin/python -m pytest tests/smoke -q`.

## Pre-Probe Verification

- RED test: `test_same_dimension_spad_projector_is_identity` failed against the previous projector because it still applied LayerNorm in the same-dimension case.
- Targeted GREEN: `.venv/bin/python -m pytest tests/smoke/test_paper_faithful_pipeline.py::test_same_dimension_spad_projector_is_identity tests/smoke/test_paper_faithful_pipeline.py::test_trainable_parameter_detection_skips_identity_projectors -q` passed.
- Added identity-projector checkpoint round-trip coverage after review.
- Final smoke: `.venv/bin/python -m pytest tests/smoke -q` passed with 43 tests and 58 warnings.

## Review

Reviewer found no blocking issues for the loop16 diff. The reviewer noted that old pre-loop16 same-dimension projector checkpoints are not guaranteed to load because their projector LayerNorm weights no longer exist. This probe starts fresh, so no migration is required for loop16.

## Revised Probe Gate

The initial 700-step loop16 probe was stopped at step 17 because it bypassed the required small-batch A/B gate against the current best baseline.

From this iteration onward, every candidate change must first run a matched small-batch A/B comparison against the current small-batch best baseline:

- Baseline: the current best small-batch run by primary hard/soft output metrics.
- Candidate: the current loop change only.
- Geometry: same GPU count, sequence length, time steps, per-GPU batch size, gradient accumulation, precision, and max optimizer steps for both runs.
- Small-batch baseline update: if the candidate has the best primary hard/soft output metrics so far, record it as the new small-batch baseline even when the gain is small.
- Long-probe permission: a candidate may start a long probe or full training only if the small-batch result is clearly better, with support from token accuracy, teacher top-1 agreement, target-rank, and target-margin metrics.
- Fail condition: if candidate output metrics lag the current small-batch best, stop and revert the candidate before any long probe.

## Small-Batch A/B Gate Result

Gate geometry:
- GPU: 1x H200
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0 before entering the training CLI.

Baseline:
- Code state: loop14 baseline, commit `5e4d6df`
- Run: `loop16-gate-baseline14-small-seq512-bs2-ga16-1xh200-20260701-2134`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/pzih6ewb
- Step 80 hard/soft: 7.4622 / 4.5289
- Last 25-step hard/soft means: 7.7862 / 4.2996
- Teacher top-1 agreement at step 80: 7.05%
- Target rank mean at step 80: 4182.5
- Target margin mean at step 80: -4.6378

Candidate:
- Code state: loop16 identity-projector candidate, commit `9c0d64f`
- Run: `loop16-gate-candidate-small-seq512-bs2-ga16-1xh200-20260701-2134`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/cvxuw267
- Step 80 hard/soft: 7.4532 / 4.5195
- Last 25-step hard/soft means: 7.7798 / 4.2932
- Teacher top-1 agreement at step 80: 6.07%
- Target rank mean at step 80: 4162.4
- Target margin mean at step 80: -4.6015

Decision:
- Promote loop16 to the current small-batch best baseline because hard loss, soft loss, last-25 hard/soft means, target rank, and target margin are all slightly better than loop14.
- Do not launch a long probe from this result alone. The gain is small and teacher top-1 agreement is worse, so the next loop must beat loop16 in a matched small-batch gate before any long run.

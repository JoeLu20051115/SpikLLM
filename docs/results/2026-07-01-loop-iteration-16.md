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

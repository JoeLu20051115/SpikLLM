# Task 1 Report

## Status
Completed.

## Summary
Replaced the placeholder list-oriented language-model scaffold with a callable tensor-native `nn.Module` interface for `BiSpikForCausalLM` and `BiSpikModel`. The forward path now accepts `input_ids`, optional `attention_mask` and `labels`, and returns tensor-backed `logits`, optional `hidden_states`, optional `attentions`, optional `spike_stats`, `embedding_states`, and `loss`.

## Files Changed
- `bispikclm/models/bispik_model.py`
- `bispikclm/models/bispik_lm.py`
- `bispikclm/models/__init__.py`
- `tests/smoke/test_scaffold.py`
- `.superpowers/sdd/task-1-report.md`

## TDD Evidence
### Red
1. Updated `tests/smoke/test_scaffold.py` to replace the legacy list-based forward assertion with `test_lm_forward_returns_tensor_features` from the task brief.
2. Ran the target test in a torch-enabled interpreter:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features -v`
   - Result: `FAILED`
   - Failure: `TypeError: BiSpikForCausalLM object is not callable`
3. This confirmed the missing callable tensor-native forward contract before implementation.

### Green
1. Implemented `BiSpikModel(nn.Module)` with embedding lookup, simple per-layer attention/MLP blocks, optional hidden-state capture, optional attention capture, and minimal spike-stat placeholders.
2. Implemented `BiSpikForCausalLM(nn.Module)` to delegate to `BiSpikModel`, project logits, and compute optional causal LM loss.
3. Re-ran the target test:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features -v`
   - Result: `PASSED`

## Verification
Ran the full smoke scaffold file after the fix:
- Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py -v`
- Result: `4 passed in 0.94s`

## Notes
- `bispikclm/models/bispik_config.py` did not require a semantic change; the existing config already contained the required Task 1 fields and values.
- The default `/usr/bin/python3` environment in this workspace does not have `torch` installed, so verification used the existing torch-enabled Miniforge interpreter above.

## Concerns
- The per-layer spike statistics are intentionally minimal placeholders for Task 1. SFSA and SpAD-specific behavior remains out of scope until later tasks.

## Review Fix Addendum

### Findings Addressed
- Removed the parallel `_BiSpikLayer` path from `bispikclm/models/bispik_model.py` and rebuilt the model stack on `BiSpikBlock.from_config(...)`.
- Extended smoke coverage to exercise `labels`, `attention_mask`, `embedding_states`, and the `None` contract when optional outputs are not requested.

### Red
1. Extended `tests/smoke/test_scaffold.py` with:
   - stronger forward-contract assertions in `test_lm_forward_returns_tensor_features`
   - `test_lm_model_uses_bispik_block_stack`
2. Ran the focused review-fix targets:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features tests/smoke/test_scaffold.py::test_lm_model_uses_bispik_block_stack -v`
   - Result: `1 passed, 1 failed`
   - Failure: `assert all(isinstance(layer, BiSpikBlock) for layer in model.model.layers)`
3. This confirmed the Task 1 implementation was still bypassing the existing block scaffold.

### Green
1. Repaired `BiSpikAttention`, `BiSpikMLP`, and `BiSpikBlock` so the tensor-native path runs through the existing scaffold while preserving the legacy plain-`forward()` compatibility used by the smoke test.
2. Rebuilt `BiSpikModel.layers` from `BiSpikBlock.from_config(...)` and kept the Task 1 output contract intact.
3. Re-ran the focused smoke file:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py -v`
   - Result: `5 passed in 0.96s`


## Final Rereview Fix Addendum

### Findings Addressed
- Made the tensor attention path causal while retaining the optional key padding mask.
- Updated LM loss masking so shifted labels at masked positions are ignored.
- Added focused smoke regressions that fail without those two behaviors.

### Red
1. Extended `tests/smoke/test_scaffold.py` with:
   - `test_attention_is_causal_and_respects_padding_mask`
   - `test_lm_loss_ignores_masked_positions`
2. Ran the focused rereview targets:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_attention_is_causal_and_respects_padding_mask tests/smoke/test_scaffold.py::test_lm_loss_ignores_masked_positions -v`
   - Result: `2 failed`
   - Failures:
     - future-position attention weights were non-zero above the causal diagonal
     - `output["loss"]` did not match the mask-filtered expected LM loss

### Green
1. Added a lower-triangular causal score mask in `bispikclm/models/bispik_attention.py` before applying the optional padding mask.
2. Applied `attention_mask[..., 1:]` to shifted labels in `bispikclm/models/bispik_lm.py` before cross-entropy.
3. Re-ran the focused rereview targets:
   - Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_attention_is_causal_and_respects_padding_mask tests/smoke/test_scaffold.py::test_lm_loss_ignores_masked_positions -v`
   - Result: `2 passed in 0.93s`

### Verification
- Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py -v`
- Result: `7 passed in 0.96s`

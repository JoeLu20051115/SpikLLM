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

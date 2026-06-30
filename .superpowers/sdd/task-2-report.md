# Task 2 Report

## Status

Completed.

## Summary

Implemented a tensor-native, true per-head softmax-free spiking attention path. `BiSpikAttention` now projects Q/K/V into `[batch, heads, seq, head_dim]`, applies causal and padding masks, thresholds scores into binary attention spikes, normalizes over the active spike support without `torch.softmax`, and returns `context`, `attention_scores`, and `attention_spikes` for downstream distillation.

`BiSpikBlock` now consumes the attention dict and routes `attention_spikes` upward as the attention feature when requested.

## Files Changed

- `bispikclm/models/bispik_attention.py`
- `bispikclm/models/bispik_block.py`
- `tests/smoke/test_scaffold.py`

## TDD Evidence

### Red

- Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_attention_path_is_tensor_native_and_softmax_free -v`
- Result: failed with `AssertionError: BiSpikAttention must not call torch.softmax`
- Why expected: the previous tensor path still called `torch.softmax(scores, dim=-1)`.

### Green

- Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py::test_attention_path_is_tensor_native_and_softmax_free tests/smoke/test_scaffold.py::test_attention_is_causal_and_respects_padding_mask tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features -v`
- Result: `3 passed`

## Verification

- Command: `/mnt/data3/data_xingrui/miniforge3/bin/python -m pytest tests/smoke/test_scaffold.py -v`
- Result: `8 passed`

## Self-Review

- The implementation uses real per-head tensor structure rather than broadcasting a single score matrix.
- The causal mask and padding mask from Task 1 remain intact.
- The attention path no longer calls `torch.softmax`.

## Concerns

- This is still a minimal softmax-free spiking attention path. The later SpikingJelly task should replace the thresholding primitive with the configured surrogate-gradient spiking neuron where appropriate.

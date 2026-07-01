# Loop Iteration 18 - Constant Temporal Input Pre-Gate Abort

Date: 2026-07-01

## Hypothesis

The current model feeds a manually ramped embedding current into the spiking stack across time steps: step 1 receives `1/T`, step 2 receives `2/T`, and so on. A paper-style LIF rate-coding interpretation could instead use a constant input current and let the recurrent membrane state create temporal dynamics.

## Candidate

- Code candidate: replace `hidden_state = base_embedding * step_scale` with `hidden_state = base_embedding` in `BiSpikModel.forward`.
- No long probe was launched.
- No small-batch gate was launched.

## Pre-Gate Verification

Command:

```bash
/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.venv/bin/python -m pytest \
  tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features \
  tests/smoke/test_paper_faithful_pipeline.py::test_spad_attention_and_feature_losses_include_rate_mse_branches -q
```

Result:
- Failed before training.
- `test_lm_forward_returns_tensor_features` requires `embedding_states` for adjacent time steps to differ.
- The constant-input candidate makes `embedding_states[0]` and `embedding_states[1]` identical, violating the existing temporal-feature invariant.

## Decision

- Abort before any training run.
- Do not keep the constant temporal input candidate.
- Keep loop16 as the current small-batch best baseline.

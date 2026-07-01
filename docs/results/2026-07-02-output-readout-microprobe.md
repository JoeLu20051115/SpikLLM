# Output Readout Microprobe - No Loop Promotion

Date: 2026-07-02

## Purpose

The current LM logits are computed from the time-averaged final hidden state:

`last_hidden_state = mean_t H_T(t)`

This runtime-only probe tested whether decoding from the final time step instead of the temporal mean improves early hard/soft losses:

`last_hidden_state = H_T(T)`

SpAD still received the normal temporal `hidden_states`, `attentions`, and `embedding_states`. No repository code was changed for the probe.

## Setup

- Code baseline: `main` at `dec9c4f`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused across variants
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Results

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| last-step readout | 7.9807 | 4.5363 | 7.9346 | 4.5257 | +0.0242 / +0.0335 | +0.0395 / +0.0484 |

## Decision

Do not create loop36 from last-step readout. It worsens both primary hard and soft losses at the final step and over the last-10 window.

Keep loop16 as the current best baseline.

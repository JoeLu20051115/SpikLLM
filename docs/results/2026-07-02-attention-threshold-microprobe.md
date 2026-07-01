# Attention Threshold Microprobe - No Loop Promotion

Date: 2026-07-02

## Purpose

Loop35 tested an `SN_Attn` threshold floor of `1.0`. It improved a fixed dummy-batch screen, but failed the official 80-step streaming gate against loop16. Before creating another loop around larger attention thresholds, run a cheap streaming microprobe with identical data and initialization to decide whether thresholds `2.0` or `4.0` deserve a full loop.

This is a diagnostic screen, not a numbered loop. No repository code was changed for the variants.

## Microprobe Setup

- Code baseline: `main` at `1550098`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused for all variants
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
| `SN_Attn` threshold 2.0 | 7.9347 | 4.5028 | 7.9025 | 4.5136 | -0.0217 / -0.0000 | +0.0073 / +0.0363 |
| `SN_Attn` threshold 4.0 | 7.9285 | 4.4940 | 7.8912 | 4.4892 | -0.0279 / -0.0087 | -0.0039 / +0.0120 |

## Decision

Do not create a loop36 candidate from larger `SN_Attn` thresholds. Thresholds `2.0` and `4.0` only improve the final 20-step hard loss slightly, while the last-10 soft loss is worse than the base in both cases. This is below the required "clearly better on the small screen before an 80-step gate" bar.

Skip an official 80-step gate for these threshold-only variants. Keep loop16 as the current best baseline.

Next direction: investigate why fixed-batch improvements do not transfer to streaming, and prioritize output loss leverage or gradient-routing diagnostics over further attention-threshold-only changes.

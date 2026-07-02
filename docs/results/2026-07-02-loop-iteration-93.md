# Loop Iteration 93 - Residual Zero MLP Projector Screen80

Date: 2026-07-02

## Purpose

Observed 80-step extension of loop92's residual zero-initialized MLP projector screen.

This is still a small-batch candidate gate run. It must clearly beat the current best baseline before any full-run GPU time is justified.

## Setup

- tmux session: `loop93_residual_zero_mlp_screen80_gpu2`
- Script: `logs/loop93-residual-zero-mlp-screen80-gpu2-20260702-093025/runner.py`
- Log: `logs/loop93-residual-zero-mlp-screen80-gpu2-20260702-093025/train.log`
- Summary: `logs/loop93-residual-zero-mlp-screen80-gpu2-20260702-093025/summary.json`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `80`
- W&B: disabled

Candidate:

```text
hidden_projector = tensor + 0.1 * Linear2(GELU(Linear1(tensor)))
```

with `Linear2.weight` initialized to zero.

## Result

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step80 hard | 7.4572 | 7.4566 | -0.0006 |
| step80 soft | 4.5187 | 4.5299 | +0.0112 |
| last10 hard | 7.7706 | 7.7597 | -0.0110 |
| last10 soft | 4.2969 | 4.2936 | -0.0033 |
| last20 hard | 7.7759 | 7.7684 | -0.0075 |
| last20 soft | 4.3223 | 4.3172 | -0.0051 |
| step80 total | 4.2358 | 4.2341 | -0.0016 |
| step80 feature | 0.4267 | 0.3562 | -0.0705 |
| step80 token accuracy | 5.38% | 4.60% | -0.78 pp |
| step80 teacher top-1 agreement | 8.02% | 5.58% | -2.45 pp |
| step80 top-5 accuracy | 19.57% | 19.47% | -0.10 pp |
| step80 target rank mean | 4138.2 | 4066.9 | -71.3 |
| step80 target margin mean | -4.6461 | -4.6550 | -0.0088 |
| step80 spike rate | 27.34% | 32.67% | +5.33 pp |
| step80 grad norm | 0.5223 | 0.5908 | +0.0685 |

Decision: reject as a baseline replacement. The candidate has tiny hard/window improvements but worsens step80 soft loss, token accuracy, teacher agreement, top-5, spike rate, and grad norm. It does not satisfy the clear small-batch win rule.

Current best baseline remains loop16/current-source baseline, with loop14 three-GPU geometry as the best historical long-run setup.

# Loop Iteration 92 - Residual Zero MLP Projector Screen40

Date: 2026-07-02

## Purpose

Observed single-GPU 40-step small-screen comparing the current baseline against a residual zero-initialized MLP hidden projector.

This follows the user's gate rule: a candidate must clearly beat the current best baseline on a small run before using full-run GPU time.

## Setup

- tmux session: `loop92_residual_zero_mlp_screen40_gpu2`
- Script: `logs/loop92-residual-zero-mlp-screen40-gpu2-20260702-092615/runner.py`
- Log: `logs/loop92-residual-zero-mlp-screen40-gpu2-20260702-092615/train.log`
- Summary: `logs/loop92-residual-zero-mlp-screen40-gpu2-20260702-092615/summary.json`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `40`
- W&B: disabled

Candidate change:

```text
hidden_projector = tensor + 0.1 * Linear2(GELU(Linear1(tensor)))
```

with `Linear2.weight` initialized to zero.

The SpAD loss set and weights remained unchanged.

## Result

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step40 hard | 7.6710 | 7.6838 | +0.0128 |
| step40 soft | 4.4294 | 4.4143 | -0.0151 |
| last10 hard | 7.8349 | 7.8200 | -0.0149 |
| last10 soft | 4.4474 | 4.4292 | -0.0182 |
| last20 hard | 7.7931 | 7.7965 | +0.0034 |
| last20 soft | 4.4301 | 4.4406 | +0.0106 |
| step40 total | 4.3212 | 4.3175 | -0.0037 |
| step40 feature | 0.4751 | 0.3990 | -0.0761 |
| step40 token accuracy | 4.40% | 4.70% | +0.29 pp |
| step40 teacher top-1 agreement | 5.28% | 5.97% | +0.68 pp |
| step40 top-5 accuracy | 13.60% | 13.11% | -0.49 pp |
| step40 target rank mean | 4285.7 | 4062.7 | -223.1 |
| step40 target margin mean | -5.1024 | -5.0534 | +0.0490 |
| step40 spike rate | 33.16% | 39.36% | +6.19 pp |
| step40 grad norm | 0.4328 | 0.5093 | +0.0765 |

Decision: reject as a baseline replacement. The candidate improves some secondary metrics and last10 hard/soft slightly, but it is not clearly better: step40 hard is worse, last20 hard/soft are worse, top-5 is worse, and grad norm/spike rate are higher. Do not run this variant full-scale.

Current best baseline remains loop16/current-source baseline, with loop14 three-GPU geometry as the best historical long-run setup.

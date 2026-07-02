# Loop Iteration 85 - Same-Dim SFA MLP+LayerNorm Screen80

Date: 2026-07-02

## Purpose

Observed single-GPU 80-step extension of the same-dimension SFA MLP+LayerNorm projector screen from loop84.

This is a small-batch gate run only. It compares the candidate against the current baseline before any full run.

## Setup

- Script: `logs/loop85-sfa-mlp-ln-screen80-gpu2-20260702-090555/runner.py`
- Log: `logs/loop85-sfa-mlp-ln-screen80-gpu2-20260702-090555/train.log`
- Summary: `logs/loop85-sfa-mlp-ln-screen80-gpu2-20260702-090555/summary.json`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `80`
- W&B: disabled

Candidate change:

```text
hidden_projector = Linear(dim, dim, bias=False) -> GELU -> Linear(dim, dim, bias=False) -> LayerNorm(dim)
```

## Result

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step80 hard | 7.4572 | 7.4523 | -0.0050 |
| step80 soft | 4.5187 | 4.5240 | +0.0054 |
| last10 hard | 7.7706 | 7.7650 | -0.0057 |
| last10 soft | 4.2969 | 4.2911 | -0.0058 |
| last20 hard | 7.7759 | 7.7727 | -0.0033 |
| last20 soft | 4.3223 | 4.3143 | -0.0080 |
| step80 total | 4.2358 | 4.2701 | +0.0343 |
| step80 feature | 0.4267 | 0.7400 | +0.3133 |
| step80 token accuracy | 5.38% | 4.99% | -0.39 percentage points |
| step80 teacher top-1 agreement | 8.02% | 7.44% | -0.59 percentage points |
| step80 target rank mean | 4138.2 | 4119.3 | -19.0 |
| step80 target margin mean | -4.6461 | -4.5690 | +0.0772 |
| step80 spike rate | 27.34% | 34.62% | +7.28 percentage points |
| step80 grad norm | 0.5223 | 0.5420 | +0.0196 |

Decision: reject as a baseline replacement. The candidate is only marginally different on hard/soft, has worse soft at step80, worse total and feature loss, and lower token accuracy and teacher agreement. This does not satisfy the user's "clearly better than current best baseline on small-batch first" rule.

Current best baseline remains loop16/current-source baseline, with loop14 geometry as the best historical long-run setup.

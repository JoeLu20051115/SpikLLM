# Loop Iteration 84 - Same-Dim SFA MLP+LayerNorm Screen

Date: 2026-07-02

## Purpose

Observed single-GPU 40-step small-screen artifact comparing the current best baseline against a trainable same-dimension SFA hidden projector.

The screen followed the user's small-batch gate rule: compare against the current best baseline before considering any long/full run.

## Setup

- Script: `/tmp/spikllm_loop84_sfa_mlp_ln.py`
- Log: `logs/loop84-sfa-mlp-ln-screen40-gpu2-20260702-090241/train.log`
- Summary: `logs/loop84-sfa-mlp-ln-screen40/summary.json`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Data: FineWeb, fixed prefetched microbatches
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `40`
- W&B: disabled

Candidate change:

```text
hidden_projector = Linear(dim, dim, bias=False) -> GELU -> Linear(dim, dim, bias=False) -> LayerNorm(dim)
```

The embedding projector and all five SpAD loss weights remained unchanged.

## Result

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step40 hard | 7.6710 | 7.6750 | +0.0040 |
| step40 soft | 4.4294 | 4.3906 | -0.0388 |
| last10 hard | 7.8349 | 7.8141 | -0.0208 |
| last10 soft | 4.4474 | 4.4139 | -0.0335 |
| last20 hard | 7.7931 | 7.7885 | -0.0046 |
| last20 soft | 4.4301 | 4.4223 | -0.0078 |
| step40 total | 4.3212 | 4.3416 | +0.0204 |
| step40 feature | 0.4751 | 0.7637 | +0.2886 |
| step40 token accuracy | 4.40% | 2.05% | -2.35 pp |
| step40 teacher top-1 agreement | 5.28% | 3.23% | -2.05 pp |
| step40 target rank mean | 4285.7 | 4012.2 | -273.6 |
| step40 target margin mean | -5.1024 | -5.1834 | -0.0810 |
| step40 spike rate | 33.16% | 38.54% | +5.38 pp |
| step40 grad norm | 0.4328 | 4.4194 | +3.9866 |

Decision: reject as a baseline replacement. The candidate has tiny last-window hard/soft improvements, but it is not clearly better: step40 hard and total loss are worse, feature loss is materially worse, token accuracy and teacher agreement are worse, and gradient norm is much higher. Do not spend full-run GPU time on this variant.

Current best baseline remains the loop16/current-source baseline, with loop14 as the best historical long-run geometry.

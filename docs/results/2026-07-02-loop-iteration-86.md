# Loop Iteration 86 - Two-GPU Backward Smoke

Date: 2026-07-02

## Purpose

Observed two-GPU smoke test after loop83 exposed a rank-1 backward crash at the long-run two-GPU geometry.

## Setup

- Log: `logs/loop86-loop16-2gpu-backward-smoke-seq1024-bs4-ga1-20260702-090730/train.log`
- Source baseline: current `main` / loop16 baseline code
- Sequence length: `1024`
- Time steps: `4`
- Batch size: `4`
- Gradient accumulation: `1`
- GPUs: two
- W&B: disabled

## Result

The smoke emitted two loss dictionaries and did not show the CUBLAS illegal-memory-access failure seen in loop83.

Loss rows:

| Row | Hard | Soft | Total | Embedding | Attention | Feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.7265 | 7.3569 | 6.2085 | 3.3224 | 0.6171 | 0.5725 |
| 2 | 10.5929 | 7.1536 | 6.0888 | 3.2353 | 0.6080 | 0.5703 |

Decision: smoke only. This indicates the crash is not guaranteed on a single backward with `GA=1`, but it does not validate the long two-GPU `GA=64` setup. It does not update the baseline.

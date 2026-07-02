# Loop Iteration 94 - Residual Zero MLP Projector Screen160

Date: 2026-07-02

## Purpose

Observed 160-step extension of the residual zero-initialized MLP projector screen from loops92-93.

This is a small-batch candidate gate run only. It is not the requested three-GPU best-baseline run.

## Setup

- tmux session: `loop94_residual_zero_mlp_screen160_gpu2`
- Script: `logs/loop94-residual-zero-mlp-screen160-gpu2-20260702-093510/runner.py`
- Log: `logs/loop94-residual-zero-mlp-screen160-gpu2-20260702-093510/train.log`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Teacher: `facebook/opt-125m`
- Sequence length: `512`
- Time steps: `4`
- Batch size: `2`
- Gradient accumulation: `16`
- Optimizer steps: `160`
- W&B: disabled

Candidate:

```text
hidden_projector = tensor + 0.1 * Linear2(GELU(Linear1(tensor)))
```

with `Linear2.weight` initialized to zero.

## Interim State

At the first record, the run was still in progress. The base half had reached step35 in the local log:

- base step25 hard/soft: `7.7194 / 4.2881`;
- base step30 hard/soft: `7.8054 / 4.4754`;
- base step35 hard/soft: `7.7613 / 4.2908`.

No candidate result or final summary had been written yet.

Decision at record time: continue observing. Loops92-93 did not show a clear small-batch win for this candidate, so loop94 needed to show a materially stronger result before any baseline update or full run would be justified.

## Final Result

| Metric | Base | Candidate | Candidate - Base |
| --- | ---: | ---: | ---: |
| step160 hard | 7.9777 | 7.9782 | +0.0005 |
| step160 soft | 4.9140 | 4.9154 | +0.0014 |
| last10 hard | 7.8128 | 7.8135 | +0.0007 |
| last10 soft | 4.3154 | 4.3165 | +0.0011 |
| last20 hard | 7.8078 | 7.8093 | +0.0015 |
| last20 soft | 4.3340 | 4.3338 | -0.0002 |
| step160 total | 4.4824 | 4.4752 | -0.0072 |
| step160 feature | 0.3853 | 0.3159 | -0.0695 |
| step160 token accuracy | 1.76% | 1.76% | 0.00 pp |
| step160 teacher top-1 agreement | 2.25% | 2.25% | 0.00 pp |
| step160 top-5 accuracy | 14.29% | 14.29% | 0.00 pp |
| step160 target rank mean | 4608.8 | 4655.6 | +46.8 |
| step160 target margin mean | -4.9267 | -4.8779 | +0.0488 |
| step160 spike rate | 22.35% | 24.65% | +2.30 pp |
| step160 grad norm | 0.7221 | 0.7390 | +0.0169 |

Decision: reject as a baseline replacement. The 160-step extension confirms the residual zero MLP projector is not clearly better: step160 hard/soft and last10 hard/soft are all slightly worse, with only feature/total and margin showing small improvements. It does not satisfy the small-batch clear-win rule.

Current best baseline remains loop16/current-source baseline, with loop14 three-GPU geometry as the best historical long-run setup.

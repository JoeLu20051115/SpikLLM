# Loop Iteration 68 - Temporal STA/HTA Screen

Date: 2026-07-02

## Hypothesis

The paper's BPTT appendix describes the total loss as a sum of per-time-step losses, while the current local implementation computes logits only from the time-averaged final hidden state. EA, SAA, and SFA already use explicit temporal fusion, but STA and HTA are applied once after the final hidden-state average.

Loop68 tests whether applying the same STA and HTA definitions at every student time step, then averaging over time to preserve loss scale, improves the early hard/soft gate:

- keep teacher-copied token/position embeddings, tied LM head, SFSA/SFFN, EA/SAA/SFA, all five SpAD lambdas, optimizer, scheduler, labels, and teacher logits unchanged;
- return per-step final hidden states from the student;
- compute per-step student logits with the same final layer norm, tied LM head, and readout scale;
- replace only the STA/HTA aggregation with a per-step mean.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before the run

## Runs

- Base reference: first 20 rows from `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Candidate run: `loop68-temporal-sta-hta-screen20`
- Candidate W&B: `wandb/offline-run-20260702_074817-6ihb5jw4/run-6ihb5jw4.wandb`

## Result

| Variant | Step 20 hard | Step 20 soft | Last5 hard | Last5 soft | Last10 hard | Last10 soft | Full20 hard | Full20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9879 | 4.5508 | 7.9028 | 4.5642 | 7.9978 | 4.6208 | 8.7184 | 5.1846 |
| temporal STA/HTA | 8.0708 | 4.5631 | 7.9973 | 4.6031 | 8.1214 | 4.7167 | 8.8885 | 5.3333 |

Deltas for candidate vs base:

- Step 20 hard/soft: +0.0829 / +0.0123
- Last5 hard/soft: +0.0946 / +0.0390
- Last10 hard/soft: +0.1236 / +0.0959
- Full20 hard/soft: +0.1700 / +0.1487

Secondary diagnostics at step 20:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Hidden std | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 5.77% | 7.53% | 19.37% | 4868.9 | -4.8015 | 42.80% | 0.4561 | 1.5326 | 1.2597 | 0.9955 |
| temporal STA/HTA | 3.52% | 4.40% | 16.83% | 4967.9 | -5.2202 | 40.06% | 0.4809 | 1.6331 | 1.5032 | 0.9948 |

Window slopes:

| Variant | Last5 hard slope/100 | Last5 soft slope/100 | Last10 hard slope/100 | Last10 soft slope/100 | Full20 hard slope/100 | Full20 soft slope/100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | -0.6711 | -3.7062 | -2.1419 | -1.8595 | -15.8978 | -12.6514 |
| temporal STA/HTA | -1.0482 | -5.0905 | -3.0087 | -3.8406 | -16.7027 | -13.7830 |

## Decision

Fail loop68. Do not extend to 40 or 80 steps:

- It is worse than the matched base on every hard/soft metric window.
- Step20 hard and soft both regress.
- Last10 hard/soft both regress despite a stronger local downward slope.
- Token accuracy and teacher agreement regress at step20.
- The lower spike rate does not translate into better output alignment.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Per-step STA/HTA averaging is not a promotion candidate in this form.


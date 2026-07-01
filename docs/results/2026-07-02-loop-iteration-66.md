# Loop Iteration 66 - Random Embedding Input Scale Screen

Date: 2026-07-02

## Hypothesis

Loop63's random-embedding run was more paper-faithful and competitive, but its spike rate was lower than the current teacher-copy baseline. A scale diagnostic showed:

- teacher-copy token embedding std: 0.0554, so current scale50 gives token-current std about 2.77;
- random token embedding std: 0.0200, so current scale50 gives token-current std about 1.00;
- random embedding with input scale 125 gives token-current std about 2.50.

Loop66 tests whether the random-init under-firing is just an input-current scale mismatch:

- disable teacher token/position embedding copy, as in loop63;
- set `student.config.input_scale = 125.0`;
- keep tied LM head, SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Random-only reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-only W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Candidate run: `loop66-random-emb-inputscale125-screen40`
- Candidate W&B: `wandb/offline-run-20260702_073408-au99eizt/run-au99eizt.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 |
| random + input scale 125 | 7.6659 | 4.4200 | 7.8026 | 4.4074 | 7.7750 | 4.4161 | 7.8445 | 4.5000 |

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0051 / -0.0093
- Last10 hard/soft: -0.0323 / -0.0401
- Last20 hard/soft: -0.0181 / -0.0140
- Last25 hard/soft: +0.0295 / +0.0431

Deltas for candidate vs random-only:

- Step 40 hard/soft: +0.0183 / +0.0340
- Last10 hard/soft: +0.0001 / +0.0095
- Last20 hard/soft: -0.0039 / +0.0017
- Last25 hard/soft: -0.0027 / -0.0002
- Full40 hard/soft: -0.0027 / -0.0003

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% | 0.4328 | 0.9885 |
| random embedding | 4.50% | 5.48% | 4033.3 | -5.1189 | 25.28% | 2.2442 | 1.0062 |
| random + input scale 125 | 3.82% | 4.89% | 4247.6 | -4.9977 | 24.92% | 4.7462 | 1.0070 |

Auxiliary losses at step 40:

| Variant | EA loss | Attention loss | Feature loss |
| --- | ---: | ---: | ---: |
| random embedding | 0.5227 | 0.5860 | 0.3801 |
| random + input scale 125 | 2.8024 | 0.5811 | 0.3798 |

## Decision

Fail loop66. Do not extend to an 80-step gate:

- It does not beat loop63 random-only at the final step; step40 hard and soft both regress.
- Last10 soft and last20 soft regress vs random-only.
- Token accuracy and teacher agreement regress vs random-only.
- Spike rate remains low despite the higher input scale.
- The higher input scale mostly reintroduces the high EA loss that random initialization had avoided.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Random embedding plus input scale 125 is not a promotion candidate.

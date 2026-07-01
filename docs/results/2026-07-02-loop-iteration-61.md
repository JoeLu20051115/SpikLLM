# Loop Iteration 61 - Zero MLP Bias Screen

Date: 2026-07-02

## Hypothesis

Loop24 tested transformer-scale linear initialization and zeroed SFFN biases together, but the weight-scale change made the candidate substantially under-fire. Loop61 isolates the smaller initialization hypothesis:

- keep SFSA/SFFN weights at the current baseline initialization;
- keep all five SpAD losses and paper weights unchanged;
- after student construction, zero only the SFFN/MLP `fc1` and `fc2` biases.

This removes input-independent MLP current while avoiding the broader weight-scale change from loop24. This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 2x H200, base on GPU0 and candidate on GPU1
- Dataset: FineWeb-Edu streaming; base and candidate use matched seed and streaming order
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant
- Candidate changed tensors: 24 MLP bias tensors zeroed

## Result

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Candidate run: `loop61-zero-mlp-bias-screen40-zero-bias`
- Candidate W&B: `wandb/offline-run-20260702_070959-67jz3ut2/run-67jz3ut2.wandb`

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 |
| zero MLP bias | 7.6820 | 4.3992 | 7.8192 | 4.4236 | 7.7990 | 4.4295 | 7.8234 | 4.4576 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.0110 / -0.0301
- Last10 hard/soft: -0.0157 / -0.0238
- Last20 hard/soft: +0.0059 / -0.0006
- Last25 hard/soft: +0.0084 / +0.0007

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% | 0.9885 |
| zero MLP bias | 4.50% | 5.48% | 4157.3 | -5.2994 | 35.15% | 0.9885 |

Full 40-step means:

| Variant | Mean hard | Mean soft | Mean token acc | Mean teacher agreement | Mean target rank | Mean target margin | Mean spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 8.2558 | 4.8073 | 3.36% | 5.14% | 5326.7 | -5.1755 | 44.27% |
| zero MLP bias | 8.2630 | 4.8079 | 3.42% | 5.47% | 5303.8 | -5.3369 | 44.36% |

## Decision

Fail loop61. Do not extend to an 80-step gate:

- Last10 hard/soft improve, but the improvement does not survive broader windows.
- Step40 hard loss regresses by +0.0110.
- Last20 and last25 hard loss regress.
- Last25 soft loss also regresses slightly.
- Target margin is much worse at step 40 and across the full 40-step window.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Zeroing only the MLP biases is not a promotion candidate under the small-screen gate.

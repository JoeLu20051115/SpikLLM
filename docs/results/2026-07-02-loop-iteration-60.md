# Loop Iteration 60 - Reset Gradient Screen

Date: 2026-07-02

## Hypothesis

Current model LIF nodes are constructed with `detach_reset=True`, which removes the reset term from the backward path. The BiSpikCLM appendix derives BPTT dynamics with reset included in the eligibility trace, so loop60 tests whether preserving reset gradients improves early hard/soft behavior.

Candidate behavior:

- keep the forward dynamics unchanged;
- keep all five SpAD losses and paper weights unchanged;
- after student construction, set `detach_reset=False` on every model LIF node.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming; candidate compared to the immediately preceding matched base run
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant
- Scheduler horizon equals the 20-step screen length
- Candidate changed modules: 96 LIF nodes with `detach_reset=False`

## Result

- Base run: `loop59-temporal-logit-fusion-screen20-base`
- Base W&B: `wandb/offline-run-20260702_065913-4w33mojq/run-4w33mojq.wandb`
- Candidate run: `loop60-reset-grad-screen20-detach-false`
- Candidate W&B: `wandb/offline-run-20260702_070344-pllqtpjc/run-pllqtpjc.wandb`

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 5081.1 | -5.0093 | 45.11% |
| reset gradient | 7.9579 | 4.4973 | 7.9021 | 4.4822 | 6.26% | 7.24% | 5095.4 | -5.1718 | 42.15% |

Deltas for candidate vs base:

- Step 20 hard/soft: +0.0125 / -0.0105
- Last10 hard/soft: +0.0020 / +0.0022
- Step 20 token accuracy / teacher agreement: +2.25 pp / +1.17 pp
- Step 20 target rank / margin: +14.3 / -0.1625
- Step 20 spike rate: -2.96 pp

Full 20-step means:

| Variant | Mean hard | Mean soft | Mean token acc | Mean teacher agreement | Mean target rank | Mean target margin | Mean spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 8.5097 | 4.9765 | 3.62% | 6.21% | 6422.7 | -5.4513 | 51.28% |
| reset gradient | 8.5096 | 4.9820 | 3.86% | 5.69% | 6390.6 | -5.4564 | 49.63% |

Full-window candidate deltas:

- Mean hard/soft: -0.0002 / +0.0055
- Mean token accuracy / teacher agreement: +0.23 pp / -0.52 pp
- Mean target rank / margin: -32.0 / -0.0051
- Mean spike rate: -1.66 pp

## Decision

Fail loop60. Do not extend to 40 steps:

- The final soft loss improves, but final hard loss regresses.
- Last10 hard and soft both regress.
- Full-20 soft loss regresses.
- Teacher agreement regresses over the full 20-step window.
- Spike rate drops further, which does not address the loop55 under-firing relative to loop14.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Reset-gradient preservation is not a promotion candidate under the small-screen gate.

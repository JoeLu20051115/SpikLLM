# Loop Iteration 57 - Paper Dynamics Bundle Screen

Date: 2026-07-02

## Hypothesis

After projector/output-path variants failed, the next distinct hypothesis is that isolated paper-dynamics changes were too weak but a coherent block-level bundle might improve the small screen.

The candidate combines three paper-motivated runtime changes:

- replace the temporal input ramp with a mean-preserving constant input scale, `(T + 1) / (2T)`;
- set model LIF nodes to soft/subtractive reset via `v_reset=None`;
- set internal block `attention_norm` and `mlp_norm` to `Identity`.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming; base and candidate use matched seed and streaming order
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

- Base run: `loop57-paper-dynamics-bundle-screen20-base`
- Base W&B: `wandb/offline-run-20260702_064640-uj7cjg70/run-uj7cjg70.wandb`
- Candidate run: `loop57-paper-dynamics-bundle-screen20-paper-dynamics`
- Candidate W&B: `wandb/offline-run-20260702_064728-5vqyxy7d/run-5vqyxy7d.wandb`

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 45.11% |
| paper dynamics bundle | 7.9365 | 4.4590 | 7.9023 | 4.4440 | 6.36% | 8.81% | 43.41% |

Deltas for candidate vs base:

- Step 20 hard/soft: -0.0089 / -0.0488
- Last10 hard/soft: +0.0023 / -0.0361
- Step 20 token accuracy / teacher agreement: +2.35 pp / +2.74 pp
- Step 20 target rank / margin: -225.3 / -0.1199
- Step 20 spike rate: -1.70 pp

Full 20-step means:

| Variant | Mean hard | Mean soft | Mean token acc | Mean teacher agreement | Mean spike rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 8.5097 | 4.9765 | 3.62% | 6.21% | 51.28% |
| paper dynamics bundle | 9.6665 | 5.2193 | 3.20% | 5.26% | 53.63% |

The candidate has a much worse early phase, then recovers by step 20.

## Decision

Fail loop57. Do not extend to 40 steps:

- The final step improves, especially soft loss, but the improvement is not broad enough.
- Last10 hard loss is slightly worse than base.
- Last10 token accuracy and teacher agreement are worse than base.
- Full 20-step hard/soft means are much worse because the candidate starts from a bad early regime.
- The candidate does not correct loop55's under-firing at the final step; step-20 spike rate is lower than base.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. A paper-dynamics bundle may recover late, but it fails the required small-screen promotion bar.

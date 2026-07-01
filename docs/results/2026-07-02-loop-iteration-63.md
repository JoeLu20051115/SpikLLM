# Loop Iteration 63 - Paper Random Embedding Initialization Gate

Date: 2026-07-02

## Hypothesis

The arXiv source for BiSpikCLM states that SpAD enables the student SNN to be trained from random initialization. Current `main` copies the OPT teacher token and position embeddings into the student before training, which is a stability-oriented deviation from that statement.

Loop63 tests the paper-faithful variant:

- keep transformer-scale student token and position embedding initialization from `initializer_range=0.02`;
- disable teacher token/position embedding copy;
- keep tied LM head, SFSA/SFFN, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged.

This was a runtime monkeypatch only. No source code was changed.

## 40-Step Screen

Setup:

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming; compared against the matched loop61 base
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

Runs:

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Candidate run: `loop63-paper-random-embedding-screen40-random-emb`
- Candidate W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 |

Deltas for random embedding vs base:

- Step 40 hard/soft: -0.0234 / -0.0434
- Last10 hard/soft: -0.0324 / -0.0496
- Last20 hard/soft: -0.0142 / -0.0156
- Last25 hard/soft: +0.0321 / +0.0433

The candidate was promoted to an 80-step official gate because the final step, last10, and last20 hard/soft losses all improved. The last25/full-window means were worse because the random-initialized run starts from a worse early phase.

## 80-Step Gate

Run:

- Candidate run: `loop63-paper-random-embedding-gate80-random-emb`
- Candidate W&B: `wandb/offline-run-20260702_072056-vl15sdve/run-vl15sdve.wandb`

Current source-equivalent base:

- Base run: `loop38-stableclip-small-seq512-bs2ga16-t4-80step-20260702-022151`
- Base W&B: `wandb/offline-run-20260702_022154-cwmvdi1s/run-cwmvdi1s.wandb`

| Variant | Step 80 hard | Step 80 soft | Last25 hard | Last25 soft | Last40 hard | Last40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current base | 7.4572 | 4.5187 | 7.7819 | 4.2997 | 7.7818 | 4.3152 |
| random embedding | 7.4536 | 4.5357 | 7.7709 | 4.2944 | 7.7726 | 4.3125 |

Deltas for random embedding vs current base:

- Step 80 hard/soft: -0.0036 / +0.0170
- Last25 hard/soft: -0.0110 / -0.0053
- Last40 hard/soft: -0.0091 / -0.0027

Secondary diagnostics at step 80:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current base | 5.38% | 8.02% | 4138.2 | -4.6461 | 27.34% | 0.9846 |
| random embedding | 5.48% | 7.24% | 4144.3 | -4.5314 | 19.78% | 1.0038 |

Comparison with official best small baseline loop16:

- Loop16 step80 hard/soft: 7.4532 / 4.5196
- Loop16 last25 hard/soft: 7.7798 / 4.2932
- Random embedding step80 hard/soft: 7.4536 / 4.5357
- Random embedding last25 hard/soft: 7.7709 / 4.2944

## Decision

Fail loop63 as an official promotion candidate:

- The 40-step screen was genuinely positive and justified the 80-step gate.
- At 80 steps, random embedding improves current-base last25/last40 hard/soft means and slightly improves current-base step80 hard.
- It does not clearly beat the official best loop16 baseline: step80 soft is worse, last25 soft is slightly worse, and step80 hard is effectively tied but marginally worse.
- Teacher agreement is lower than the current-base run.
- Spike rate is much lower than current base and does not address loop55's under-firing relative to loop14.

Do not launch a long or full run from loop63. Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior.

The result is still useful: disabling teacher embedding copy is more paper-faithful and competitive after the early phase, but it is not strong enough to replace the current baseline under the user's small-screen promotion rule.

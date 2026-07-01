# Loop Iteration 56 - Embedding Alignment LayerNorm Screen

Date: 2026-07-02

## Hypothesis

Loop55 showed that the current code under-fires relative to loop14 under matched 3GPU geometry. A major metric difference is that loop16's identity same-dimension SpAD projector leaves the embedding-alignment loss around 3.1, while loop14's legacy same-dimension LayerNorm projector kept it near 0.88.

Prior loops already tested hidden/SFA-only LayerNorm and the full legacy projector. This screen isolates only the embedding-alignment path:

- base: current code, same-dimension embedding and hidden SpAD projectors are identities;
- candidate: same-dimension embedding/EA projector is `LayerNorm`; hidden/SFA projector remains identity.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming; base and candidate use matched seed and streaming order
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant
- Scheduler horizon equals each screen length, so compare only within each matched table.

## 20-Step Screen

- Base run: `loop56-ea-layernorm-screen20-base`
- Base W&B: `wandb/offline-run-20260702_063829-531inukl/run-531inukl.wandb`
- Candidate run: `loop56-ea-layernorm-screen20-ea-ln`
- Candidate W&B: `wandb/offline-run-20260702_063917-mgvy8gus/run-mgvy8gus.wandb`

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | EA loss | Token acc | Teacher agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 3.1847 | 4.01% | 6.07% |
| EA LayerNorm | 7.9066 | 4.4906 | 7.8717 | 4.4775 | 0.8829 | 5.97% | 6.95% |

Deltas for EA LayerNorm vs base:

- Step 20 hard/soft: -0.0388 / -0.0171
- Last10 hard/soft: -0.0283 / -0.0026
- Step 20 token accuracy / teacher agreement: +1.96 pp / +0.88 pp
- Step 20 target rank / margin: -121.8 / +0.1275
- Step 20 spike rate: -1.31 pp

The 20-step final metrics were positive, but recent soft improvement was very small, so the candidate was extended only to a 40-step confirmation.

## 40-Step Confirmation

- Base run: `loop56-ea-layernorm-screen40-base`
- Base W&B: `wandb/offline-run-20260702_064047-5ws6reau/run-5ws6reau.wandb`
- Candidate run: `loop56-ea-layernorm-screen40-ea-ln`
- Candidate W&B: `wandb/offline-run-20260702_064212-tz8h9th5/run-tz8h9th5.wandb`

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6693 | 4.3722 | 7.7956 | 4.3550 | 7.7669 | 4.3695 |
| EA LayerNorm | 7.6646 | 4.3810 | 7.7836 | 4.3722 | 7.7630 | 4.3800 |

Deltas for EA LayerNorm vs base:

- Step 40 hard/soft: -0.0047 / +0.0088
- Last10 hard/soft: -0.0120 / +0.0171
- Last20 hard/soft: -0.0038 / +0.0105
- Last25 hard/soft: -0.0044 / +0.0108
- Step 40 token accuracy / teacher agreement: +1.08 pp / +1.47 pp
- Step 40 target rank / margin: -71.4 / +0.0320
- Step 40 spike rate: -2.72 pp

Auxiliary losses at step 40:

| Variant | EA loss | Feature loss | Grad norm | Spike rate |
| --- | ---: | ---: | ---: | ---: |
| base | 3.0455 | 0.4815 | 0.3618 | 35.999% |
| EA LayerNorm | 0.8706 | 0.4623 | 0.3941 | 33.280% |

## Decision

Fail loop56. Do not launch an 80-step gate:

- The 20-step signal did not survive cleanly to 40 steps.
- Step-40 soft loss regressed, and last10/last20/last25 soft means all regressed.
- The hard-loss gain was too small to outweigh the soft-loss and recent-window regressions.
- The candidate further reduced spike rate rather than correcting loop55's under-firing against loop14.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. EA LayerNorm can reduce auxiliary embedding loss, but that does not translate into a clear hard/soft output win.

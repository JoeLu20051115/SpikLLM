# Loop Iteration 64 - Random Embedding plus Attention Output Bypass Screen

Date: 2026-07-02

## Hypothesis

Loop63 showed that paper-random embedding initialization was competitive and improved some recent windows, but its step80 spike rate was much lower than the current base. Loop52 showed that bypassing the extra SFSA pre-projection `attn_out_lif` increased spike rate and slightly improved some recent windows, although it failed the 80-step gate.

Loop64 tests the targeted combination:

- disable teacher token/position embedding copy, as in loop63;
- replace each `block.attention.attn_out_lif` with `Identity`, as in loop52;
- keep tied LM head, SFSA/SFFN otherwise, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged.

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
- Candidate changed modules: 12 `attn_out_lif` modules bypassed

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Random-only reference run: `loop63-paper-random-embedding-screen40-random-emb`
- Random-only W&B: `wandb/offline-run-20260702_071830-mzsll1pq/run-mzsll1pq.wandb`
- Candidate run: `loop64-random-emb-bypass-attnout-screen40`
- Candidate W&B: `wandb/offline-run-20260702_072552-g5jb4wh2/run-g5jb4wh2.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 |
| random embedding | 7.6476 | 4.3860 | 7.8025 | 4.3978 | 7.7789 | 4.4144 | 7.8471 | 4.5002 |
| random + bypass `attn_out_lif` | 7.6708 | 4.4010 | 7.8271 | 4.4145 | 7.7897 | 4.4190 | 7.8511 | 4.5025 |

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0002 / -0.0284
- Last10 hard/soft: -0.0078 / -0.0329
- Last20 hard/soft: -0.0034 / -0.0111
- Last25 hard/soft: +0.0361 / +0.0457

Deltas for candidate vs random-only:

- Step 40 hard/soft: +0.0232 / +0.0150
- Last10 hard/soft: +0.0246 / +0.0167
- Last20 hard/soft: +0.0108 / +0.0046
- Last25 hard/soft: +0.0040 / +0.0023

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Target rank | Target margin | Spike rate | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 4285.7 | -5.1024 | 33.16% | 0.4328 | 0.9885 |
| random embedding | 4.50% | 5.48% | 4033.3 | -5.1189 | 25.28% | 2.2442 | 1.0062 |
| random + bypass `attn_out_lif` | 4.50% | 5.48% | 4393.4 | -5.1732 | 27.54% | 12.6479 | 1.0067 |

The candidate's logged recent-window mean grad norm was extremely large because one or more steps reported overflow-scale norms. This was not observed in the random-only 40-step screen.

## Decision

Fail loop64. Do not extend to an 80-step gate:

- The combination is worse than random-only on step40 hard/soft and every recent hard/soft window.
- Step40 target rank and target margin both regress relative to random-only.
- Bypassing `attn_out_lif` increases spike rate, but the added activity does not improve output losses.
- The candidate introduces unstable gradient-norm behavior.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. Random embedding remains a useful paper-fidelity result from loop63, but combining it with `attn_out_lif` bypass is not supported.

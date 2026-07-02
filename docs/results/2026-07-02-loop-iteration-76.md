# Loop Iteration 76 - Token-Random plus No Internal Pre-LN Screen

Date: 2026-07-02

## Hypothesis

Loop72 token-random improved step80 hard loss and last25 hard/soft means, but failed promotion because step80 soft remained worse than loop16. Loop47 showed that removing internal block pre-LayerNorms produced a strong 20-step signal before failing 40-step confirmation. Since token-random changes the tied output path while preserving teacher position embeddings, loop76 tests whether the paper-style no-pre-LN block path becomes useful in that narrower setting.

Candidate:

- reinitialize only the student token embedding / tied LM head from `Normal(0, initializer_range)` after the default teacher copy;
- keep position embeddings copied from the teacher;
- replace every block `attention_norm` and `mlp_norm` with `Identity`;
- keep SFSA/SFFN otherwise, all five SpAD losses, paper weights, optimizer, scheduler, labels, teacher logits, readout, temporal input ramp, and reset behavior unchanged.

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
- Seed: Python, NumPy, and Torch set to 0 before the run
- Candidate changed modules: 12 blocks with identity `attention_norm` and `mlp_norm`

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Token-random reference run: `loop71-embedding-init-split-screen40-token-random`
- Token-random W&B: `wandb/offline-run-20260702_080220-x9pdfso2/run-x9pdfso2.wandb`
- Candidate run: `loop76-token-random-no-preln-screen40`
- Candidate W&B: `wandb/offline-run-20260702_082205-0nv9g62n/run-0nv9g62n.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| token-random | 7.6580 | 4.3853 | 7.8217 | 4.4034 | 7.7895 | 4.4061 | 7.8470 | 4.4870 | 8.5789 | 5.1308 |
| token-random + no pre-LN | 7.6637 | 4.3849 | 7.8325 | 4.3724 | 7.7935 | 4.3950 | 7.8548 | 4.4838 | 8.6191 | 5.1478 |

Deltas for candidate vs token-random:

- Step 40 hard/soft: +0.0057 / -0.0003
- Last10 hard/soft: +0.0108 / -0.0309
- Last20 hard/soft: +0.0039 / -0.0110
- Last25 hard/soft: +0.0078 / -0.0033
- Full40 hard/soft: +0.0402 / +0.0170

Deltas for candidate vs base:

- Step 40 hard/soft: -0.0073 / -0.0444
- Last10 hard/soft: -0.0024 / -0.0750
- Last20 hard/soft: +0.0004 / -0.0350
- Last25 hard/soft: +0.0398 / +0.0269
- Full40 hard/soft: +0.3633 / +0.3405

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Hidden std | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 0.4711 | 1.5517 | 0.4328 | 0.9885 |
| token-random | 4.50% | 5.48% | 13.60% | 4228.8 | -5.0888 | 26.32% | 0.4932 | 1.5849 | 5.0594 | 1.0052 |
| token-random + no pre-LN | 3.33% | 5.19% | 14.87% | 4052.9 | -4.9982 | 34.45% | 0.3712 | 1.5636 | 0.2276 | 1.0055 |

## Decision

Fail loop76. Do not extend to an 80-step gate:

- It improves soft loss vs token-random across recent windows, but hard loss regresses at step40, last10, last20, last25, and full40.
- It is not a clean win over base because last20 hard is effectively tied and last25/full40 hard/soft regress.
- Token accuracy regresses vs token-random and base.
- Teacher agreement regresses vs token-random.
- The soft-loss gain alone is insufficient under the user's hard+soft small-screen promotion rule.

Keep loop16 as the official small-gate baseline and loop14 as the best historical long-run behavior. Removing internal pre-LNs does not solve token-random's 80-step soft-loss gap without hurting hard-loss behavior.


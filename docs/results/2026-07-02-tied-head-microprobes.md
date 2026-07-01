# Tied Embedding/LM-Head Microprobes - No Loop Promotion

Date: 2026-07-02

## Purpose

Several prior notes pointed to possible gradient conflict on the teacher-initialized token embedding matrix, which is tied to the LM head. OPT-125M itself uses tied token embeddings and LM head weights, and the current student mirrors that behavior.

Three runtime-only diagnostics tested whether the tied head path should become the next loop:

1. freeze the teacher-initialized token embedding / tied LM head;
2. freeze both token embedding / tied LM head and position embedding;
3. untie the LM head from token embedding while initializing both from the teacher token matrix.

No repository code was changed for these probes.

## Setup

- Code baseline: `main` at `70ca1eb`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused for base and variants
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Freeze Probes

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| freeze token/head | 8.2367 | 4.9463 | 8.1493 | 4.9065 | +0.2802 / +0.4435 | +0.2542 / +0.4292 |
| freeze token/head + position | 8.2425 | 4.9667 | 8.1591 | 4.9301 | +0.2861 / +0.4639 | +0.2639 / +0.4529 |

Decision: fail. Freezing the teacher-initialized tied head makes hard/soft much worse, so the output head must adapt during early training.

## Untied LM-Head Probe

The untied variant clones the teacher-initialized token embedding into a separate trainable `lm_head`, leaving token embedding and output head to update independently. This is a diagnostic only; it increases parameter count and is not promoted without a strong result.

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| untied LM head | 7.9188 | 4.4976 | 7.8816 | 4.4827 | -0.0376 / -0.0052 | -0.0135 / +0.0054 |

Decision: fail the micro-screen. The final step improves slightly, but last10 soft is worse and the gain is not strong enough to justify extra parameters or an 80-step gate.

## Overall Decision

Do not create loop36 from tied-head freezing or untied-head variants. Keep loop16 as the current best baseline.

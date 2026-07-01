# Loop Iteration 42 - SFSA Projection Bias Screen

Date: 2026-07-02

## Hypothesis

The current SFSA implementation uses bias-free Q/K/V/Out projections, while OPT-family linear projections normally include bias and the paper's SFSA algorithm only specifies `Linear`, not `bias=False`. This screen tests a narrow version of that difference: keep the existing projection weights and add zero-initialized trainable biases to Q/K/V/Out, avoiding an initialization confound.

This was run as a monkeypatch only. No source code was changed for the training run.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- Candidate behavior: add trainable zero biases to SFSA Q/K/V/Out projections.
- Run: `loop42-attnbias-small-seq512-bs2ga16-t4-80step-20260702-042855`
- Local W&B: `wandb/offline-run-20260702_042857-tsaiaj13/run-tsaiaj13.wandb`
- Output: `output/loop42-attnbias-small-seq512-bs2ga16-t4-80step-20260702-042855`
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16

## Result

- Step 80 hard/soft: 7.4502 / 4.5244
- First 25 hard/soft means: 8.5399 / 5.0428
- Last 25 hard/soft means: 7.7749 / 4.2911
- Last 25 hard/soft slopes per 100 steps: -0.3290 / +0.3334
- Step 80 token accuracy: 4.01%
- Step 80 teacher top-1 agreement: 6.46%
- Step 80 target rank / margin: 4148.0 / -4.6260
- Step 80 spike rate mean: 31.25%

## Comparison

Current official metric baseline, loop16:
- Step 80 hard/soft: 7.4532 / 4.5196
- Last 25 hard/soft means: 7.7798 / 4.2932
- Token accuracy / teacher agreement: 4.79% / 6.07%
- Target rank / margin: 4162.4 / -4.6015

Loop42 improves:
- Step 80 hard loss: 7.4502 vs 7.4532
- Last 25 hard mean: 7.7749 vs 7.7798
- Last 25 soft mean: 4.2911 vs 4.2932
- Teacher agreement and target rank slightly

Loop42 regresses:
- Step 80 soft loss: 4.5244 vs 4.5196
- Step 80 token accuracy: 4.01% vs 4.79%
- Step 80 target margin: -4.6260 vs -4.6015
- Last 25 soft slope is positive

## Decision

Do not promote loop42. The attention-bias candidate is a near tie on the primary losses, but the win is too small and not supported by the output diagnostics. It does not satisfy the continuation rule and is not clear enough to justify a full or long run.

Keep loop16 as the official small-gate metric baseline. Do not add SFSA projection biases to `main` from this result alone.

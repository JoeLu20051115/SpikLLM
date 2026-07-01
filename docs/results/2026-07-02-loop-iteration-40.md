# Loop Iteration 40 - Legacy Same-Dimension SpAD Projector Screen

Date: 2026-07-02

## Hypothesis

Loop14's best long-run behavior used the pre-loop16 same-dimension SpAD projector, which applied a trainable LayerNorm even when student and teacher dimensions matched. Loop16 changed same-dimension projectors to exact identities and slightly improved the official 80-step small gate. Since loop39 did not reproduce loop14's long-run trend, this loop screens whether restoring the legacy same-dimension projector is worth a new code loop.

This was run as a monkeypatch only. No source code was changed for the training run.

## Setup

- Base code: loop38 stable gradient clipping.
- Candidate behavior: same-dimension `SpADProjector` uses `Identity -> LayerNorm`.
- Run: `loop40-legacyprojector-small-seq512-bs2ga16-t4-80step-20260702-041328`
- Local W&B: `wandb/offline-run-20260702_041330-6ski7rmq/run-6ski7rmq.wandb`
- Output: `output/loop40-legacyprojector-small-seq512-bs2ga16-t4-80step-20260702-041328`
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16

## Result

- Step 80 hard/soft: 7.4659 / 4.5350
- Last 25 hard/soft means: 7.7785 / 4.2940
- Last 25 hard/soft slopes per 100 steps: -0.2803 / +0.3059
- Step 80 token accuracy: 4.50%
- Step 80 teacher top-1 agreement: 5.48%
- Step 80 target rank / margin: 4179.2 / -4.6661
- Step 80 embedding loss: 0.8439
- Step 80 feature loss: 0.8098

## Comparison

Current official metric baseline, loop16:
- Step 80 hard/soft: 7.4532 / 4.5196
- Last 25 hard/soft means: 7.7798 / 4.2932
- Token accuracy / teacher agreement: 4.79% / 6.07%
- Target rank / margin: 4162.4 / -4.6015

Loop38 stable clip with identity projector:
- Step 80 hard/soft: 7.4572 / 4.5187
- Last 25 hard/soft means: 7.7819 / 4.2997
- Token accuracy / teacher agreement: 5.38% / 8.02%
- Target rank / margin: 4138.2 / -4.6461

The legacy projector improves embedding loss by construction, but primary output losses and output diagnostics do not improve. It is worse than loop16 on step-80 hard/soft, token accuracy, teacher agreement, target rank, and target margin.

## Decision

Fail loop40. Do not restore the legacy same-dimension SpAD projector.

Keep the loop16 identity projector behavior and loop38 stable clipping. The next candidate should focus elsewhere; projector LayerNorm alignment can lower auxiliary losses without improving hard/soft output behavior.

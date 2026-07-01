# Loop Iteration 43 - FP32 SpAD Loss Screen

Date: 2026-07-02

## Hypothesis

The model forward pass runs in bf16. If the five SpAD distillation losses are accumulated from bf16 tensors, small numerical differences in KL/feature/attention terms might blunt the early signal. This screen keeps the forward pass and optimizer unchanged, but converts floating student and teacher output tensors to fp32 immediately before `compute_multilevel_distillation`.

This was run as a monkeypatch only. No source code was changed for the training run.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- Candidate behavior: compute the SpAD loss from fp32 copies of student/teacher outputs.
- Run: `loop43-fp32loss-small-seq512-bs2ga16-t4-80step-20260702-043351`
- Local W&B: `wandb/offline-run-20260702_043353-z46r252o/run-z46r252o.wandb`
- Output: `output/loop43-fp32loss-small-seq512-bs2ga16-t4-80step-20260702-043351`
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16 forward, fp32 loss inputs

## Result

- Step 80 hard/soft: 7.4576 / 4.5231
- First 25 hard/soft means: 8.5420 / 5.0380
- Last 25 hard/soft means: 7.7751 / 4.2902
- Last 25 hard/soft slopes per 100 steps: -0.2722 / +0.2856
- Step 80 token accuracy: 4.50%
- Step 80 teacher top-1 agreement: 5.58%
- Step 80 target rank / margin: 4185.4 / -4.6627
- Step 80 spike rate mean: 27.10%

## Comparison

Current official metric baseline, loop16:
- Step 80 hard/soft: 7.4532 / 4.5196
- Last 25 hard/soft means: 7.7798 / 4.2932
- Token accuracy / teacher agreement: 4.79% / 6.07%
- Target rank / margin: 4162.4 / -4.6015

Loop43 improves:
- Last 25 hard mean: 7.7751 vs 7.7798
- Last 25 soft mean: 4.2902 vs 4.2932

Loop43 regresses:
- Step 80 hard loss: 7.4576 vs 7.4532
- Step 80 soft loss: 4.5231 vs 4.5196
- Step 80 token accuracy: 4.50% vs 4.79%
- Step 80 teacher agreement: 5.58% vs 6.07%
- Step 80 target rank / margin: 4185.4 / -4.6627 vs 4162.4 / -4.6015
- Last 25 soft slope is positive

## Decision

Do not promote loop43. FP32 loss inputs do not provide a clear small-gate win over loop16 and do not satisfy the continuation rule. Keep loop16 as the official small-gate metric baseline, and do not add the fp32-loss wrapper to source from this result.

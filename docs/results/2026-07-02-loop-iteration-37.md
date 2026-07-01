# Loop Iteration 37 - Seq1024 Geometry Diagnostic

Date: 2026-07-02

## Hypothesis

Loop36 showed that larger paper-batch geometry improves some secondary metrics but does not clearly beat loop16 at `seq512`. Loop14's best long run used `seq1024`, so this diagnostic tests whether increasing sequence length is the missing geometry factor.

No model, loss, optimizer, or SpAD/SFSA code was changed for this loop.

## Attempts

### Seq1024 bs8 GA16

- Run: `loop37-geometry-loop16-seq1024-bs8ga16-t4-80step-20260702-014748`
- GPU: 1x H200, GPU0
- Sequence length: 1024
- Time steps: 4
- Batch size: 8
- Gradient accumulation: 16
- Max optimizer steps: 80

Result: failed before completing the gate with CUDA OOM.

Failure location:
- `bispikclm/models/bispik_model.py`, while stacking per-time-step attention tensors.
- CUDA tried to allocate 1.50 GiB with only 901 MiB free.
- The process had 138.91 GiB in use on the 139.80 GiB visible GPU.

Decision: `seq1024 bs8` is too close to the memory limit when SpAD attention histories are retained.

### Seq1024 bs4 GA32

This fallback keeps the same effective tokens per optimizer step as `seq1024 bs8 GA16`, but halves the microbatch.

- Run: `loop37-geometry-loop16-seq1024-bs4ga32-t4-80step-20260702-014836`
- Local W&B: `wandb/offline-run-20260702_014838-er5z5vax/run-er5z5vax.wandb`
- Output: `output/loop37-geometry-loop16-seq1024-bs4ga32-t4-80step-20260702-014836`
- GPU: 1x H200, GPU0
- Sequence length: 1024
- Time steps: 4
- Batch size: 4
- Gradient accumulation: 32
- Effective tokens per optimizer step: 131,072
- Max optimizer steps: 80
- Precision: bf16

Result:
- First non-finite `train/grad_norm`: step 23
- First NaN hard/soft/total loss: step 56
- Last finite step: 55
- Step 55 hard/soft: 8.6673 / 4.9700
- Last 25 finite hard/soft means through step 55: 8.6857 / 4.9910
- Final step hard/soft: NaN / NaN
- Final spike rate: 0.0
- Final target rank mean: 1.0

## Decision

Fail loop37. The `seq1024 bs4 GA32` fallback fits in memory, but the native clipping path becomes numerically unstable and training collapses to NaNs before the 80-step gate finishes.

Do not launch a long or full run from this geometry. Keep loop16 as the current metric baseline.

## Next Step

Investigate the non-finite gradient norm before changing model structure. The failure pattern suggests a numerical clipping issue:

- `grad_norm` becomes non-finite before the losses become NaN.
- The loss collapse follows the first non-finite gradient norm by roughly 30 steps.
- The issue appears under larger effective-batch/sequence geometry where accumulated gradients are larger.

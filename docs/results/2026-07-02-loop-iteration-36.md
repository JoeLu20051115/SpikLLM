# Loop Iteration 36 - Paper Batch Geometry Diagnostic

Date: 2026-07-02

## Hypothesis

Recent code-level candidates did not clearly beat loop16 on the official small gate. The remaining discrepancy with the paper is training geometry: the official gate uses `seq512`, `bs2`, `GA16` on 1x H200, while the paper reports batch size 16 and gradient accumulation 16. Loop14 also reached substantially lower losses under a much larger 3-GPU `seq1024` geometry.

This diagnostic keeps the loop16 code unchanged and tests whether a more paper-like batch geometry improves early hard/soft losses before spending more time on structural changes.

## Code State

- Code baseline: loop16 identity same-dimension SpAD projector.
- Latest repository state before the run: docs-only changes after loop16; no model or loss code changes.
- Distillation weights, temperature, optimizer, schedule, gradient clip, SFSA/SFFN, labels, teacher logits, and readout are unchanged.

## 20-Step Geometry Check

These short runs used `max_steps=20`, so their scheduler differs from the 80-step gate. They are only a quick capacity/signal check.

Common setup:
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0
- W&B mode: offline

Official small geometry:
- Run: `geomdiag-loop16-bs2ga16-seq512-t4-20step-20260702-012752`
- Local W&B: `wandb/offline-run-20260702_012755-zn7abihw/run-zn7abihw.wandb`
- Step 20 hard/soft: 7.9564 / 4.5028
- Last 10 hard/soft means: 7.8951 / 4.4773
- Last 10 slopes per 100 steps: hard +0.2620, soft +0.6962
- Step 20 target rank / margin: 5179.2 / -5.1143

Paper-batch geometry:
- Run: `geomdiag-loop16-bs16ga16-seq512-t4-20step-20260702-012856`
- Local W&B: `wandb/offline-run-20260702_012858-2vcie8x5/run-2vcie8x5.wandb`
- Step 20 hard/soft: 7.8386 / 4.3263
- Last 10 hard/soft means: 7.7993 / 4.3642
- Last 10 slopes per 100 steps: hard -0.4502, soft -0.4617
- Step 20 target rank / margin: 4283.1 / -5.0468

The 20-step check confirms that `bs16`, `GA16`, `seq512`, `T4` fits on one H200 and gives a better early signal than `bs2`, `GA16`. Because the scheduler differs at `max_steps=20`, this is not sufficient for promotion.

## 80-Step Paper-Batch Gate

Setup:
- Run: `loop36-geometry-loop16-bs16ga16-seq512-t4-80step-20260702-013301`
- Local W&B: `wandb/offline-run-20260702_013303-pbu5gurp/run-pbu5gurp.wandb`
- Output: `output/loop36-geometry-loop16-bs16ga16-seq512-t4-80step-20260702-013301`
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 16
- Gradient accumulation: 16
- Effective tokens per optimizer step: 131,072
- Max optimizer steps: 80
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0
- Peak allocated memory logged by PyTorch: 96.08 GB
- Tokens seen at step 80: 10,485,760

Result:
- Step 80 hard/soft: 7.7240 / 4.3851
- First 25 hard/soft means: 8.4713 / 4.9497
- Last 25 hard/soft means: 7.7169 / 4.3393
- Last 40 hard/soft means: 7.7159 / 4.3375
- Last 25 slopes per 100 steps: hard -0.1175, soft -0.1857
- Last 40 slopes per 100 steps: hard -0.0546, soft -0.0013
- Step 80 token accuracy: 3.78%
- Step 80 teacher top-1 agreement: 6.19%
- Step 80 target rank mean: 3283.8
- Step 80 target margin mean: -4.5628

## Comparison With Current Best Loop16 Gate

Current best baseline remains loop16:
- Code state: loop16 identity-projector candidate, commit `9c0d64f`
- Refresh run: `loop16-refresh-small-seq512-bs2-ga16-1xh200-20260702-001801`
- W&B: `1f8plmtv`
- Step 80 hard/soft: 7.4532 / 4.5196
- Last 25 hard/soft means: 7.7798 / 4.2932
- Step 80 token accuracy: 4.79%
- Step 80 teacher top-1 agreement: 6.07%
- Step 80 target rank mean: 4162.4
- Step 80 target margin mean: -4.6015

The paper-batch run improves:
- Step 80 soft loss: 4.3851 vs 4.5196
- Last 25 hard mean: 7.7169 vs 7.7798
- Step 80 target rank: 3283.8 vs 4162.4
- Step 80 target margin: -4.5628 vs -4.6015
- Teacher agreement slightly: 6.19% vs 6.07%

It regresses:
- Step 80 hard loss: 7.7240 vs 7.4532
- Last 25 soft mean: 4.3393 vs 4.2932
- Step 80 token accuracy: 3.78% vs 4.79%

## Decision

Do not promote loop36 to full-training continuation. The larger paper-batch geometry is informative and improves several secondary/output-distribution metrics, but it is not a clear win on the primary hard/soft gate and does not meet the continuation rule:

- Hard and soft losses are not both below 5.
- The recent last-25 and last-40 slopes are too weak to count as a clear continuing downward trend.

Keep loop16 as the current best code baseline. Do not launch full training from this geometry result alone.

## Next Step

The geometry hypothesis is still plausible because loop14 used `seq1024` and reached much lower losses after a longer run. The next low-risk diagnostic should test sequence-length geometry, not another model patch:

- Use loop16 code.
- Run `seq1024`, `T4`, with a microbatch that fits one H200, likely `bs8`, `GA16`.
- Compare against the current loop16 baseline as a training-geometry diagnostic, not as a code-candidate promotion unless hard/soft losses clearly improve and recent slopes support extension.

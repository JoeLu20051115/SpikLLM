# Loop Iteration 38 - Stable Gradient Clipping

Date: 2026-07-02

## Hypothesis

Loop37 failed with `grad_norm=inf` before losses became NaN. Loop36 also logged non-finite gradient norms under `seq512 bs16 GA16`, even though the run completed. The paper uses gradient clipping with threshold 0.7; this loop keeps that threshold unchanged and tests whether the implementation should compute the norm more stably before applying the same clip.

## Root-Cause Diagnostic

Native clipping was instrumented on the loop36 geometry (`seq512`, `bs16`, `GA16`, `T4`, `max_steps=80`) and stopped after the 25th clip call.

At clip call 24:
- Number of gradient tensors: 149
- Non-finite gradient values: 0
- Maximum finite gradient magnitude: `4.7848e20`
- Safe float64 total norm: `1.7090e21`
- Native `torch.nn.utils.clip_grad_norm_` returned: `inf`

This proves the failure is not caused by NaN/Inf gradient values entering the clipper. The gradients are finite but extremely large; native norm computation overflows to `inf`, which loses the finite direction information. A stable clipper can still apply the paper's threshold 0.7 by scaling the finite gradient vector.

## Code Change

- Add `clip_grad_norm_stable(parameters, max_norm, norm_type=2.0)`.
- Compute gradient norms tensor-by-tensor in float64.
- Scale each gradient on its own device and dtype.
- Replace native clipping in the real SpAD training loop and dummy-batch diagnostic.
- Keep optimizer, LR schedule, loss weights, temperature, and `gradient_clip=0.7` unchanged.

## Tests

RED:
- `tests/smoke/test_paper_faithful_pipeline.py::test_stable_gradient_clip_preserves_huge_finite_gradient_direction`
- Failed before implementation because `clip_grad_norm_stable` did not exist.

GREEN:
- Targeted test passed.
- Full smoke passed: `44 passed, 58 warnings`.

Post-fix diagnostic:
- `graddiag-stableclip-seq512-bs16ga16-t4-max80-stop25-20260702-021753`
- Clip calls 20-25 all returned finite norms.

## 80-Step Gates

### Official Small Gate

- Run: `loop38-stableclip-small-seq512-bs2ga16-t4-80step-20260702-022151`
- Local W&B: `wandb/offline-run-20260702_022154-cwmvdi1s/run-cwmvdi1s.wandb`
- Geometry: `seq512`, `T4`, `bs2`, `GA16`, 1x H200, bf16, 80 steps

Result:
- Step 80 hard/soft: 7.4572 / 4.5187
- Last 25 hard/soft means: 7.7819 / 4.2997
- Last 25 hard/soft slopes per 100 steps: -0.2835 / +0.3007
- Step 80 token accuracy: 5.38%
- Step 80 teacher agreement: 8.02%
- Step 80 target rank / margin: 4138.2 / -4.6461
- First non-finite grad norm: none

Comparison with loop16 current best:
- Loop16 step 80 hard/soft: 7.4532 / 4.5196
- Loop16 last 25 hard/soft means: 7.7798 / 4.2932
- Loop16 token accuracy / teacher agreement: 4.79% / 6.07%
- Loop16 target rank / margin: 4162.4 / -4.6015

Official-gate decision: not a clear win. Stable clipping improves token accuracy, teacher agreement, and target rank, but primary hard loss, last-25 hard, last-25 soft, and target margin do not beat loop16.

### Paper-Batch Seq512 Gate

- Run: `loop38-stableclip-paperbatch-seq512-bs16ga16-t4-80step-20260702-022520`
- Local W&B: `wandb/offline-run-20260702_022522-n72i083f/run-n72i083f.wandb`
- Geometry: `seq512`, `T4`, `bs16`, `GA16`, 1x H200, bf16, 80 steps

Result:
- Step 80 hard/soft: 7.7256 / 4.3874
- Last 25 hard/soft means: 7.7184 / 4.3407
- Last 25 hard/soft slopes per 100 steps: -0.1297 / -0.1804
- Step 80 token accuracy: 3.72%
- Step 80 teacher agreement: 6.31%
- Step 80 target rank / margin: 3310.4 / -4.5717
- First non-finite grad norm: none

Comparison with loop36 native clipping on the same geometry:
- Loop36 step 80 hard/soft: 7.7240 / 4.3851
- Loop36 last 25 hard/soft means: 7.7169 / 4.3393
- Loop36 first non-finite grad norm: step 24

Paper-batch decision: stable clipping fixes non-finite grad-norm logging but does not improve hard/soft losses on this 80-step gate.

### Seq1024 Gate

- Run: `loop38-stableclip-seq1024-bs4ga32-t4-80step-20260702-023732`
- Local W&B: `wandb/offline-run-20260702_023734-tbktr3k4/run-tbktr3k4.wandb`
- Geometry: `seq1024`, `T4`, `bs4`, `GA32`, 1x H200, bf16, 80 steps

Result:
- Step 80 hard/soft: 7.7228 / 4.4668
- Last 25 hard/soft means: 7.7201 / 4.4160
- Last 25 hard/soft slopes per 100 steps: -0.1715 / -0.1761
- Step 80 token accuracy: 3.76%
- Step 80 teacher agreement: 4.79%
- Step 80 target rank / margin: 3291.2 / -4.5468
- First NaN loss: none
- First non-finite grad norm: none

Comparison with loop37 native clipping on the same geometry:
- Loop37 first non-finite grad norm: step 23
- Loop37 first NaN loss: step 56
- Loop37 final hard/soft: NaN / NaN

Seq1024 decision: stable clipping fixes the NaN collapse, but the finite 80-step losses still do not beat the current loop16 small-gate baseline and do not meet the continuation criterion.

## Decision

Keep the stable gradient clipping code because it fixes a proven numerical bug while preserving the paper's gradient clipping threshold. Do not promote loop38 as the new training baseline by hard/soft metrics:

- Hard and soft losses are not both below 5.
- The official small gate does not clearly beat loop16 primary metrics.
- The paper-batch and seq1024 gates remain too flat in the recent window.

Current metric baseline remains loop16. Future candidates should run on top of stable clipping to avoid reintroducing the non-finite clipping path, but they still must beat loop16/loop38 gate metrics before any full run.

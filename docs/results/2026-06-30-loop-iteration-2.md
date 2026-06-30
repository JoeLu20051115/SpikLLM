# 2026-06-30 BiSpikCLM Loop Iteration 2

## Hypothesis

Loop 1 failed because the tied embedding/lm-head matrix used PyTorch's default embedding initialization (`std ~= 1.0`), while OPT uses transformer-scale tied embeddings (`std ~= 0.055`). This made the student overconfident and caused hard/soft loss rebound through logit-scale expansion.

## Change

- Added `initializer_range = 0.02`.
- Initialized token and position embeddings at transformer scale.
- Preserved tied `lm_head`.
- Added input embedding scaling before spiking blocks so the SNN still receives a usable current despite transformer-scale embeddings.
- Set `padding_idx` on the token embedding.
- Updated the fixed-batch SpAD smoke to use the same projector path and paper SpAD weights as training.

No SpAD loss weights, temperature, teacher logits, labels, or data source were changed.

## Review

Reviewer found no Critical, Important, or Minor training-path issues for `32fdb21..79dd658`.

## Probe

- Run name: `loop2-opt125m-seq1024-bs8-ga32-3xh200-20260630-232210`
- W&B URL: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/qkgh5d6e
- Local W&B file: `wandb/run-20260630_232215-qkgh5d6e/run-qkgh5d6e.wandb`
- Output dir: `output/loop2-opt125m-seq1024-bs8-ga32-3xh200-20260630-232210`

## Decision

Stopped at step 125. The run was much healthier than loop 1 and the change is worth keeping, but it did not pass the C acceptance rule because hard loss plateaued above 5.

Analyzer output at stop:

```json
{
  "decision": "fail",
  "hard_early_mean": 9.190110751241445,
  "hard_last": 7.635702595114708,
  "hard_recent_mean": 8.429480994194746,
  "hard_slope_per_100": -2.7759689710768285,
  "last_step": 125,
  "reason": "insufficient_hard_soft_descent",
  "soft_early_mean": 5.667314384281635,
  "soft_last": 4.459174856543541,
  "soft_recent_mean": 5.058952824175358,
  "soft_slope_per_100": -2.3301454332902773
}
```

Diagnostic comparison:

- Loop 1 step 47: hard `12.5487`, soft `9.5958`, logit std `2.5407`, logit abs max `24.6250`.
- Loop 2 step 54: hard `8.8436`, soft `5.3439`, logit std `0.8217`, logit abs max `4.2500`.
- Loop 2 step 76: hard `7.9882`, soft `4.6736`, logit std `1.0773`, logit abs max `6.0000`.
- Loop 2 step 125: hard `7.6357`, soft `4.4592`, logit std `1.4834`, logit abs max `7.2500`.

## Next Hypothesis

After stabilizing logit scale, soft distillation reaches the target region but hard CE plateaus. The next iteration should focus on the hard-label learning signal and the student output representation, not on SpAD weights.

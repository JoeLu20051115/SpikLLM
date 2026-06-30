# 2026-06-30 BiSpikCLM Loop Iteration 1

## Hypothesis

The previous GA=1 run failed mainly because it was not paper-constrained and had too much output-loss noise. This iteration tested a closer paper-constrained probe shape without changing SpAD weights.

## Change

Added local W&B probe analysis and output diagnostics only. No loss weights, temperature, data source, labels, teacher logits, or hard/soft loss semantics were changed.

## Probe

- Run name: `loop1-opt125m-seq1024-bs8-ga32-3xh200-20260630-224411`
- W&B URL: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/yd6ew7q0
- Local W&B file: `wandb/run-20260630_224416-yd6ew7q0/run-yd6ew7q0.wandb`
- Output dir: `output/loop1-opt125m-seq1024-bs8-ga32-3xh200-20260630-224411`
- Command: `torchrun --nproc_per_node=3 --master_port=29641 -m bispikclm.train.train_spad --config configs/bispikclm_opt125m_spad.toml --output-dir output/loop1-opt125m-seq1024-bs8-ga32-3xh200-20260630-224411 --sequence-length 1024 --precision bf16 --batch-size 8 --gradient-accumulation-steps 32 --wandb --wandb-project bispikclm --wandb-run-name loop1-opt125m-seq1024-bs8-ga32-3xh200-20260630-224411 --train`

## Decision

Stopped early at step 47. The run began with strong hard/soft descent, but then reversed into high-level oscillation before reaching the acceptance threshold.

Analyzer output at stop:

```json
{
  "decision": "fail",
  "hard_early_mean": 13.874263614416122,
  "hard_last": 12.548720806837082,
  "hard_recent_mean": 13.874263614416122,
  "hard_slope_per_100": -19.3252559951149,
  "last_step": 47,
  "reason": "insufficient_hard_soft_descent",
  "soft_early_mean": 10.538823783397675,
  "soft_last": 9.595775336027145,
  "soft_recent_mean": 10.538823783397675,
  "soft_slope_per_100": -19.52168164304376
}
```

Local diagnostic pattern:

- Step 26: hard `11.0211`, soft `7.2924`, logit std `0.8280`, logit abs max `8.3125`.
- Step 47: hard `12.5487`, soft `9.5958`, logit std `2.5407`, logit abs max `24.6250`.

## Next Hypothesis

The student output path lacks the final decoder normalization present in OPT. Without an LM-head pre-normalization, LIF hidden-state scale drift is amplified by the tied embedding head, causing hard and soft losses to rebound after the initial descent.

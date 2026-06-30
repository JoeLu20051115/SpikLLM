# BiSpikCLM Loop Iteration 6

Runs:

- OOM: `loop6-opt125m-t4-thr07-seq1024-bs8-ga32-3xh200-20260701-020755`
- Probe: `loop6b-opt125m-t4-thr07-seq1024-bs4-ga64-3xh200-20260701-020907`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/50ophvtn

## Setup

- Base code: `3af76a3 config: use four spiking time steps`.
- Hardware: 3x H200.
- Sequence length: 1024.
- OOM attempt: batch per rank 8, gradient accumulation 32.
- Valid probe: batch per rank 4, gradient accumulation 64.
- Effective tokens per optimizer step: 786,432.
- Time steps: 4.
- Spike threshold: 0.7.
- Optimizer/schedule: Adam, cosine decay, warmup ratio 0.2, gradient clip 0.7.
- SpAD weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`, `temperature=2.0`.

## Result

The batch 8 run OOMed during stacked attention-state materialization. The batch 4 / GA64 run fit in memory, but hard-label learning plateaued again.

- Last step hard loss: 7.7828.
- Last step soft loss: 4.5924.
- First 20 step hard mean: 10.5158.
- Middle 20 step hard mean: 7.8948.
- Last 20 step hard mean: 7.7225.
- Last 50 step hard mean: 7.7281.
- Last 50 step soft mean: 4.4391.
- Last 50 step token accuracy mean: 0.0373.
- Recent hard slope per 100 steps: -1.8927.
- Recent soft slope per 100 steps: -2.1359.

This did not meet the probe acceptance criterion for continuing full training.

## Root Cause Update

Increasing temporal resolution from 2 to 4 did not break the recurring hard-loss plateau. Across loop 3 through loop 6, the model repeatedly saturates around `hard_loss ~= 7.7`, `token_accuracy ~= 0.04`, and `logit_std ~= 1.5`.

The next iteration reverts to `time_steps=2` for throughput and adds an explicit trainable readout gain to address the output-margin bottleneck without changing SpAD loss weights.

# BiSpikCLM Loop Iteration 5

Run: `loop5-opt125m-thr07-seq1024-bs8-ga32-3xh200-20260701-013016`
W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/ba6q9dui

## Setup

- Base code: `92b93b5 fix: use paper-supported spiking threshold`.
- Hardware: 3x H200.
- Sequence length: 1024.
- Batch per rank: 8.
- Gradient accumulation: 32.
- Effective tokens per optimizer step: 786,432.
- Time steps: 2.
- Spike threshold: 0.7.
- Optimizer/schedule: Adam, cosine decay, warmup ratio 0.2, gradient clip 0.7.
- SpAD weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`, `temperature=2.0`.

## Result

Stopped at optimizer step 132 because hard-label learning plateaued again.

- Last step hard loss: 7.7723.
- Last step soft loss: 4.3051.
- First 20 step hard mean: 10.6904.
- Middle 20 step hard mean: 7.7972.
- Last 20 step hard mean: 7.7110.
- Last 50 step hard mean: 7.7208.
- Last 50 step soft mean: 4.4407.
- Last 50 step token accuracy mean: 0.0377.
- Recent hard slope per 100 steps: -1.0997.
- Recent soft slope per 100 steps: -1.2229.

The lower threshold improved the early phase but did not solve the hard-loss plateau. This did not meet the probe acceptance criterion for continuing full training.

## Root Cause Update

The repeated plateau across loop 3, loop 4, and loop 5 suggests the remaining bottleneck is not a single attention-loss scaling issue or threshold propagation bug. With only `time_steps=2`, the final temporal average has very coarse spike-rate states, and the output margin again saturates around `logit_std ~= 1.5`.

The next iteration will use the paper-supported `time_steps=4` setting while keeping SpAD weights unchanged.

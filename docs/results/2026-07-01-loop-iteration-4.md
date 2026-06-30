# BiSpikCLM Loop Iteration 4

Run: `loop4b-opt125m-seq1024-bs8-ga32-3xh200-20260701-004326`
W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/99fulwi3

## Setup

- Base code: `164c014 fix: align spad attention distributions`.
- Hardware: 3x H200.
- Sequence length: 1024.
- Batch per rank: 8.
- Gradient accumulation: 32.
- Effective tokens per optimizer step: 786,432.
- Optimizer/schedule: Adam, cosine decay, warmup ratio 0.2, gradient clip 0.7.
- SpAD weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`, `temperature=2.0`.

## Result

Stopped at optimizer step 156 because hard-label learning plateaued.

- Last step hard loss: 7.6653.
- Last step soft loss: 4.5449.
- First 20 step hard mean: 10.8205.
- Middle 20 step hard mean: 7.7586.
- Last 20 step hard mean: 7.7202.
- Last 50 step hard mean: 7.7180.
- Last 50 step soft mean: 4.4496.
- Last 50 step token accuracy mean: 0.0387.
- Recent hard slope per 100 steps: -0.1404.
- Recent soft slope per 100 steps: -0.2001.

The attention branch no longer collapsed to a near-zero loss, but hard loss and token accuracy reproduced the loop 3 plateau. This did not meet the probe acceptance criterion for continuing full training.

## Root Cause Update

The remaining failure is not explained by loss weights, optimizer, warmup, or causal shifting. The current evidence points to insufficient spike activation/output separability: logit standard deviation plateaued around 1.5, hard loss stayed around 7.7, and token accuracy stayed near 4%.

The next iteration will align the firing threshold path with the paper-supported low-threshold setting and ensure the MLP LIF uses the configured threshold.

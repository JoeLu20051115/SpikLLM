# BiSpikCLM Loop Iteration 3

Run: `loop3-opt125m-seq1024-bs8-ga32-3xh200-20260701-000132`
W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/gwwcr54s

## Setup

- Student/teacher: BiSpikCLM OPT-125M shape distilled from `facebook/opt-125m`.
- Hardware: 3x H200.
- Sequence length: 1024.
- Batch per rank: 8.
- Gradient accumulation: 32.
- Effective tokens per optimizer step: 786,432.
- Optimizer/schedule: Adam, cosine decay, warmup ratio 0.2, gradient clip 0.7.
- SpAD weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`, `temperature=2.0`.

## Result

Stopped at optimizer step 133 because hard-label learning plateaued.

- Last step hard loss: 7.7671.
- Last step soft loss: 4.3693.
- First 20 step hard mean: 10.6996.
- Middle 20 step hard mean: 7.8861.
- Last 20 step hard mean: 7.7102.
- Last 20 step soft mean: 4.4450.
- Last 20 step token accuracy mean: 0.0394.
- Recent hard slope per 100 steps: -1.2734.
- Recent soft slope per 100 steps: -1.3871.

Soft loss reached the target band, but hard loss stayed around 7.7 and token accuracy remained near random. This did not meet the probe acceptance criterion for continuing full training.

## Root Cause

The attention alignment branch was under-penalizing collapsed student attention. With `seq=1024`, teacher softmax attention probabilities are usually far below the spike threshold. Feeding those probabilities directly into the T=2 rate encoder produced mostly zero teacher attention rates. The raw elementwise MSE was also diluted by the full attention matrix size, so zero student attention could receive a near-zero attention loss.

The next iteration changes only the attention-map alignment calculation. It keeps the SpAD weights, temperature, optimizer, schedule, and batch geometry unchanged.

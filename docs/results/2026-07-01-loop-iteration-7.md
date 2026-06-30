# Loop Iteration 7 - Readout Scale 2.0 Probe

Date: 2026-07-01

Run:
- Name: `loop7-opt125m-readout2-thr07-seq1024-bs8-ga32-3xh200-20260701-030858`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/u99uq596
- Local file: `wandb/run-20260701_030903-u99uq596/run-u99uq596.wandb`
- Output: `output/loop7-opt125m-readout2-thr07-seq1024-bs8-ga32-3xh200-20260701-030858`

Configuration:
- Teacher: `facebook/opt-125m`
- GPUs: 3x H200
- Sequence length: 1024
- Per-GPU batch size: 8
- Gradient accumulation: 32
- Time steps: 2
- Spike threshold: 0.7
- Readout scale: trainable, initialized at 2.0
- SpAD weights unchanged: EA 0.2, SAA 0.1, SFA 0.1, STA 0.3, HTA 0.3
- Temperature: 2.0
- Optimizer/schedule unchanged: Adam, learning rate 5e-4, cosine schedule, warmup ratio 0.2, gradient clip 0.7

Result:
- Stopped at step 899 because hard loss plateaued.
- Last hard loss: 7.76398
- Last soft loss: 4.39790
- Last 100-step hard mean: 7.72188
- Last 100-step soft mean: 4.43709
- Hard slope per 100 steps: -0.01276
- Soft slope per 100 steps: +0.00924
- Token accuracy remained around 3-5%.
- Logit std increased to about 1.52, but that did not translate into next-token accuracy.

Decision:
- Fail. Loop7 had the best early descent among the first seven loops, reaching the hard-loss plateau faster than earlier runs, but it did not continue toward the required hard/soft <5 target.
- This is not acceptable for full training.

Best-so-far assessment across loops 1-7:
- Best early behavior: loop7, because hard loss reached the 7.7 band fastest.
- Best stable non-wasteful behavior before loop7: loop4/loop5/loop6 are effectively tied, all plateauing near hard 7.72 and soft 4.4-4.6.
- No loop among the first seven passes the acceptance criterion.

Next debugging direction:
- The repeated hard-loss plateau near 7.7 with token accuracy around 4% suggests a representation or readout bottleneck, not a loss-weight issue.
- Keep the paper loss weights fixed.
- Add low-frequency diagnostics for spike rates, hidden variance, target-token rank, teacher agreement, target margin, and readout scale before making the next code change.

# Loop Iteration 14 - T4 Teacher Embedding Init + Unit Readout

Date: 2026-07-01

Run:
- Name: `loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/b2iflr1z
- Local file: `wandb/run-20260701_110436-b2iflr1z/run-b2iflr1z.wandb`
- Output: `output/loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431`

Configuration:
- Teacher: `facebook/opt-125m`
- GPUs: 3x H200
- Sequence length: 1024
- Per-GPU batch size: 4
- Gradient accumulation: 64
- Effective tokens per optimizer step: 786,432
- Time steps: 4
- Spike threshold: 0.7
- Readout scale: trainable, initialized at 1.0
- Student token and position embeddings initialized from the OPT teacher.
- SpAD weights unchanged: EA 0.2, SAA 0.1, SFA 0.1, STA 0.3, HTA 0.3
- Temperature: 2.0
- Optimizer/schedule unchanged: Adam, learning rate 5e-4, cosine schedule, warmup ratio 0.2, gradient clip 0.7

Interim improvement:
- Step 620 temporarily satisfied the early strong-downward-trend continuation rule.
- Last hard loss: 6.1809
- Last soft loss: 2.8587
- Last 100-step hard mean: 6.2019
- Last 100-step soft mean: 2.8004
- Recent hard slope per 100 steps: -0.2962
- Recent soft slope per 100 steps: -0.2853
- Token accuracy in the latest logged step: 13.44%
- Teacher top-1 agreement in the latest logged step: 21.24%
- Readout scale in the latest logged step: 0.9282
- Step 624 remained in `extend`: hard 6.2884, soft 2.8512, 100-step hard slope -0.2321, 100-step soft slope -0.2020.

Final result:
- Stopped at step 663 because the strong-downward-trend condition did not persist.
- Last hard loss: 6.1154
- Last soft loss: 2.6974
- Last 100-step hard mean: 6.1369
- Last 100-step soft mean: 2.7290
- Recent hard slope per 100 steps: -0.0853
- Recent soft slope per 100 steps: -0.0739
- Last 250-step hard mean: 6.2605
- Last 250-step soft mean: 2.8480
- 250-step hard slope per 100 steps: -0.1446
- 250-step soft slope per 100 steps: -0.1361
- Token accuracy in the latest logged step: 15.37%
- Teacher top-1 agreement in the latest logged step: 24.56%
- Target rank mean in the latest logged step: 1041.62
- Readout scale in the latest logged step: 0.9215
- Tokens seen at step 663: 521,404,416.

Comparison with loop13:
- Loop13 stopped at step 504 with hard 6.5502, soft 3.0326, hard slope -0.0765, and soft slope -0.0258.
- Loop14 has lower hard and soft losses, a stronger recent descent, and previously reached 25.49% teacher top-1 agreement at step 596.

Decision:
- Fail for full-training continuation. Loop14 is the best run so far, but it did not keep hard/soft losses below 5 or maintain a strong enough downward trend.
- Stop the run and keep the code baseline for the next hypothesis, because it materially improved loop13's hard/soft losses and teacher agreement.
- The remaining bottleneck is now a slower plateau near hard 6.1 and soft 2.7 rather than the earlier hard 7.7 / soft 4.4 plateau.

Next step:
- Start loop15 from a single root-cause hypothesis.
- Avoid repeating loop11 and loop12 changes: initializing full student layers from the teacher and preserving analog residual block outputs both degraded hard-label behavior sharply in short probes.
- Prioritize the output-distribution bottleneck: despite better representation losses and teacher agreement, the target margin remains strongly negative and hard loss plateaus above 6.

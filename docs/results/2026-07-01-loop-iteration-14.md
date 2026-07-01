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

Interim result at step 620:
- Last hard loss: 6.1809
- Last soft loss: 2.8587
- Last 100-step hard mean: 6.2019
- Last 100-step soft mean: 2.8004
- Recent hard slope per 100 steps: -0.2962
- Recent soft slope per 100 steps: -0.2853
- Token accuracy in the latest logged step: 13.44%
- Teacher top-1 agreement in the latest logged step: 21.24%
- Readout scale in the latest logged step: 0.9282

Comparison with loop13:
- Loop13 stopped at step 504 with hard 6.5502, soft 3.0326, hard slope -0.0765, and soft slope -0.0258.
- Loop14 has lower hard and soft losses, a stronger recent descent, and previously reached 25.49% teacher top-1 agreement at step 596.

Decision:
- Continue loop14. It does not yet satisfy the strict hard/soft <5 pass criterion, but it now satisfies the early strong-downward-trend condition used for continuation.
- Do not stop for code modification while this trend holds.
- Keep monitoring through the 1B-token point, which occurs at approximately optimizer step 1272 for this configuration.

Next step:
- Re-check at checkpoint-step-1000 and at the 1B-token point.
- If hard and soft losses keep improving, treat this run as the accepted full-training continuation.
- If the 100-step and 250-step trends flatten before hard loss approaches 5, stop the run, record the final evidence, and start loop15 from a single new root-cause hypothesis.

# Loop Iteration 13 - Teacher Embedding Init + Unit Readout

Date: 2026-07-01

Run:
- Name: `loop13-stable-teacher-emb-readout1-seq1024-bs8-ga32-3xh200-20260701-084625`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/6k97ff3z
- Local file: `wandb/run-20260701_084631-6k97ff3z/run-6k97ff3z.wandb`
- Output: `output/loop13-stable-teacher-emb-readout1-seq1024-bs8-ga32-3xh200-20260701-084625`

Configuration:
- Teacher: `facebook/opt-125m`
- GPUs: 3x H200
- Sequence length: 1024
- Per-GPU batch size: 8
- Gradient accumulation: 32
- Time steps: 2
- Spike threshold: 0.7
- Readout scale: trainable, initialized at 1.0
- Student token and position embeddings initialized from the OPT teacher.
- SpAD weights unchanged: EA 0.2, SAA 0.1, SFA 0.1, STA 0.3, HTA 0.3
- Temperature: 2.0
- Optimizer/schedule unchanged: Adam, learning rate 5e-4, cosine schedule, warmup ratio 0.2, gradient clip 0.7

Result:
- Stopped at step 504 because hard loss plateaued above target.
- Last hard loss: 6.55019
- Last soft loss: 3.03262
- Last 100-step hard mean: 6.51724
- Last 100-step soft mean: 3.08205
- Recent hard slope per 100 steps: -0.07652
- Recent soft slope per 100 steps: -0.02578
- Token accuracy reached roughly 11-14% in the latest window.
- Teacher top-1 agreement reached roughly 18-22% in the latest window.

Decision:
- Best run so far, but still fail for full continuation because hard loss did not approach 5 and recent descent became weak.
- Keep the underlying code changes: teacher embedding initialization and unit readout scale.

Next step:
- Re-test `time_steps=4` on top of the now-stable teacher-embedding/readout setup.
- Use batch size 4 and gradient accumulation 64 to keep the same effective tokens per optimizer step without OOM.

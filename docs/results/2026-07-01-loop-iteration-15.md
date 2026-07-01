# Loop Iteration 15 - Final LayerNorm Teacher Init

Date: 2026-07-01

Run:
- Name: `loop15-sanity2-finalnorm-fresh-t4-seq1024-bs4-ga96-2xh200-20260701-174419`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/s3gapvlf
- Local file: `wandb/run-20260701_174458-s3gapvlf/run-s3gapvlf.wandb`
- Output: `output/loop15-sanity2-finalnorm-fresh-t4-seq1024-bs4-ga96-2xh200-20260701-174419`

Configuration:
- Teacher: `facebook/opt-125m`
- GPUs: 2x H200, because GPU1 was occupied by another user task.
- Sequence length: 1024
- Per-GPU batch size: 4
- Gradient accumulation: 96
- Effective tokens per optimizer step: 786,432
- Time steps: 4
- Spike threshold: 0.7
- Readout scale: trainable, initialized at 1.0
- Student token and position embeddings initialized from the OPT teacher.
- Student final layer norm initialized from the OPT teacher.
- SpAD weights unchanged: EA 0.2, SAA 0.1, SFA 0.1, STA 0.3, HTA 0.3
- Temperature: 2.0
- Optimizer/schedule unchanged: Adam, learning rate 5e-4, cosine schedule, warmup ratio 0.2, gradient clip 0.7

Final result:
- Stopped at step 280 because the hard/soft continuation rule was not met and loop15 was behind loop14 at comparable steps.
- Last hard loss: 6.8937
- Last soft loss: 3.2522
- Last 100-step hard mean: 6.7612
- Last 100-step soft mean: 3.3189
- 100-step hard slope per 100 steps: -0.2786
- 100-step soft slope per 100 steps: -0.3583
- Last 250-step hard mean: 7.1470
- Last 250-step soft mean: 3.7806
- 250-step hard slope per 100 steps: -0.5274
- 250-step soft slope per 100 steps: -0.6398
- Token accuracy in the latest logged step: 13.64%
- Teacher top-1 agreement in the latest logged step: 20.41%
- Target rank mean in the latest logged step: 1773.4
- Target margin mean in the latest logged step: -4.200
- Tokens seen at step 280: 220,200,960.
- No checkpoint was written because the run was intentionally stopped before `max_steps=700`.

Comparison with loop14:
- At step 277, loop15 had hard/soft losses 6.7256 / 3.2230.
- At step 277, loop14 had hard/soft losses 6.4662 / 3.0918.
- At step 280, loop15 had rebounded to hard/soft losses 6.8937 / 3.2522.
- Loop14 later reached hard/soft losses 6.1154 / 2.6974 at step 663, with teacher top-1 agreement 24.56%.

Decision:
- Fail for full-training continuation.
- Final layer norm teacher initialization did not improve the output-distribution bottleneck. It also did not preserve a robust short-window hard-loss descent after step 250.
- Keep loop14 as the best run so far.

Next step:
- Start loop16 from a single root-cause hypothesis focused on hard-label output calibration.
- Avoid repeating loop11 and loop12 changes: initializing full student layers from the teacher and preserving analog residual block outputs both degraded early hard-label behavior sharply.
- Do not add another broad teacher-weight initialization change unless there is direct evidence that the initialized component improves hard loss without hurting teacher agreement.

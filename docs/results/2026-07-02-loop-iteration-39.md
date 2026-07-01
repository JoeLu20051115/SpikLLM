# Loop Iteration 39 - Stable Clip 2GPU Medium Geometry

Date: 2026-07-02

## Hypothesis

Loop14 did not satisfy the continuation rule at step 80, but did satisfy the short-window `extend` rule by step 160 and later reached the best long-run losses so far. The 80-step gates after loop36 may therefore be too short to detect a viable geometry. This loop tests a medium probe closer to loop14 while avoiding GPU1, which is occupied by another process.

## Setup

- Code state: stable gradient clipping from loop38.
- Run: `loop39-stableclip-seq1024-bs4ga64-2xh200-160step-20260702-030008`
- Local W&B: `wandb/offline-run-20260702_030011-uv6boauc/run-uv6boauc.wandb`
- Output: `output/loop39-stableclip-seq1024-bs4ga64-2xh200-160step-20260702-030008`
- GPUs: 2x H200, GPU0 and GPU2
- Sequence length: 1024
- Time steps: 4
- Per-GPU batch size: 4
- Gradient accumulation: 64
- Effective tokens per optimizer step: 524,288
- Max optimizer steps: 160
- Precision: bf16
- W&B mode: offline
- GPU1 was not used.

## Result

Loop39 completed without OOM or NaNs.

- Step 160 hard/soft: 7.7017 / 4.3892
- First 25 hard/soft means: 8.5164 / 5.1145
- Last 25 hard/soft means: 7.6977 / 4.4396
- Last 80 hard/soft means: 7.7191 / 4.4355
- Last 25 hard/soft slopes per 100 steps: -0.2216 / +0.2434
- Last 80 hard/soft slopes per 100 steps: -0.0658 / -0.0863
- Step 160 token accuracy: 3.64%
- Step 160 teacher top-1 agreement: 5.87%
- Step 160 target rank / margin: 2886.0 / -4.4343
- Step 160 tokens seen: 83,886,080

## Comparison With Loop14 at Step 160

Loop14 geometry:
- GPUs: 3x H200
- Sequence length: 1024
- Per-GPU batch size: 4
- Gradient accumulation: 64
- Effective tokens per optimizer step: 786,432
- Step 160 tokens seen: 125,829,120

Loop14 step 160:
- Hard/soft: 7.1540 / 3.6369
- Last 25 hard/soft means: 7.1954 / 3.7994
- Last 25 hard/soft slopes per 100 steps: -1.2872 / -1.1085
- Token accuracy: 6.52%
- Teacher top-1 agreement: 11.90%
- Target rank / margin: 2896.1 / -5.1233
- Analyzer decision with window 25: `extend`

Loop39 is worse than loop14 on hard loss, soft loss, last-25 trend, token accuracy, and teacher agreement at the same optimizer step. It also fails to show the strong short-window descent that justified extending loop14.

## Decision

Fail loop39. Do not extend this run to 240 steps or launch a full run:

- It does not meet the continuation rule.
- The recent soft-loss slope is positive in the last 25-step window.
- At step 160 it is materially behind loop14 despite using stable clipping.
- The gap is not just a single metric; it covers hard/soft losses and output diagnostics.

Keep loop16 as the current official small-gate metric baseline and loop14 as the best historical long-run behavior. The next loop should not spend more compute on this exact 2GPU geometry unless GPU1 becomes available for a true loop14-matched rerun or a separate code-level hypothesis first beats the baseline on a small gate.

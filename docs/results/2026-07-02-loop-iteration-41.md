# Loop Iteration 41 - Long Scheduler Horizon Small Probe

Date: 2026-07-02

## Hypothesis

All recent 80/160-step probes used `max_steps` both as the early-stop limit and as the cosine scheduler horizon. That compresses the learning-rate schedule: for an 80-step gate, LR is already decayed to zero at step 80. Loop14, by contrast, used a 2000-step run plan and was still in warmup at step 160 when it first satisfied the short-window `extend` rule.

This loop adds and tests a separate scheduler horizon so a probe can stop early while using a longer training schedule.

## Code Change

- Added `TrainingConfig.scheduler_max_steps`.
- Added `--scheduler-max-steps` CLI argument.
- Added `resolve_scheduler_steps`.
- Training still stops at `max_steps`, but warmup/cosine schedule uses `scheduler_max_steps` when provided.

Verification:
- RED test: `test_probe_scheduler_horizon_can_differ_from_stop_steps` failed before implementation because `resolve_scheduler_steps` did not exist.
- Targeted GREEN passed.
- Full smoke passed: `45 passed, 58 warnings`.
- Dry-run confirmed `resolved_max_steps=80` and `scheduler_max_steps=2000` can coexist.

Code commit:
- `7c4d24c train: decouple scheduler horizon for probes`

## Probe Setup

- Run: `loop41-longscheduler-small-seq512-bs2ga16-t4-160step-20260702-042129`
- Local W&B: `wandb/offline-run-20260702_042131-wuk8hr2r/run-wuk8hr2r.wandb`
- Output: `output/loop41-longscheduler-small-seq512-bs2ga16-t4-160step-20260702-042129`
- GPU: 1x H200, GPU0
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Stop steps: 160
- Scheduler horizon: 2000
- Warmup steps: 400
- Precision: bf16

## Result

- Step 160 hard/soft: 7.9169 / 4.8862
- First 25 hard/soft means: 9.7014 / 5.9675
- Last 25 hard/soft means: 7.8139 / 4.3951
- Last 80 hard/soft means: 7.7938 / 4.3581
- Last 25 hard/soft slopes per 100 steps: +0.3403 / -0.5043
- Last 80 hard/soft slopes per 100 steps: +0.0050 / +0.0215
- Step 160 token accuracy: 4.21%
- Step 160 teacher top-1 agreement: 6.85%
- Step 160 target rank / margin: 4733.3 / -4.8835
- Step 160 LR: 0.0002
- Tokens seen: 2,621,440

## Decision

Fail loop41. The long scheduler horizon is a useful training-harness capability, but it does not solve the current official-small-geometry training behavior:

- Hard and soft losses are not both below 5.
- Recent hard loss is not descending in the last 25 or last 80 steps.
- Step-160 hard/soft and output diagnostics are worse than loop16/loop38 small-gate behavior.

Keep the scheduler-horizon support for future probes, because it makes early-stop diagnostics closer to real long-run scheduling. Do not launch a full run from this result.

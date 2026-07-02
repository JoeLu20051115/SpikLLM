# Loop Iteration 82 - Current Best Baseline Three-GPU Run

Date: 2026-07-02

## Purpose

After loop77-81 failed to produce a small-screen candidate that clearly beats the current best reference, the user requested running the best baseline directly on three GPUs.

This run uses the current official small-gate best code baseline from loop16, with the proven loop14 three-GPU geometry:

- current `main` source: same-dimension SpAD projectors are exact identities;
- student token and position embeddings are initialized from the OPT teacher;
- readout scale is trainable and initialized at 1.0;
- SFSA/SFFN structure is unchanged;
- all five SpAD losses and paper weights are unchanged: `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`.

## Baseline Choice

- Official small-gate baseline: loop16.
  - Step80 hard/soft: `7.4532 / 4.5195`.
  - Last25 hard/soft: `7.7798 / 4.2932`.
- Best historical long-run behavior: loop14.
  - 3x H200, `seq=1024`, `T=4`, per-rank batch `4`, gradient accumulation `64`.
  - Stopped at step663 with hard/soft `6.1154 / 2.6974`.

Loop82 combines these: loop16/current source with loop14 long-run geometry.

## Launch

Formal run:

- Run name: `loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845`
- tmux session: `loop82_loop16_baseline_seq1024_bs4_ga64_3xh200_1bt_20260702_084845`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/9xwae378
- Local W&B: `wandb/run-20260702_084851-9xwae378/run-9xwae378.wandb`
- Output: `output/loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845`
- Log: `logs/loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845/train.log`
- Launch script: `logs/loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845/launch.sh`

An earlier `nohup` attempt exited without entering training because the child process was not kept alive after the exec session ended. A foreground `timeout` diagnostic proved the command was valid and created W&B run `ar58b7du`; that diagnostic run was killed by the timeout and is not the formal loop82 training run.

## Command

```bash
.venv/bin/torchrun --nproc_per_node=3 --master_port=29783 \
  -m bispikclm.train.train_spad \
  --config configs/bispikclm_opt125m_spad.toml \
  --output-dir output/loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845 \
  --sequence-length 1024 \
  --time-steps 4 \
  --batch-size 4 \
  --gradient-accumulation-steps 64 \
  --precision bf16 \
  --wandb \
  --wandb-project bispikclm \
  --wandb-run-name loop82-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-084845 \
  --train
```

With the config's `target_tokens=1000000000`, this geometry resolves to about 1272 optimizer steps:

`4 batch/rank * 64 grad_accum * 1024 seq * 3 ranks = 786432 tokens/step`.

## Launch Verification

Immediately after tmux launch:

- torchrun parent process was active on master port `29783`;
- three rank worker processes were active;
- GPUs 0, 1, and 2 each held about `83.7 GiB` and reported `100%` utilization;
- W&B online sync was active for run `9xwae378`.

## Monitoring Plan

Monitor W&B and the local log. Apply the same continuation rule:

- continue if hard and soft are below 5 early, or both show a clear downward trend;
- otherwise stop and record the final state if the trend stalls as in loop14.

No source code was changed for this loop.

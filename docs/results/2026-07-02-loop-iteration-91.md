# Loop Iteration 91 - Three-GPU Baseline Cancelled By Resource Conflict

Date: 2026-07-02

## Purpose

After loop89 screen160 and loop90 ended, all three GPUs were free. A clean three-GPU run of the current best baseline was launched immediately with the loop14 geometry.

Intended setup:

- current `main` / loop16 source baseline;
- loop14 three-GPU geometry: `seq=1024`, `T=4`, per-rank batch `4`, gradient accumulation `64`;
- three GPUs: `CUDA_VISIBLE_DEVICES=0,1,2`;
- precision: `bf16`;
- W&B enabled.

## Three-GPU Attempt

- Run name: `loop91-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-092542`
- tmux session: `loop91_loop16_baseline_seq1024_bs4_ga64_3xh200_1bt_20260702_092542`
- Port: `57055`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/fs1oeuwz
- Local W&B: `wandb/run-20260702_092548-fs1oeuwz/run-fs1oeuwz.wandb`
- Log: `logs/loop91-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-092542/train.log`
- Output: `output/loop91-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-092542`

The run reached W&B initialization but had zero optimizer-step rows.

## Resource Conflict

Immediately after launch, two separate single-GPU baseline sessions appeared and occupied GPUs 0 and 1:

- `loop91a-loop16-baseline-seq512-bs2-ga16-gpu0-1bt-20260702-092515`
  - tmux: `loop91a_loop16_baseline_seq512_bs2_ga16_gpu0_1bt_20260702_092515`
  - W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/8dsmnrub
  - local W&B: `wandb/run-20260702_092548-8dsmnrub/run-8dsmnrub.wandb`
- `loop91b-loop16-baseline-seq512-bs2-ga16-gpu1-1bt-20260702-092515`
  - tmux: `loop91b_loop16_baseline_seq512_bs2_ga16_gpu1_1bt_20260702_092515`
  - W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/bh2edq5w
  - local W&B: `wandb/run-20260702_092548-bh2edq5w/run-bh2edq5w.wandb`

Because the three-GPU run would have shared GPUs 0 and 1 with these single-GPU sessions, the three-GPU tmux session was stopped to avoid a contaminated baseline and wasted resources.

At first parse:

| Run | Rows | Last step | Hard | Soft |
| --- | ---: | ---: | ---: | ---: |
| loop91 three-GPU | 0 | n/a | n/a | n/a |
| loop91a single-GPU | 32 | 32 | 10.9300 | 7.1140 |
| loop91b single-GPU | 20 | 20 | 11.1989 | 7.0685 |

Decision: cancelled resource-conflict attempt. No baseline update. A valid three-GPU baseline still requires all three GPUs to be free at launch and during early monitoring.

## Single-GPU Monitor - 2026-07-02T09:54:58+08:00

The single-GPU runs that caused the resource conflict are still active and blocking a clean three-GPU baseline launch.

Latest local W&B parse:

| Run | Rows | Last step | Hard | Soft | Total | Token acc. | Teacher agree | Tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop91a | 976 | 976 | 6.8924 | 3.4537 | 3.7319 | 8.90% | 14.38% | 15,990,784 |
| loop91b | 927 | 927 | 7.1240 | 3.6305 | 3.8672 | 9.10% | 15.36% | 15,187,968 |

Recent trend windows:

| Run | Window | Hard mean | Soft mean | Hard slope/100 | Soft slope/100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| loop91a | last25 | 7.0463 | 3.5719 | +0.3994 | -0.4027 |
| loop91a | last50 | 7.0254 | 3.5583 | +0.1419 | +0.0221 |
| loop91a | last100 | 7.0438 | 3.5570 | -0.0556 | -0.0092 |
| loop91a | last200 | 7.1002 | 3.6219 | -0.1061 | -0.1123 |
| loop91b | last25 | 7.1631 | 3.6664 | -1.0403 | -0.6275 |
| loop91b | last50 | 7.1474 | 3.6514 | -0.0892 | -0.0382 |
| loop91b | last100 | 7.1801 | 3.6954 | -0.0968 | -0.1420 |
| loop91b | last200 | 7.2142 | 3.7511 | -0.0697 | -0.1088 |

Interpretation: soft loss is below 5 on both single-GPU runs and hard loss is still above 5. The long-window hard/soft trend remains weakly downward, but these runs are not the requested three-GPU loop14 geometry and cannot replace the queued clean baseline.

## Single-GPU Monitor - 2026-07-02T10:24:56+08:00

The same single-GPU runs remain active on GPUs 0 and 1. Both are still improving on longer windows, but hard loss remains above 5.

Latest local W&B parse:

| Run | Rows | Last step | Hard | Soft | Total | Token acc. | Teacher agree | Tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop91a | 1,960 | 1,960 | 6.4623 | 2.9015 | 3.2410 | 13.11% | 26.52% | 32,112,640 |
| loop91b | 1,874 | 1,874 | 6.4577 | 3.0981 | 3.3003 | 11.06% | 19.67% | 30,703,616 |

Recent trend windows:

| Run | Window | Hard mean | Soft mean | Hard slope/100 | Soft slope/100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| loop91a | last25 | 6.5001 | 2.9932 | +0.4407 | +0.4002 |
| loop91a | last50 | 6.5443 | 2.9973 | -0.2230 | +0.0119 |
| loop91a | last100 | 6.5500 | 3.0107 | -0.0488 | -0.0489 |
| loop91a | last200 | 6.5748 | 3.0411 | -0.0534 | -0.0565 |
| loop91a | last400 | 6.6305 | 3.0900 | -0.0508 | -0.0486 |
| loop91a | last800 | 6.6946 | 3.2022 | -0.0390 | -0.0569 |
| loop91b | last25 | 6.5999 | 3.0789 | -0.0602 | -0.0817 |
| loop91b | last50 | 6.6137 | 3.0884 | -0.0724 | -0.1174 |
| loop91b | last100 | 6.6476 | 3.1150 | -0.1244 | -0.0811 |
| loop91b | last200 | 6.6859 | 3.1410 | -0.0814 | -0.0529 |
| loop91b | last400 | 6.7181 | 3.1973 | -0.0333 | -0.0591 |
| loop91b | last800 | 6.8057 | 3.3200 | -0.0454 | -0.0613 |

Interpretation: these single-GPU runs support the current best baseline direction because soft loss is well below 5 and hard loss keeps moving down over 100+ step windows. They still do not satisfy the hard-loss threshold, and they do not replace the queued loop95 three-GPU loop14-geometry run.

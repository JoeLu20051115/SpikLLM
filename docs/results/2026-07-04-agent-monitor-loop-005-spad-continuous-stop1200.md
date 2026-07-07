# Agent Monitor Loop 005: SpAD-Continuous Stop At 1200

Date: 2026-07-04 12:42:48 +08

Scope: stop the active SpAD-continuous control run immediately after `checkpoint-step-1200.pt` was written and verified.

## Run

Output directory:

```text
/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/loop-opd-sa-topd/output/spad-continuous-3xh200-step1000-to4000-ckpt100-20260704-103227
```

Started from:

```text
checkpoint-step-1000.pt
```

Stopping rule requested:

```text
Stop after checkpoint-step-1200.pt is saved.
```

## Stop Execution

Monitor observed:

```text
2026-07-04 12:40:45 +08 latest_step=1199 checkpoint_1200=missing
2026-07-04 12:41:15 +08 checkpoint_1200_stable size=1502657945 latest_step=1200
```

After the file was present and size-stable, the SpAD process group was terminated:

```text
2300969
2300971
2301030
2301031
2301032
```

The torchrun log records the expected signal shutdown after the requested stop:

```text
Received 1 death signal, shutting down workers
Sending process 2301030 closing signal SIGHUP
Sending process 2301031 closing signal SIGHUP
Sending process 2301032 closing signal SIGHUP
```

Evaluation: this signal traceback is an expected result of the manual stop after the target checkpoint was saved, not evidence that training crashed before the checkpoint.

## Checkpoint Verification

Saved checkpoints:

```text
checkpoint-step-1100.pt  1.4G  2026-07-04 11:19
checkpoint-step-1200.pt  1.4G  2026-07-04 12:41
```

`checkpoint-step-1200.pt` was loaded with `torch.load(..., map_location="cpu")`.

Verified checkpoint contents:

```text
step = 1200
keys = ['config', 'embedding_projector', 'hidden_projector', 'optimizer', 'scheduler', 'step', 'student']
```

## Final Metrics Row

Last metrics row:

```text
train/step = 1200
train/tokens_seen = 943718400
loss/total_loss = 2.5345164239406586
loss/hard_loss = 5.790673457086086
loss/soft_loss = 2.444898247718811
train/grad_norm = 0.21172700822353363
train/silent_layer_count = 0.0
train/overactive_layer_count = 0.0
train/teacher_topk_mass = 0.8711955547332764
train/teacher_top1_agreement = 0.23118279874324799
train/target_rank_mean = 828.3917236328125
train/target_margin_mean = -3.4315125942230225
```

Evaluation: metrics at step 1200 show no NaN, no OOM, no grad-norm explosion, and clean spike-health counters. The run was intentionally stopped after saving the requested checkpoint.

## Post-Stop Process And GPU State

Post-stop process check:

```text
No SpAD train_spad or torchrun process remained.
```

Post-stop GPU state:

| GPU | Util | Memory | Assessment |
| ---: | ---: | ---: | --- |
| 0 | 0% | 4 / 143771 MiB | SpAD released |
| 1 | 100% | 51577 / 143771 MiB | external `yueming` job remains |
| 2 | 0% | 4 / 143771 MiB | SpAD released |

Remaining GPU process:

```text
2308514 python on GPU1, ~51568 MiB
```

## Monitor Decision

The requested stop condition was satisfied. Keep `checkpoint-step-1200.pt` as the current SpAD-continuous control checkpoint for follow-up evaluation.

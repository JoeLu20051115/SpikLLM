# Agent Monitor Loop 004: SpAD-Continuous Status

Date: 2026-07-04 12:01:26 +08

Scope: monitor the active SpAD-continuous three-H200 run launched from the canonical step1000 checkpoint.

## Run

Output directory:

```text
/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/loop-opd-sa-topd/output/spad-continuous-3xh200-step1000-to4000-ckpt100-20260704-103227
```

Launch command:

```text
torchrun --standalone --nproc_per_node=3 -m bispikclm.train.train_spad \
  --config configs/bispikclm_opt125m_spad_continue_ckpt100.toml \
  --resume-from checkpoint-step-1000.pt \
  --max-steps 4000 \
  --scheduler-max-steps 4000 \
  --sequence-length 1024 \
  --time-steps 4 \
  --batch-size 4 \
  --gradient-accumulation-steps 64 \
  --seed 1000 \
  --train
```

## Current Progress

Latest observed metrics row:

```text
train/step = 1155
train/tokens_seen = 908328960
loss/total_loss = 2.5886658616364002
loss/hard_loss = 5.867967359721661
loss/soft_loss = 2.539132658392191
train/grad_norm = 0.28330227732658386
train/silent_layer_count = 0.0
train/overactive_layer_count = 0.0
train/teacher_topk_mass = 0.8356866836547852
train/teacher_top1_agreement = 0.2646627426147461
```

Evaluation: the job is still active and has advanced past the first checkpoint boundary. No NaN, OOM, or CUDA crash was observed in the latest metrics. Spike health is clean at the latest observed row.

## Checkpoints

Existing checkpoint:

```text
checkpoint-step-1100.pt  1.4G  2026-07-04 11:19
```

Next expected checkpoint:

```text
checkpoint-step-1200.pt
```

At this monitor point, `checkpoint-step-1200.pt` had not appeared yet because the run was at step 1155.

## GPU And Process State

SpAD worker processes:

```text
2301030
2301031
2301032
```

GPU state:

| GPU | Util | Total Memory Used | SpAD Worker Memory | Assessment |
| ---: | ---: | ---: | ---: | --- |
| 0 | 100% | 134871 / 143771 MiB | 87142 MiB | active, shared |
| 1 | 100% | 137481 / 143771 MiB | 86366 MiB | active, shared |
| 2 | 100% | 130145 / 143771 MiB | 86398 MiB | active, shared |

Additional non-SpAD GPU jobs observed:

```text
yueming 2311279 python Full_Comparision_NormalizedAtt_Prime.py --gpu 0  ~47714 MiB
yueming 2308514 python Full_Comparision_NormalizedAtt_Prime.py --gpu 1  ~51100 MiB
yueming 2309945 python Full_Comparision_NormalizedAtt_Prime.py --gpu 2  ~43732 MiB
```

Evaluation: the SpAD run remains alive, but all three GPUs are now shared with unrelated `yueming` jobs and have limited remaining memory headroom. This increases OOM risk for future checkpointing or transient allocation spikes.

## Log Notes

`train.log` contains HuggingFace dataset network retry warnings near startup, including read timeouts and temporary connectivity failures. These did not stop the run; metrics continued through at least step 1155.

## Monitor Decision

Continue monitoring. Do not restart or modify the run.

Next checks:

1. Confirm `checkpoint-step-1200.pt` appears.
2. Watch GPU memory carefully because external jobs are now sharing all three cards.
3. Re-check metrics around step 1200 for grad norm, spike health, loss drift, and teacher agreement.

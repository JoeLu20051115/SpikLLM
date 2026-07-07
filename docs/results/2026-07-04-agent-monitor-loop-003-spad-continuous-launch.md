# Agent Monitor Loop 003: SpAD-Continuous Launch

Date: 2026-07-04 10:33:41 +08

Scope: launch and verify a three-H200 SpAD-continuous run from the canonical step1000 SpAD checkpoint. This loop is not SA-TOPD; it is the SpAD-only continuation control requested for comparison.

## Pre-Launch Finding

At the start of this loop, all three GPUs were occupied by an already-running K96 TOPD job:

```text
torchrun --standalone --nproc_per_node=3 -m bispikclm.train.train_topd ...
--config configs/bispikclm_opt125m_topd_k96_full.toml
--top-k 96 --disable-rate-bridge --disable-spike-reg
--output-dir output/sa-topd-k96-full-3xh200-step1000-4000step-20260704-101721
```

Evaluation: that job was not the requested SpAD-continuous control. It was stopped before launching this loop to avoid mixing TOPD and SpAD-only continuation results on the same three GPUs.

## Configuration Change

A dedicated SpAD continuation config was added:

```text
configs/bispikclm_opt125m_spad_continue_ckpt100.toml
```

Only material change versus the default OPT-125M SpAD config:

```text
checkpoint_interval = 100
```

Reason: `train_spad.py` reads checkpoint cadence from TOML and has no CLI override for checkpoint interval.

## Launch

Worktree:

```text
/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/loop-opd-sa-topd
```

Output directory:

```text
output/spad-continuous-3xh200-step1000-to4000-ckpt100-20260704-103227
```

tmux session:

```text
spad_cont_20260704-103227
```

Worker processes observed:

```text
2301030
2301031
2301032
```

Launch intent:

| Field | Value |
| --- | --- |
| Mode | SpAD-only continuation |
| Resume checkpoint | loop14 step1000 SpAD checkpoint |
| Resume step | 1000 from checkpoint state |
| Max step | 4000 |
| Checkpoint interval | 100 |
| GPUs | 0,1,2 |
| DDP workers | 3 |
| Sequence length | 1024 |
| Batch size | 4 |
| Gradient accumulation | 64 |
| Time steps | 4 |
| Prompt buffer | none; streaming FineWeb-Edu |
| WandB | disabled |

## Verification

Resolved config confirmed:

```text
checkpoint_interval = 100
max_steps = 4000
scheduler_max_steps = 4000
sequence_length = 1024
batch_size = 4
gradient_accumulation_steps = 64
time_steps = 4
prompt_buffer_path = null
resume_from = checkpoint-step-1000.pt
```

First metrics row confirmed the continuation started after the checkpoint:

```text
train/step = 1001
train/peak_memory_gb = 76.26798009872437
train/silent_layer_count = 0.0
train/overactive_layer_count = 0.0
train/teacher_topk_mass = 0.8365685939788818
```

GPU state after launch:

| GPU | Util | Memory | Worker |
| ---: | ---: | ---: | --- |
| 0 | 100% | 87151 / 143771 MiB | 2301030 |
| 1 | 100% | 85589 / 143771 MiB | 2301031 |
| 2 | 100% | 85621 / 143771 MiB | 2301032 |

## Monitor Decision

Launch accepted. The run is now the correct SpAD-continuous three-H200 control from step1000, with checkpoints scheduled every 100 steps.

Next monitor checks:

1. Confirm `checkpoint-step-1100.pt` appears.
2. Check metrics around steps 1100, 1200, and 1300 for NaN, OOM, grad norm spikes, silent/overactive layers, and loss divergence.
3. Record whether this SpAD-continuous control improves, degrades, or stays flat relative to the earlier step1000 and step2000 evaluations.

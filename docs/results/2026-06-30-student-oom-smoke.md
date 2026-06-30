# 2026-06-30 BiSpikCLM-125M Student OOM Smoke

目标：在 3x NVIDIA H200 NVL 上验证 BiSpikCLM-125M 学生模型使用 FineWeb-Edu sample-10BT、SpAD 训练、seq=1024、W&B 监控时是否会 OOM。

## 结果

| 配置 | 结果 | 备注 |
| --- | --- | --- |
| seq=1024, batch_size=16, grad_accum=16, nproc=3, max_steps=1 | OOM | OOM 发生在 SpAD attention rate encoding，单卡约 136GB 已分配后仍需额外显存。 |
| seq=1024, batch_size=8, grad_accum=32, nproc=3, max_steps=1 | 通过 | W&B run: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/24o2wrsk |

通过配置的关键指标：

| 指标 | 数值 |
| --- | --- |
| tokens_seen | 786,432 |
| peak_memory_gb | 92.458 |
| total_loss | 2071.6445 |
| soft_loss | 6893.7930 |
| hard_loss | 10.8262 |
| grad_norm | 43.1922 |
| lr | 0.0005 |

## 可用于正式 1B token 训练的命令

不要设置 `MAX_STEPS`，让配置里的 `target_tokens = 1000000000` 生效。按当前配置，每个 optimizer step 处理：

`8 * 32 * 1024 * 3 = 786432` tokens

因此 1B token 需要约：

`ceil(1000000000 / 786432) = 1272` optimizer steps

```bash
cd /mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM

PATH="$PWD/.venv/bin:$PATH" \
WANDB=1 \
WANDB_PROJECT=bispikclm \
WANDB_RUN_NAME=opt125m-seq1024-bs8-ga32-3xh200-1bt \
NPROC_PER_NODE=3 \
SEQUENCE_LENGTH=1024 \
BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=32 \
bash scripts/run_sft.sh
```

## 注意

当前不建议启用 PyTorch gradient checkpointing：SpikingJelly LIF 节点是有状态模块，checkpoint 的反向重算可能污染膜电位状态；并且本次主要显存瓶颈来自 SpAD 需要保留的 attention、hidden_states 和 logits，checkpoint 不能完整消除这部分占用。

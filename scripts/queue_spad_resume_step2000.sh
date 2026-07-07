#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--monitor" ]]; then
  metrics_jsonl="${2:?metrics_jsonl is required}"
  monitor_log="${3:?monitor_log is required}"
  train_session="${4:?train_session is required}"
  interval_seconds="${MONITOR_INTERVAL_SECONDS:-120}"

  mkdir -p "$(dirname "$monitor_log")"
  while tmux has-session -t "$train_session" 2>/dev/null; do
    timestamp="$(date --iso-8601=seconds)"
    metric_summary="$({
      python - "$metrics_jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("status=WAIT metrics=pending")
    raise SystemExit(0)

lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    print("status=WAIT metrics=empty")
    raise SystemExit(0)

row = json.loads(lines[-1])
step = row.get("train/step")
total_loss = row.get("loss/total_loss")
grad_norm = row.get("train/grad_norm")
silent = row.get("train/silent_layer_count")
overactive = row.get("train/overactive_layer_count")
topk_mass = row.get("train/teacher_topk_mass")
top1 = row.get("train/teacher_top1_agreement")
lr = row.get("train/lr")
peak_memory = row.get("train/peak_memory_gb")

flags = []
if silent not in (None, 0, 0.0):
    flags.append(f"silent={silent}")
if overactive not in (None, 0, 0.0):
    flags.append(f"overactive={overactive}")
if isinstance(total_loss, (int, float)) and not math.isfinite(total_loss):
    flags.append("loss=nonfinite")
if isinstance(grad_norm, (int, float)) and (not math.isfinite(grad_norm) or grad_norm > 1.0):
    flags.append(f"grad_norm={grad_norm:.4f}")
if isinstance(topk_mass, (int, float)) and (topk_mass < 0.60 or topk_mass > 0.95):
    flags.append(f"topk_mass={topk_mass:.4f}")

status = "ALERT" if flags else "OK"
parts = [
    f"status={status}",
    f"step={step}",
    f"total_loss={total_loss}",
    f"grad_norm={grad_norm}",
    f"silent={silent}",
    f"overactive={overactive}",
    f"topk_mass={topk_mass}",
    f"top1={top1}",
    f"lr={lr}",
    f"peak_memory_gb={peak_memory}",
]
if flags:
    parts.append("flags=" + ",".join(flags))
print(" ".join(parts))
PY
    } 2>&1)"
    gpu_summary="$({
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
        awk -F', ' '{printf "gpu%s=%s/%sMiB@%s%%;", $1, $2, $3, $4}'
    } 2>&1)"
    printf '[%s] %s | %s\n' "$timestamp" "$metric_summary" "$gpu_summary" | tee -a "$monitor_log"
    sleep "$interval_seconds"
  done
  printf '[%s] status=STOP training_session=%s ended\n' "$(date --iso-8601=seconds)" "$train_session" | tee -a "$monitor_log"
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
worktree="${WORKTREE:-$repo_root/.worktrees/loop-opd-sa-topd}"
resume_from="${RESUME_FROM:-$repo_root/.worktrees/loop14-baseline-5e4d6df/output/loop14-t4-teacher-emb-readout1-seq1024-bs4-ga64-3xh200-20260701-110431/checkpoint-step-2000.pt}"
config_path="${CONFIG_PATH:-configs/bispikclm_opt125m_spad_continue_ckpt100.toml}"
max_steps="${MAX_STEPS:-4000}"
scheduler_max_steps="${SCHEDULER_MAX_STEPS:-4000}"
sequence_length="${SEQUENCE_LENGTH:-1024}"
time_steps="${TIME_STEPS:-4}"
batch_size="${BATCH_SIZE:-4}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-64}"
seed="${SEED:-1000}"
queue_interval_seconds="${QUEUE_INTERVAL_SECONDS:-120}"
gpu_mem_threshold_mib="${GPU_MEM_THRESHOLD_MIB:-8192}"
gpu_util_threshold="${GPU_UTIL_THRESHOLD:-20}"
timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="spad-orig-3xh200-step2000-to4000-ckpt100-${timestamp}"
output_dir="$worktree/output/$run_name"
metrics_jsonl="$output_dir/metrics.jsonl"
train_log="$output_dir/train.log"
queue_log="$output_dir/queue.log"
monitor_log="$output_dir/health-monitor.log"
run_info="$output_dir/run-info.txt"
launch_cmd_file="$output_dir/launch.cmd"
train_session="spad_resume_${timestamp}"
monitor_session="${train_session}_health"

mkdir -p "$output_dir"

gpu_gate() {
  python - "$gpu_mem_threshold_mib" "$gpu_util_threshold" <<'PY'
import csv
import subprocess
import sys

mem_threshold = int(sys.argv[1])
util_threshold = int(sys.argv[2])
cmd = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
rows = []
for line in subprocess.check_output(cmd, text=True).strip().splitlines():
    index, memory_used, util = [item.strip() for item in line.split(",")]
    rows.append((int(index), int(memory_used), int(util)))

rows.sort()
summary = "; ".join(f"gpu{idx}:mem={mem}MiB util={util}%" for idx, mem, util in rows)
ready = all(mem <= mem_threshold and util <= util_threshold for idx, mem, util in rows[:3])
print(summary)
raise SystemExit(0 if ready else 1)
PY
}

launch_cmd=(
  env CUDA_VISIBLE_DEVICES=0,1,2
  torchrun --standalone --nproc_per_node=3
  -m bispikclm.train.train_spad
  --config "$config_path"
  --resume-from "$resume_from"
  --output-dir "$output_dir"
  --metrics-jsonl "$metrics_jsonl"
  --max-steps "$max_steps"
  --scheduler-max-steps "$scheduler_max_steps"
  --sequence-length "$sequence_length"
  --time-steps "$time_steps"
  --batch-size "$batch_size"
  --gradient-accumulation-steps "$gradient_accumulation_steps"
  --seed "$seed"
  --wandb
  --wandb-project bispikclm
  --wandb-run-name "$run_name"
  --train
)

{
  printf 'worktree=%s\n' "$worktree"
  printf 'resume_from=%s\n' "$resume_from"
  printf 'config_path=%s\n' "$config_path"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'metrics_jsonl=%s\n' "$metrics_jsonl"
  printf 'train_log=%s\n' "$train_log"
  printf 'queue_log=%s\n' "$queue_log"
  printf 'monitor_log=%s\n' "$monitor_log"
  printf 'train_session=%s\n' "$train_session"
  printf 'monitor_session=%s\n' "$monitor_session"
  printf 'max_steps=%s\n' "$max_steps"
  printf 'scheduler_max_steps=%s\n' "$scheduler_max_steps"
  printf 'sequence_length=%s\n' "$sequence_length"
  printf 'time_steps=%s\n' "$time_steps"
  printf 'batch_size=%s\n' "$batch_size"
  printf 'gradient_accumulation_steps=%s\n' "$gradient_accumulation_steps"
  printf 'seed=%s\n' "$seed"
} > "$run_info"

printf '%q ' "${launch_cmd[@]}" > "$launch_cmd_file"
printf '\n' >> "$launch_cmd_file"

printf '[%s] queue_start run_name=%s\n' "$(date --iso-8601=seconds)" "$run_name" | tee -a "$queue_log"
until gpu_status="$(gpu_gate)"; do
  printf '[%s] waiting_for_gpus %s\n' "$(date --iso-8601=seconds)" "$gpu_status" | tee -a "$queue_log"
  sleep "$queue_interval_seconds"
done
printf '[%s] gpu_gate_open %s\n' "$(date --iso-8601=seconds)" "$gpu_status" | tee -a "$queue_log"

tmux new-session -d -s "$train_session" -c "$worktree" "${launch_cmd[*]} 2>&1 | tee '$train_log'"
tmux new-session -d -s "$monitor_session" -c "$worktree" "'$0' --monitor '$metrics_jsonl' '$monitor_log' '$train_session'"

printf '[%s] launched train_session=%s monitor_session=%s\n' "$(date --iso-8601=seconds)" "$train_session" "$monitor_session" | tee -a "$queue_log"
printf 'output_dir=%s\ntrain_session=%s\nmonitor_session=%s\n' "$output_dir" "$train_session" "$monitor_session"
#!/usr/bin/env bash
set -euo pipefail

args=(
  --config "${CONFIG:-configs/bispikclm_opt125m_spad.toml}"
)

if [[ -n "${TEACHER_MODEL:-}" ]]; then
  args+=(--teacher-model "${TEACHER_MODEL}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${SEQUENCE_LENGTH:-}" ]]; then
  args+=(--sequence-length "${SEQUENCE_LENGTH}")
fi
if [[ -n "${TIME_STEPS:-}" ]]; then
  args+=(--time-steps "${TIME_STEPS}")
fi
if [[ -n "${PRECISION:-}" ]]; then
  args+=(--precision "${PRECISION}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  args+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  args+=(--learning-rate "${LEARNING_RATE}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "${MAX_STEPS}")
fi
if [[ "${WANDB:-0}" == "1" ]]; then
  args+=(--wandb --wandb-project "${WANDB_PROJECT:-bispikclm}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    args+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
fi

args+=("${TRAIN_MODE:---train}")

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" -m bispikclm.train.train_spad "${args[@]}"

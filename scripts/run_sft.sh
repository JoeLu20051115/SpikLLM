#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" -m bispikclm.train.train_spad \
  --teacher-model "${TEACHER_MODEL:-models/opt-125m}" \
  --output-dir "${OUTPUT_DIR:-output/v1-opt-sft}" \
  --sequence-length "${SEQUENCE_LENGTH:-16}" \
  --time-steps "${TIME_STEPS:-2}" \
  --max-steps "${MAX_STEPS:-1}" \
  --dummy-batch

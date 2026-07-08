#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:?run_dir is required}"
baseline_json="${2:?baseline_json is required}"
interval_seconds="${MONITOR_INTERVAL_SECONDS:-120}"
checkpoint_path="$run_dir/checkpoint-step-2100.pt"
eval_dir="$run_dir/wikitext-ppl-step2100"
eval_json="$eval_dir/wikitext_step2100_rawtest.json"
compare_json="$eval_dir/compare_vs_step2000.json"
monitor_log="$eval_dir/monitor.log"

mkdir -p "$eval_dir"

printf '[%s] monitor_start checkpoint=%s\n' "$(date --iso-8601=seconds)" "$checkpoint_path" | tee -a "$monitor_log"
while [[ ! -f "$checkpoint_path" ]]; do
  printf '[%s] waiting checkpoint-step-2100.pt\n' "$(date --iso-8601=seconds)" | tee -a "$monitor_log"
  sleep "$interval_seconds"
done

printf '[%s] checkpoint_ready %s\n' "$(date --iso-8601=seconds)" "$checkpoint_path" | tee -a "$monitor_log"
cd /home/xingrui/lueq/SpikLLM_OPD/SpikLLM
python scripts/eval_wikitext_ppl_checkpoint.py \
  --checkpoint "$checkpoint_path" \
  --device cuda \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --block-size 2048 \
  --output "$eval_json" | tee -a "$monitor_log"

python - "$baseline_json" "$eval_json" "$compare_json" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
current = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
step2000 = baseline['runs']['loop14_offline_step2000']
payload = {
    'baseline_checkpoint_step': step2000['checkpoint_step'],
    'baseline_ppl': step2000['ppl'],
    'baseline_avg_nll': step2000['avg_nll'],
    'current_checkpoint_step': current['checkpoint_step'],
    'current_ppl': current['ppl'],
    'current_avg_nll': current['avg_nll'],
    'delta_ppl': current['ppl'] - step2000['ppl'],
    'delta_avg_nll': current['avg_nll'] - step2000['avg_nll'],
    'improved': current['ppl'] < step2000['ppl'],
    'baseline_source': step2000['checkpoint'],
    'current_source': current['checkpoint'],
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '[%s] compare_ready %s\n' "$(date --iso-8601=seconds)" "$compare_json" | tee -a "$monitor_log"
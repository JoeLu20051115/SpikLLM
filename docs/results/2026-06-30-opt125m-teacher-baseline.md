# OPT-125M Teacher Baseline Results

Date: 2026-06-30

Model: `facebook/opt-125m`

Purpose: record the current OPT-125M teacher zero-shot accuracy used as the reference baseline before BiSpikCLM SpAD training.

## Environment

| Item | Value |
|---|---|
| Commit | `719609c` |
| Python env | `SpikLLM/.venv` |
| `torch` | `2.12.1+cu130` |
| `transformers` | `4.57.6` |
| `datasets` | `2.21.0` |
| GPU | `3 x NVIDIA H200 NVL` |
| Evaluator | `bispikclm.train.eval_lm` zero-shot multiple-choice log-likelihood |

## Commands

Full evaluation:

```bash
.venv/bin/python -m bispikclm.train.eval_lm \
  --zero-shot \
  --model facebook/opt-125m \
  --device cuda
```

Limit-100 precheck:

```bash
.venv/bin/python -m bispikclm.train.eval_lm \
  --zero-shot \
  --model facebook/opt-125m \
  --limit 100 \
  --device cuda
```

Limit-1 smoke:

```bash
.venv/bin/python -m bispikclm.train.eval_lm \
  --zero-shot \
  --model facebook/opt-125m \
  --limit 1 \
  --device cpu
```

## Summary

| Setting | ARC-e | ARC-c | WG | BQ | PIQA | HS | OBQA | HQA | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper OPT 0.125B | 43.60 | 19.30 | 52.30 | 54.60 | 62.40 | 32.10 | 20.20 | 23.70 | 38.60 |
| Current full eval | 43.86 | 18.73 | 50.36 | 54.98 | 62.95 | 28.44 | 12.40 | 22.76 | 36.81 |
| Current limit=100 | 40.00 | 24.00 | 59.00 | 62.00 | 65.00 | 36.00 | 15.00 | 30.00 | 41.38 |
| Current limit=1 smoke | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 100.00 | 37.50 |

## Full Evaluation Counts

| Task | Correct | Total | Accuracy |
|---|---:|---:|---:|
| ARC-e | 250 | 570 | 43.86 |
| ARC-c | 56 | 299 | 18.73 |
| WinoGrande | 638 | 1267 | 50.36 |
| BoolQ | 1798 | 3270 | 54.98 |
| PIQA | 1157 | 1838 | 62.95 |
| HellaSwag | 2856 | 10042 | 28.44 |
| OpenBookQA | 62 | 500 | 12.40 |
| HeadQA | 624 | 2742 | 22.76 |

Macro average: 36.81

Micro average: 36.25

## Notes

- The full evaluation is the only current run suitable for baseline comparison; `limit=100` and `limit=1` are monitoring/smoke runs.
- Differences from the paper are recorded as measured environment/evaluator variance, not treated as a training blocker.
- Future BiSpikCLM checkpoints should be appended here using the same task order and evaluator command so trends are comparable.

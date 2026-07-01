# BiSpikCLM Loop Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable paper-faithful probe loop, use it to diagnose hard/soft loss stagnation, and continue only a probe that passes the approved C acceptance rule.

**Architecture:** Keep the training path in `bispikclm/train/train_spad.py`, add small repo-local scripts for W&B history analysis and probe launch hygiene, and make one root-cause code change per iteration. Probe acceptance is computed from local W&B history, not visual inspection.

**Tech Stack:** Python 3.12, PyTorch, transformers, datasets, torchrun/DDP, W&B local datastore, shell scripts.

## Global Constraints

- Teacher: OPT-family causal LM, starting with `facebook/opt-125m`.
- Data: FineWeb-Edu streaming corpus.
- Student: BiSpikCLM causal LM path, not an ANN substitute.
- Loss weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`.
- Distillation temperature: `2.0`.
- Optimizer/schedule defaults: Adam, cosine schedule, warmup ratio `0.2`, gradient clip `0.7`.
- Do not train on a fixed batch for probe acceptance.
- Do not leak labels into logits.
- Do not replace student logits with teacher logits.
- Do not disable or down-weight hard loss or soft loss.
- Do not change the paper loss weights to make curves look better.
- Acceptance rule C: pass if hard and soft losses fall below 5 within 2000 optimizer steps; otherwise allow one extra 2000-step window only if both losses show strong downward trends.

---

### Task 1: Add Local W&B Probe Analysis

**Files:**
- Create: `scripts/analyze_wandb_probe.py`
- Test: `tests/smoke/test_probe_analysis.py`

**Interfaces:**
- Consumes: a local W&B `.wandb` datastore file path.
- Produces: `analyze_rows(rows: list[dict[str, float]], window: int = 500) -> dict[str, object]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/smoke/test_probe_analysis.py`:

```python
from scripts.analyze_wandb_probe import analyze_rows


def _row(step: int, hard: float, soft: float) -> dict[str, float]:
    return {
        "train/step": float(step),
        "loss/hard_loss": hard,
        "loss/soft_loss": soft,
        "loss/total_loss": (hard + soft) / 2,
        "train/lr": 1e-4,
    }


def test_probe_analysis_passes_when_hard_and_soft_below_five() -> None:
    rows = [_row(step, 4.0, 4.5) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "pass"
    assert result["last_step"] == 2000


def test_probe_analysis_extends_when_trend_is_strongly_down() -> None:
    rows = [_row(step, 12.0 - step * 0.003, 11.0 - step * 0.003) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "extend"
    assert result["hard_slope_per_100"] < -0.2
    assert result["soft_slope_per_100"] < -0.2


def test_probe_analysis_fails_high_oscillation() -> None:
    rows = [_row(step, 14.0 + (step % 3), 12.0 + (step % 2)) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "fail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_probe_analysis.py -q`

Expected: FAIL because `scripts.analyze_wandb_probe` does not exist.

- [ ] **Step 3: Add the analyzer**

Create `scripts/analyze_wandb_probe.py`:

```python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _slope_per_100(rows: list[dict[str, float]], key: str) -> float:
    points = [(float(row["train/step"]), float(row[key])) for row in rows if key in row]
    if len(points) < 2:
        return 0.0
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0.0:
        return 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    return slope * 100.0


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows if key in row)


def analyze_rows(rows: list[dict[str, float]], window: int = 500) -> dict[str, Any]:
    if not rows:
        return {"decision": "fail", "reason": "no_rows"}
    ordered = sorted(rows, key=lambda row: float(row["train/step"]))
    last = ordered[-1]
    early = ordered[: min(window, len(ordered))]
    recent = ordered[-min(window, len(ordered)) :]
    hard_last = float(last["loss/hard_loss"])
    soft_last = float(last["loss/soft_loss"])
    hard_slope = _slope_per_100(recent, "loss/hard_loss")
    soft_slope = _slope_per_100(recent, "loss/soft_loss")
    hard_recent_mean = _mean(recent, "loss/hard_loss")
    soft_recent_mean = _mean(recent, "loss/soft_loss")
    hard_early_mean = _mean(early, "loss/hard_loss")
    soft_early_mean = _mean(early, "loss/soft_loss")
    if hard_last < 5.0 and soft_last < 5.0:
        decision = "pass"
        reason = "below_five"
    elif (
        hard_slope < -0.2
        and soft_slope < -0.2
        and hard_recent_mean < hard_early_mean * 0.85
        and soft_recent_mean < soft_early_mean * 0.85
    ):
        decision = "extend"
        reason = "strong_downward_trend"
    else:
        decision = "fail"
        reason = "insufficient_hard_soft_descent"
    return {
        "decision": decision,
        "reason": reason,
        "last_step": int(float(last["train/step"])),
        "hard_last": hard_last,
        "soft_last": soft_last,
        "hard_recent_mean": hard_recent_mean,
        "soft_recent_mean": soft_recent_mean,
        "hard_early_mean": hard_early_mean,
        "soft_early_mean": soft_early_mean,
        "hard_slope_per_100": hard_slope,
        "soft_slope_per_100": soft_slope,
    }


def _key_of(item: Any) -> str:
    nested = list(item.nested_key)
    return "/".join(nested) if nested else getattr(item, "key", "")


def _value_of(item: Any) -> float | str:
    try:
        return json.loads(item.value_json)
    except Exception:
        return item.value_json


def load_wandb_rows(path: Path) -> list[dict[str, float]]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    datastore = DataStore()
    datastore.open_for_scan(str(path))
    rows: list[dict[str, float]] = []
    while True:
        data = datastore.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history" or not record.history.item:
            continue
        row: dict[str, float] = {}
        for item in record.history.item:
            key = _key_of(item)
            if key:
                value = _value_of(item)
                if isinstance(value, (int, float)):
                    row[key] = float(value)
        if "train/step" in row and "loss/hard_loss" in row and "loss/soft_loss" in row:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a local W&B BiSpikCLM probe run.")
    parser.add_argument("wandb_file", type=Path)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(analyze_rows(load_wandb_rows(args.wandb_file), window=args.window), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_probe_analysis.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_wandb_probe.py tests/smoke/test_probe_analysis.py
git commit -m "test: add bispikclm probe analyzer"
```

### Task 2: Add Training Diagnostics Without Changing Loss Semantics

**Files:**
- Modify: `bispikclm/train/train_spad.py`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `student_outputs["logits"]`, `labels`, `attention_mask`, and `losses["hard_loss"]`.
- Produces: extra monitoring keys from `compute_lm_monitoring_metrics(...)`: `train/logit_mean`, `train/logit_std`, `train/logit_abs_max`, `train/valid_tokens`.

- [ ] **Step 1: Add a failing test**

Append this test to `tests/smoke/test_scaffold.py`:

```python
def test_lm_monitoring_metrics_include_logit_scale_and_valid_tokens() -> None:
    import torch

    from bispikclm.train.train_spad import compute_lm_monitoring_metrics

    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, -100]])
    attention_mask = torch.tensor([[1, 1, 0]])
    hard_loss = torch.tensor(2.0)

    metrics = compute_lm_monitoring_metrics(logits, labels, attention_mask, hard_loss)

    assert metrics["train/valid_tokens"] == 1.0
    assert metrics["train/logit_abs_max"] == 5.0
    assert "train/logit_mean" in metrics
    assert "train/logit_std" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_scaffold.py::test_lm_monitoring_metrics_include_logit_scale_and_valid_tokens -q`

Expected: FAIL because the new metric keys are missing.

- [ ] **Step 3: Implement diagnostics**

In `compute_lm_monitoring_metrics(...)`, after `shift_logits` and `valid` are computed, add valid-token logit statistics:

```python
        valid_logits = shift_logits[valid] if valid.any() else shift_logits.reshape(-1, shift_logits.size(-1))[:0]
        if valid_logits.numel():
            logit_mean = valid_logits.detach().float().mean()
            logit_std = valid_logits.detach().float().std(unbiased=False)
            logit_abs_max = valid_logits.detach().float().abs().max()
            valid_tokens = valid.sum().detach().float()
        else:
            logit_mean = hard_loss.new_zeros(())
            logit_std = hard_loss.new_zeros(())
            logit_abs_max = hard_loss.new_zeros(())
            valid_tokens = hard_loss.new_zeros(())
```

Return these keys in the metrics dict:

```python
            "train/logit_mean": float(logit_mean.detach().cpu()),
            "train/logit_std": float(logit_std.detach().cpu()),
            "train/logit_abs_max": float(logit_abs_max.detach().cpu()),
            "train/valid_tokens": float(valid_tokens.detach().cpu()),
```

- [ ] **Step 4: Run focused test**

Run: `pytest tests/smoke/test_scaffold.py::test_lm_monitoring_metrics_include_logit_scale_and_valid_tokens -q`

Expected: PASS.

- [ ] **Step 5: Run smoke tests**

Run: `pytest tests/smoke -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bispikclm/train/train_spad.py tests/smoke/test_scaffold.py
git commit -m "feat: log bispikclm output diagnostics"
```

### Task 3: Run First Paper-Constrained Probe

**Files:**
- Create: `docs/results/2026-06-30-loop-iteration-1.md`

**Interfaces:**
- Consumes: committed code from Tasks 1 and 2.
- Produces: W&B run, local checkpoint directory, and probe decision from `scripts/analyze_wandb_probe.py`.

- [ ] **Step 1: Verify GPUs are free**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: no stale BiSpikCLM process using the three H200 cards.

- [ ] **Step 2: Launch the closest fitting paper-constrained probe**

Use `batch_size=8`, `gradient_accumulation_steps=32`, `sequence_length=1024`, `world_size=3` because the documented `sequence_length=2048` and larger batch attempts OOMed on this hardware.

Run:

```bash
RUN_NAME="loop1-opt125m-seq1024-bs8-ga32-3xh200-$(date +%Y%m%d-%H%M%S)"
mkdir -p logs
setsid env PATH="$PWD/.venv/bin:$PATH" WANDB_PROJECT=bispikclm WANDB_RUN_NAME="$RUN_NAME" \
  torchrun --nproc_per_node=3 --master_port=29641 -m bispikclm.train.train_spad \
    --config configs/bispikclm_opt125m_spad.toml \
    --output-dir "output/$RUN_NAME" \
    --sequence-length 1024 \
    --precision bf16 \
    --batch-size 8 \
    --gradient-accumulation-steps 32 \
    --wandb \
    --wandb-project bispikclm \
    --wandb-run-name "$RUN_NAME" \
    --train \
  > "logs/$RUN_NAME.log" 2>&1 &
echo "$RUN_NAME"
```

Expected: parent `torchrun` and three child ranks stay alive after the first optimizer step.

- [ ] **Step 3: Monitor to 2000 optimizer steps or earlier failure**

Poll process and W&B history:

```bash
ps -ef | grep "$RUN_NAME" | grep -v grep
```

Find the local W&B file:

```bash
find wandb -path "*run-*.wandb" -newer "logs/$RUN_NAME.log" -print | tail -1
```

Analyze:

```bash
python scripts/analyze_wandb_probe.py wandb/<run-dir>/run-<id>.wandb
```

Expected: JSON decision is `pass`, `extend`, or `fail`.

- [ ] **Step 4: Record result**

Create `docs/results/2026-06-30-loop-iteration-1.md` with:

```markdown
# 2026-06-30 BiSpikCLM Loop Iteration 1

## Hypothesis

The previous GA=1 run failed mainly because it was not paper-constrained and had too much output-loss noise.

## Change

No semantic training change. Added analyzer and output diagnostics only.

## Probe

- Run name:
- W&B URL:
- Command:
- Local W&B file:
- Output dir:

## Decision

Paste analyzer JSON here.
```

- [ ] **Step 5: Commit result if probe completes**

```bash
git add docs/results/2026-06-30-loop-iteration-1.md
git commit -m "docs: record bispikclm loop iteration 1"
```

### Task 4: First Root-Cause Fix If Iteration 1 Fails

**Files:**
- Modify based on evidence:
  - `bispikclm/models/bispik_lm.py`
  - `bispikclm/models/bispik_model.py`
  - `bispikclm/train/train_spad.py`
- Test based on the touched file under `tests/smoke/`.

**Interfaces:**
- Consumes: Iteration 1 diagnostics, especially hard/soft loss, `train/logit_std`, `train/logit_abs_max`, and token accuracy.
- Produces: one minimal implementation change with tests and review before the next probe.

- [ ] **Step 1: State one hypothesis from diagnostics**

Write the hypothesis at the top of the next result file before changing code:

```markdown
# 2026-06-30 BiSpikCLM Loop Iteration 2

## Hypothesis

The evidence from iteration 1 shows `<specific failure>`, so this iteration tests `<specific minimal fix>`.
```

- [ ] **Step 2: Choose exactly one fix path**

Use this decision table:

```text
logit_std too high or abs_max exploding -> fix output scaling / normalization before lm_head
logit_std too low and token_accuracy near zero -> fix hidden-to-logit signal path
valid_tokens wrong -> fix labels / mask / shift plumbing
hard high but soft falling -> inspect hard-label CE path only
soft high but hard falling -> inspect teacher/student KL alignment only
```

- [ ] **Step 3: Write one focused failing test**

Add the smallest test that proves the chosen hypothesis. For output scale fixes, use a direct `BiSpikForCausalLM` forward test that asserts logits are finite and diagnostically bounded for a tiny config.

- [ ] **Step 4: Implement the smallest fix**

Modify only the file needed for the chosen hypothesis. Do not change SpAD weights, temperature, warmup ratio, gradient clip, or data source.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/smoke -q
```

Expected: PASS.

- [ ] **Step 6: Request code review**

Review must check paper faithfulness, no label leakage, no teacher-logit substitution, and no loss-weight changes.

- [ ] **Step 7: Launch next 2000-step probe**

Before launching a long probe, run a matched small-batch A/B gate against the current best baseline. The baseline and candidate must use the same GPU count, sequence length, time steps, per-GPU batch size, gradient accumulation, precision, and max optimizer steps. The candidate may launch a long probe only if it clearly improves the hard/soft early trajectory and is supported by token accuracy, teacher top-1 agreement, target-rank, and target-margin metrics. If the candidate is flat, worse, or only improves representation losses while output metrics lag the baseline, stop and revert it.

Use the long-probe launch shape from Task 3 only after the small-batch gate passes, unless the failure was OOM or launch-specific.

- [ ] **Step 8: Keep or revert**

Keep the change only if analyzer decision improves to `pass` or `extend`, or if hard/soft window means materially improve without violating constraints. Otherwise revert only the iteration change and form the next hypothesis.

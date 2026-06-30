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

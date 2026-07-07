from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from bispikclm.models.bispik_config import BiSpikConfig
from bispikclm.models.bispik_lm import BiSpikForCausalLM
from bispikclm.train.eval_lm import EVAL_TASKS


def load_checkpoint_model(checkpoint_path: Path, device: str) -> tuple[BiSpikForCausalLM, AutoTokenizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = BiSpikConfig(**checkpoint["config"]["model"])
    model = BiSpikForCausalLM(config)
    model.load_state_dict(checkpoint["student"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "model_config": asdict(config),
    }
    return model, tokenizer, metadata


def encode_scored_sequence(tokenizer, prompt: str, choice: str, max_length: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + choice_ids
    labels = [-100] * len(prompt_ids) + choice_ids
    if len(full_ids) > max_length:
        trim = len(full_ids) - max_length
        full_ids = full_ids[trim:]
        labels = labels[trim:]
    return full_ids, labels


def score_choices(
    model,
    tokenizer,
    prompt: str,
    choices: list[str],
    max_length: int,
    device: str,
    score_normalization: str,
) -> list[float]:
    encoded = [encode_scored_sequence(tokenizer, prompt, choice, max_length) for choice in choices]
    batch_max = max(len(input_ids) for input_ids, _ in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_rows = []
    label_rows = []
    mask_rows = []
    for input_ids, labels in encoded:
        pad = batch_max - len(input_ids)
        input_rows.append(input_ids + [pad_id] * pad)
        label_rows.append(labels + [-100] * pad)
        mask_rows.append([1] * len(input_ids) + [0] * pad)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    labels = torch.tensor(label_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    if input_ids.shape[1] < 2:
        return [float("-inf")] * len(choices)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(target.shape)
    scores = []
    for row_loss, row_mask in zip(losses, mask, strict=True):
        if not bool(row_mask.any()):
            scores.append(float("-inf"))
        else:
            selected = row_loss.masked_select(row_mask)
            score = selected.mean() if score_normalization == "mean" else selected.sum()
            scores.append(-float(score.detach().cpu()))
    return scores


def load_task_dataset(task_name: str):
    task = EVAL_TASKS[task_name]
    try:
        return (
            load_dataset(task.dataset, task.config, split=task.split, trust_remote_code=True)
            if task.config
            else load_dataset(task.dataset, split=task.split, trust_remote_code=True)
        )
    except TypeError:
        cache_dir = tempfile.mkdtemp(prefix=f"bispikclm-{task.name}-")
        return (
            load_dataset(task.dataset, task.config, split=task.split, trust_remote_code=True, cache_dir=cache_dir)
            if task.config
            else load_dataset(task.dataset, split=task.split, trust_remote_code=True, cache_dir=cache_dir)
        )


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    limit: int | None,
    device: str,
    progress_every: int,
    score_normalization: str,
) -> dict[str, object]:
    task = EVAL_TASKS[task_name]
    dataset = load_task_dataset(task_name)
    max_length = int(model.config.max_position_embeddings)
    correct = 0
    total = 0
    for row in dataset:
        prompt, choices, answer = task.formatter(row)
        if not choices or answer < 0 or answer >= len(choices):
            continue
        scores = score_choices(model, tokenizer, prompt, choices, max_length, device, score_normalization)
        correct += int(max(range(len(scores)), key=scores.__getitem__) == answer)
        total += 1
        if progress_every > 0 and total % progress_every == 0:
            print(json.dumps({"task": task_name, "correct": correct, "total": total, "accuracy": correct / total}), flush=True)
        if limit is not None and total >= limit:
            break
    return {"name": task_name, "correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def parse_score_normalization_by_task(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"score normalization override {item!r} must use task=sum|mean")
        task_name, mode = item.split("=", 1)
        task_name = task_name.strip()
        mode = mode.strip()
        if task_name not in EVAL_TASKS:
            raise ValueError(f"unknown score normalization task override {task_name!r}")
        if mode not in {"sum", "mean"}:
            raise ValueError(f"score normalization for {task_name!r} must be sum or mean")
        mapping[task_name] = mode
    return mapping


def write_output(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BiSpikCLM checkpoint on zero-shot MC tasks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task names from eval_lm.EVAL_TASKS.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--score-normalization", choices=("sum", "mean"), default="sum")
    parser.add_argument("--score-normalization-by-task", default=None)
    args = parser.parse_args()

    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown = sorted(set(task_names) - set(EVAL_TASKS))
    if unknown:
        raise ValueError(f"Unknown eval tasks: {unknown}")

    model, tokenizer, metadata = load_checkpoint_model(Path(args.checkpoint), args.device)
    score_normalization_by_task = parse_score_normalization_by_task(args.score_normalization_by_task)
    results = []
    output_path = Path(args.output)
    for task_name in task_names:
        task_score_normalization = score_normalization_by_task.get(task_name, args.score_normalization)
        result = evaluate_task(
            model,
            tokenizer,
            task_name,
            args.limit,
            args.device,
            args.progress_every,
            task_score_normalization,
        )
        results.append(result)
        total_correct = sum(int(item["correct"]) for item in results)
        total = sum(int(item["total"]) for item in results)
        payload = {
            **metadata,
            "tasks": results,
            "score_normalization": args.score_normalization,
            "score_normalization_by_task": score_normalization_by_task,
            "macro_accuracy": sum(float(item["accuracy"]) for item in results) / len(results),
            "micro_accuracy": total_correct / total if total else 0.0,
        }
        write_output(output_path, payload)
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

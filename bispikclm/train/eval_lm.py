import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Callable

from bispikclm.data.fineweb import dataset_smoke_check, dataset_summary, prepare_dataset_manifests
from bispikclm.distill.spad import SpADConfig, summarize_plan

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass(frozen=True, slots=True)
class EvalTask:
    name: str
    dataset: str
    config: str | None
    split: str
    formatter: Callable[[dict[str, object]], tuple[str, list[str], int]]


def _arc_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    choices = row["choices"]
    labels = list(choices["label"])
    texts = list(choices["text"])
    answer = str(row["answerKey"])
    prompt = f"Question: {row['question']}\nAnswer:"
    return prompt, [f" {text}" for text in texts], labels.index(answer)


def _winogrande_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    sentence = str(row["sentence"])
    prefix, suffix = sentence.split("_", maxsplit=1)
    prompt = prefix
    choices = [f"{row['option1']}{suffix}", f"{row['option2']}{suffix}"]
    return prompt, choices, int(row["answer"]) - 1


def _boolq_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    prompt = f"Passage: {row['passage']}\nQuestion: {row['question']}?\nAnswer:"
    return prompt, [" no", " yes"], 1 if bool(row["answer"]) else 0


def _piqa_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    prompt = f"Goal: {row['goal']}\nSolution:"
    return prompt, [f" {row['sol1']}", f" {row['sol2']}"], int(row["label"])


def _hellaswag_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    prompt = str(row.get("ctx", row.get("ctx_a", ""))).strip()
    return prompt, [f" {ending}" for ending in row["endings"]], int(row["label"])


def _openbookqa_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    choices = row["choices"]
    labels = list(choices["label"])
    texts = list(choices["text"])
    prompt = f"Question: {row['question_stem']}\nAnswer:"
    return prompt, [f" {text}" for text in texts], labels.index(str(row["answerKey"]))


def _headqa_formatter(row: dict[str, object]) -> tuple[str, list[str], int]:
    answers = row.get("answers", [])
    choices = [f" {answer['atext'] if isinstance(answer, dict) else answer}" for answer in answers]
    answer_index = int(row.get("ra", row.get("answer", 1))) - 1
    prompt = f"Question: {row.get('qtext', row.get('question', ''))}\nAnswer:"
    return prompt, choices, answer_index


EVAL_TASKS = {
    "arc_easy": EvalTask("arc_easy", "allenai/ai2_arc", "ARC-Easy", "validation", _arc_formatter),
    "arc_challenge": EvalTask("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", _arc_formatter),
    "winogrande": EvalTask("winogrande", "allenai/winogrande", "winogrande_xl", "validation", _winogrande_formatter),
    "boolq": EvalTask("boolq", "google/boolq", None, "validation", _boolq_formatter),
    "piqa": EvalTask("piqa", "ybisk/piqa", None, "validation", _piqa_formatter),
    "hellaswag": EvalTask("hellaswag", "Rowan/hellaswag", None, "validation", _hellaswag_formatter),
    "openbookqa": EvalTask("openbookqa", "allenai/openbookqa", "main", "validation", _openbookqa_formatter),
    "headqa": EvalTask("headqa", "dvilares/head_qa", "en", "test", _headqa_formatter),
}


def score_choice(model, tokenizer, prompt: str, choice: str, device: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prompt + choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    token_loss = F.cross_entropy(logits[mask], target[mask], reduction="sum")
    return -float(token_loss.detach().cpu())


def evaluate_task(model, tokenizer, task: EvalTask, limit: int | None, device: str) -> dict[str, object]:
    if load_dataset is None or torch is None or F is None:
        raise ImportError("datasets and torch are required for zero-shot evaluation")
    try:
        dataset = (
            load_dataset(task.dataset, task.config, split=task.split, trust_remote_code=True)
            if task.config
            else load_dataset(task.dataset, split=task.split, trust_remote_code=True)
        )
    except TypeError:
        cache_dir = tempfile.mkdtemp(prefix=f"bispikclm-{task.name}-")
        dataset = (
            load_dataset(task.dataset, task.config, split=task.split, trust_remote_code=True, cache_dir=cache_dir)
            if task.config
            else load_dataset(task.dataset, split=task.split, trust_remote_code=True, cache_dir=cache_dir)
        )
    correct = 0
    total = 0
    for row in dataset:
        prompt, choices, answer = task.formatter(row)
        if not choices or answer < 0 or answer >= len(choices):
            continue
        scores = [score_choice(model, tokenizer, prompt, choice, device) for choice in choices]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == answer)
        total += 1
        if limit is not None and total >= limit:
            break
    return {"name": task.name, "correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def evaluate_zero_shot(model_name: str, tasks: list[str], limit: int | None, device: str, progress: bool = False) -> dict[str, object]:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required for zero-shot evaluation")
    if torch is None:
        raise ImportError("torch is required for zero-shot evaluation")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    results = []
    for name in tasks:
        result = evaluate_task(model, tokenizer, EVAL_TASKS[name], limit, device)
        results.append(result)
        if progress:
            print(json.dumps(result, sort_keys=True), flush=True)
    total_correct = sum(int(result["correct"]) for result in results)
    total = sum(int(result["total"]) for result in results)
    return {
        "model": model_name,
        "tasks": results,
        "macro_accuracy": sum(float(result["accuracy"]) for result in results) / len(results) if results else 0.0,
        "micro_accuracy": total_correct / total if total else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LM evaluation entrypoint for BiSpikCLM dry runs.")
    parser.add_argument("--smoke-datasets", action="store_true", help="Validate dataset registry and manifest generation.")
    parser.add_argument("--zero-shot", action="store_true", help="Run zero-shot multiple-choice accuracy.")
    parser.add_argument("--model", default="facebook/opt-125m", help="Teacher or causal-LM checkpoint to evaluate.")
    parser.add_argument("--tasks", default=",".join(EVAL_TASKS), help="Comma-separated task names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional examples per task for smoke monitoring.")
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Optional JSON path for evaluation results.")
    parser.add_argument("--progress", action="store_true", help="Print task-level JSON progress.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_datasets:
        manifest_dir = prepare_dataset_manifests()
        payload = {
            "summary": dataset_smoke_check(),
            "datasets": dataset_summary(),
            "manifest_dir": str(manifest_dir),
            "distillation": summarize_plan(SpADConfig()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.zero_shot:
        task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
        unknown = sorted(set(task_names) - set(EVAL_TASKS))
        if unknown:
            raise ValueError(f"Unknown eval tasks: {unknown}")
        result = evaluate_zero_shot(args.model, task_names, args.limit, args.device, progress=args.progress)
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

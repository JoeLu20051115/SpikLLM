from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_reference_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    if not match:
        raise ValueError(f"could not find GSM8K final answer marker in: {answer!r}")
    return normalize_number(match.group(1))


def extract_generated_answer(text: str) -> str | None:
    marked = ANSWER_RE.search(text)
    if marked:
        return normalize_number(marked.group(1))
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def normalize_number(value: str) -> str:
    value = value.replace(",", "").strip()
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def convert_rows(dataset: Dataset, split_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(dataset):
        answer = str(row["answer"])
        rows.append(
            {
                "id": f"{split_name}-{idx:05d}",
                "source_split": split_name,
                "question": str(row["question"]),
                "answer": answer,
                "final_answer": extract_reference_answer(answer),
            }
        )
    return rows


def prepare_splits(
    dataset_name: str,
    dataset_config: str,
    validation_size: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    raw = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
    train = raw["train"].shuffle(seed=seed)
    if validation_size <= 0 or validation_size >= len(train):
        raise ValueError(f"validation_size must be in [1, {len(train) - 1}], got {validation_size}")

    validation_ds = train.select(range(validation_size))
    train_ds = train.select(range(validation_size, len(train)))
    test_ds = raw["test"]

    counts = {
        "train": write_jsonl(output_dir / "train.jsonl", convert_rows(train_ds, "train")),
        "validation": write_jsonl(output_dir / "validation.jsonl", convert_rows(validation_ds, "validation")),
        "test": write_jsonl(output_dir / "test.jsonl", convert_rows(test_ds, "test")),
    }
    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "source_splits": list(raw.keys()),
        "split_policy": "shuffle official train with fixed seed; first validation_size examples become validation",
        "seed": seed,
        "validation_size": validation_size,
        "counts": counts,
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "test": str(output_dir / "test.jsonl"),
        },
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def pad_batch(
    tokenizer: AutoTokenizer,
    examples: list[tuple[list[int], list[int]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must have a pad token id")
    max_len = max(len(input_ids) for input_ids, _ in examples)
    input_batch = []
    label_batch = []
    mask_batch = []
    for input_ids, labels in examples:
        pad_len = max_len - len(input_ids)
        input_batch.append(input_ids + [pad_id] * pad_len)
        label_batch.append(labels + [-100] * pad_len)
        mask_batch.append([1] * len(input_ids) + [0] * pad_len)
    return (
        torch.tensor(input_batch, dtype=torch.long, device=device),
        torch.tensor(label_batch, dtype=torch.long, device=device),
        torch.tensor(mask_batch, dtype=torch.long, device=device),
    )


def build_scoring_example(
    tokenizer: AutoTokenizer,
    row: dict[str, object],
    max_length: int,
) -> tuple[list[int], list[int], int]:
    prompt_ids = tokenizer(make_prompt(str(row["question"])), add_special_tokens=False)["input_ids"]
    answer_text = " " + str(row["answer"])
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    scored = sum(label != -100 for label in labels[1:])
    return input_ids, labels, scored


def evaluate_answer_likelihood(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, object]:
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_top5 = 0
    skipped = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            examples = []
            for row in batch_rows:
                input_ids, labels, scored = build_scoring_example(tokenizer, row, max_length)
                if scored == 0:
                    skipped += 1
                    continue
                examples.append((input_ids, labels))
            if not examples:
                continue
            input_ids, labels, attention_mask = pad_batch(tokenizer, examples, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(loss.detach().cpu())
            total_tokens += int(valid.sum().detach().cpu())
            predictions = logits.argmax(dim=-1)
            total_correct += int(predictions.eq(shift_labels).masked_select(valid).sum().detach().cpu())
            top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            total_top5 += int(top5.eq(shift_labels.unsqueeze(-1)).any(dim=-1).masked_select(valid).sum().detach().cpu())

    avg_nll = total_nll / max(total_tokens, 1)
    return {
        "answer_nll": total_nll,
        "answer_tokens": total_tokens,
        "answer_avg_nll": avg_nll,
        "answer_ppl": float(torch.exp(torch.tensor(avg_nll)).item()),
        "answer_token_accuracy": total_correct / max(total_tokens, 1),
        "answer_top5_accuracy": total_top5 / max(total_tokens, 1),
        "skipped_examples": skipped,
    }


def evaluate_greedy_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    limit: int,
) -> dict[str, object]:
    if limit == 0:
        return {"enabled": False}
    eval_rows = rows if limit < 0 else rows[:limit]
    correct = 0
    examples = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(eval_rows), batch_size):
                batch_rows = eval_rows[start : start + batch_size]
                prompts = [make_prompt(str(row["question"])) for row in batch_rows]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                prompt_len = encoded["input_ids"].shape[1]
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
                for row, text in zip(batch_rows, generated_texts, strict=True):
                    predicted = extract_generated_answer(text)
                    reference = str(row["final_answer"])
                    is_correct = predicted == reference
                    correct += int(is_correct)
                    if len(examples) < 8:
                        examples.append(
                            {
                                "id": row["id"],
                                "reference": reference,
                                "predicted": predicted,
                                "correct": is_correct,
                                "generation": text,
                            }
                        )
    finally:
        tokenizer.padding_side = old_padding_side
    return {
        "enabled": True,
        "examples": len(eval_rows),
        "exact_match": correct / max(len(eval_rows), 1),
        "correct": correct,
        "max_new_tokens": max_new_tokens,
        "sample_predictions": examples,
    }


def load_teacher(model_name: str, device: torch.device, dtype_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = None
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, str(dtype or "auto")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare GSM8K splits and evaluate a teacher LM baseline.")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--output-dir", default="data/gsm8k-seed20260708")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--teacher-model", default="facebook/opt-125m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-limit", type=int, default=0, help="0 disables generation, -1 evaluates all rows")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--result-path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata = prepare_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        validation_size=args.validation_size,
        seed=args.seed,
        output_dir=output_dir,
    )

    device = torch.device(args.device)
    model, tokenizer, dtype_used = load_teacher(args.teacher_model, device, args.dtype)
    test_rows = load_jsonl(output_dir / "test.jsonl")
    likelihood = evaluate_answer_likelihood(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    generation = evaluate_greedy_generation(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows,
        device=device,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.generation_limit,
    )
    payload = {
        "dataset": metadata,
        "teacher": {
            "model": args.teacher_model,
            "device": str(device),
            "dtype": dtype_used,
            "tokenizer_use_fast": False,
        },
        "test_metrics": {
            "answer_likelihood": likelihood,
            "greedy_generation": generation,
        },
        "prompt_template": "Question: {question}\\nAnswer:",
        "likelihood_protocol": "teacher-forced scoring of gold GSM8K answer tokens only; prompt tokens are masked",
    }
    result_path = Path(args.result_path) if args.result_path else output_dir / "teacher_test_metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

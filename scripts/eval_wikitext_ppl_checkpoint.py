from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from bispikclm.models.bispik_config import BiSpikConfig
from bispikclm.models.bispik_lm import BiSpikForCausalLM


def load_checkpoint_model(checkpoint_path: Path, device: str) -> tuple[BiSpikForCausalLM, AutoTokenizer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = BiSpikConfig(**checkpoint["config"]["model"])
    model = BiSpikForCausalLM(config)
    model.load_state_dict(checkpoint["student"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "model_config": asdict(config),
    }
    return model, tokenizer, metadata


def load_wikitext_text(dataset_name: str, dataset_config: str, split: str, join_text: str) -> str:
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    texts = [str(row["text"]) for row in dataset]
    return join_text.join(texts)


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    block_size: int,
    join_text: str,
) -> dict[str, object]:
    model, tokenizer, metadata = load_checkpoint_model(checkpoint_path, device)
    raw_text = load_wikitext_text(dataset_name, dataset_config, split, join_text)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    token_ids = [bos_id] + tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    raw_tokens_including_bos = len(token_ids)
    used_tokens_including_bos = (raw_tokens_including_bos // block_size) * block_size
    token_ids = token_ids[:used_tokens_including_bos]
    chunks = used_tokens_including_bos // block_size
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(chunks, block_size)

    total_nll = 0.0
    total_correct = 0.0
    total_top5 = 0.0
    scored_tokens = 0

    with torch.no_grad():
        for chunk in input_ids.split(1):
            chunk = chunk.to(device)
            attention_mask = torch.ones_like(chunk)
            outputs = model(input_ids=chunk, attention_mask=attention_mask)
            logits = outputs["logits"][:, :-1, :]
            labels = chunk[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.detach().cpu())

            predictions = logits.argmax(dim=-1)
            total_correct += float(predictions.eq(labels).sum().detach().cpu())

            top5 = logits.topk(k=5, dim=-1).indices
            total_top5 += float(top5.eq(labels.unsqueeze(-1)).any(dim=-1).sum().detach().cpu())
            scored_tokens += int(labels.numel())

    avg_nll = total_nll / scored_tokens
    payload = {
        "block_size": block_size,
        "config": dataset_config,
        "dataset": dataset_name,
        "split": split,
        "join": join_text,
        "raw_tokens_including_bos": raw_tokens_including_bos,
        "used_tokens_including_bos": used_tokens_including_bos,
        "chunks": chunks,
        "scored_tokens": scored_tokens,
        **metadata,
        "nll": total_nll,
        "avg_nll": avg_nll,
        "ppl": float(torch.exp(torch.tensor(avg_nll)).item()),
        "token_accuracy": total_correct / scored_tokens,
        "top5_accuracy": total_top5 / scored_tokens,
        "tokenizer": metadata["model_config"]["teacher_model_id"],
        "tokenizer_use_fast": False,
        "evaluation": "SparseGPT-style raw WikiText2 non-overlap block evaluation",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BiSpik checkpoint on WikiText raw PPL.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--join-text", default="\n\n")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        block_size=args.block_size,
        join_text=args.join_text,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
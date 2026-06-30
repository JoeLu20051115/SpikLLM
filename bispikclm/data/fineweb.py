from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download

try:
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover
    IterableDataset = object


@dataclass(slots=True)
class TeacherSpec:
    repo_id: str
    allow_patterns: tuple[str, ...]


@dataclass(slots=True)
class DatasetSpec:
    repo_id: str
    subset: str | None = None
    repo_type: str = "dataset"


TEACHER_SPECS = (
    TeacherSpec("facebook/opt-125m", ("config.json", "tokenizer*", "*.json", "*.txt")),
    TeacherSpec("facebook/opt-350m", ("config.json", "tokenizer*", "*.json", "*.txt")),
    TeacherSpec("facebook/opt-1.3b", ("config.json", "tokenizer*", "*.json", "*.txt")),
)

DATASET_SPECS = (
    DatasetSpec("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    DatasetSpec("allenai/ai2_arc"),
    DatasetSpec("allenai/winogrande"),
    DatasetSpec("google/boolq"),
    DatasetSpec("ybisk/piqa"),
    DatasetSpec("Rowan/hellaswag"),
    DatasetSpec("allenai/openbookqa"),
    DatasetSpec("dvilares/head_qa"),
    DatasetSpec("Salesforce/wikitext"),
)


def dataset_smoke_check() -> dict[str, int]:
    return {"teacher_specs": len(TEACHER_SPECS), "dataset_specs": len(DATASET_SPECS)}


def download_teachers() -> list[str]:
    downloaded = []
    for spec in TEACHER_SPECS:
        snapshot_download(repo_id=spec.repo_id, allow_patterns=list(spec.allow_patterns))
        downloaded.append(spec.repo_id)
    return downloaded


def default_dataset_manifest_dir() -> Path:
    return Path(tempfile.gettempdir()) / "bispikclm-datasets"


def prepare_dataset_manifests(output_dir: str | Path | None = None) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else default_dataset_manifest_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    for spec in DATASET_SPECS:
        slug = spec.repo_id.replace("/", "--")
        manifest = {"repo_id": spec.repo_id, "subset": spec.subset, "repo_type": spec.repo_type}
        (target_dir / f"{slug}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target_dir


def dataset_summary() -> list[dict[str, str | None]]:
    return [asdict(spec) for spec in DATASET_SPECS]


class SequencePackingIterableDataset(IterableDataset):
    """Stream text rows into fixed-length causal-LM token blocks."""

    def __init__(
        self,
        rows,
        tokenizer,
        sequence_length: int,
        text_field: str = "text",
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.rows = rows
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.text_field = text_field
        self.add_eos = add_eos

    def __iter__(self):
        import torch

        buffer: list[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        for row in self.rows:
            text = row.get(self.text_field, "") if isinstance(row, dict) else getattr(row, self.text_field, "")
            if not text:
                continue
            encoded = self.tokenizer(text, add_special_tokens=False)
            token_ids = list(encoded["input_ids"])
            if self.add_eos and eos_token_id is not None:
                token_ids.append(int(eos_token_id))
            buffer.extend(int(token_id) for token_id in token_ids)
            while len(buffer) >= self.sequence_length:
                chunk = buffer[: self.sequence_length]
                del buffer[: self.sequence_length]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }


def collate_packed_sequences(examples: list[dict[str, object]], pad_token_id: int = 1) -> dict[str, object]:
    import torch

    if not examples:
        raise ValueError("examples must not be empty")
    max_len = max(example["input_ids"].shape[0] for example in examples)
    batch = {
        "input_ids": torch.full((len(examples), max_len), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(examples), max_len), dtype=torch.long),
        "labels": torch.full((len(examples), max_len), -100, dtype=torch.long),
    }
    for row, example in enumerate(examples):
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        labels = example["labels"]
        length = input_ids.shape[0]
        batch["input_ids"][row, :length] = input_ids
        batch["attention_mask"][row, :length] = attention_mask
        batch["labels"][row, :length] = labels
    return batch


def build_fineweb_dataloader(
    tokenizer,
    sequence_length: int,
    batch_size: int,
    *,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
):
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from torch.utils.data import DataLoader

    rows = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    if world_size > 1:
        rows = split_dataset_by_node(rows, rank=rank, world_size=world_size)
    dataset = SequencePackingIterableDataset(rows, tokenizer=tokenizer, sequence_length=sequence_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate_packed_sequences, pad_token_id=getattr(tokenizer, "pad_token_id", 1) or 1),
    )

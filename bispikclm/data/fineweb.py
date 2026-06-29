from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download


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

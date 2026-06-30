import argparse
from dataclasses import asdict, dataclass
import json

from bispikclm.data.fineweb import download_teachers, prepare_dataset_manifests
from bispikclm.distill.spad import SpADConfig, summarize_plan
from bispikclm.models import BiSpikConfig

try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover - optional runtime dependency
    AutoModelForCausalLM = None

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


@dataclass(slots=True)
class TeacherRuntime:
    model_name: str
    available: bool
    backend: str
    teacher_model: str | None = None


def resolve_teacher_runtime(model_name: str) -> TeacherRuntime:
    if AutoModelForCausalLM is None:
        return TeacherRuntime(model_name=model_name, available=False, backend="unavailable")
    return TeacherRuntime(
        model_name=model_name,
        available=True,
        backend="transformers",
        teacher_model=AutoModelForCausalLM.__name__,
    )


def build_training_payload(config: BiSpikConfig, distill_config: SpADConfig) -> dict[str, object]:
    teacher_runtime = resolve_teacher_runtime(config.teacher_model_id)
    payload: dict[str, object] = {
        "student_config": asdict(config),
        "distillation": summarize_plan(distill_config),
        "teacher_runtime": asdict(teacher_runtime),
        "train_loop": {
            "student_model": "BiSpikForCausalLM",
            "optimizer": "torch.optim.AdamW",
            "loss": "spad_total_loss",
            "epoch": 1,
            "step": 1,
            "backward": True,
            "runtime_ready": torch is not None and AutoModelForCausalLM is not None,
        },
        "runtime_requirements": {
            "torch_available": torch is not None,
            "transformers_available": AutoModelForCausalLM is not None,
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPAD training entrypoint for BiSpikCLM prelaunch validation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training plan.")
    parser.add_argument("--download-teachers", action="store_true", help="Cache teacher metadata and tokenizer assets.")
    parser.add_argument("--prepare-datasets", action="store_true", help="Write dataset manifests for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BiSpikConfig()
    distill_config = SpADConfig()
    payload = {"plan": build_training_payload(config, distill_config)}
    if args.download_teachers:
        payload["teachers"] = download_teachers()
    if args.prepare_datasets:
        payload["dataset_manifest_dir"] = str(prepare_dataset_manifests())
    if args.dry_run or args.download_teachers or args.prepare_datasets:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

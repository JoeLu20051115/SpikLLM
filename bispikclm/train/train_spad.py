from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from bispikclm.data.fineweb import download_teachers, prepare_dataset_manifests
from bispikclm.distill.spad import SpADConfig, SpADProjector, compute_multilevel_distillation, summarize_plan
from bispikclm.models import BiSpikConfig, BiSpikForCausalLM

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from transformers import AutoConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
except ImportError:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None
    get_cosine_schedule_with_warmup = None


@dataclass(slots=True)
class TeacherRuntime:
    model_name: str
    available: bool
    backend: str
    teacher_model: str | None = None


@dataclass(slots=True)
class TrainingConfig:
    teacher_model: str = "facebook/opt-125m"
    output_dir: str = "output/v1-opt-sft"
    learning_rate: float = 5e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_steps: int = 1
    warmup_ratio: float = 0.2
    gradient_clip: float = 0.7
    sequence_length: int = 16
    time_steps: int = 2


def resolve_teacher_runtime(model_name: str) -> TeacherRuntime:
    if AutoModelForCausalLM is None:
        return TeacherRuntime(model_name=model_name, available=False, backend="unavailable")
    return TeacherRuntime(model_name=model_name, available=True, backend="transformers", teacher_model=AutoModelForCausalLM.__name__)


def build_training_payload(config: BiSpikConfig, distill_config: SpADConfig) -> dict[str, object]:
    teacher_runtime = resolve_teacher_runtime(config.teacher_model_id)
    return {
        "student_config": asdict(config),
        "distillation": summarize_plan(distill_config),
        "teacher_runtime": asdict(teacher_runtime),
        "train_loop": {
            "student_model": "BiSpikForCausalLM",
            "optimizer": "torch.optim.Adam",
            "scheduler": "cosine_decay",
            "loss": "lambda1*EA + lambda2*SAA + lambda3*SFA + lambda4*STA + lambda5*HTA",
            "backward": True,
            "runtime_ready": torch is not None and AutoModelForCausalLM is not None,
        },
        "runtime_requirements": {
            "torch_available": torch is not None,
            "transformers_available": AutoModelForCausalLM is not None,
        },
    }


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def load_teacher(model_name: str, device: torch.device) -> nn.Module:
    if AutoConfig is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required to load the OPT teacher")
    teacher_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, config=teacher_config)
    return freeze_teacher(teacher.to(device))


def build_student_from_teacher(teacher: nn.Module, train_config: TrainingConfig) -> tuple[BiSpikForCausalLM, SpADProjector, SpADProjector]:
    teacher_config = teacher.config
    student_config = BiSpikConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=min(getattr(teacher_config, "hidden_size", 768), 128),
        intermediate_size=min(getattr(teacher_config, "ffn_dim", 3072), 256),
        num_attention_heads=4,
        num_hidden_layers=min(getattr(teacher_config, "num_hidden_layers", 12), 2),
        max_position_embeddings=max(getattr(teacher_config, "max_position_embeddings", 2048), train_config.sequence_length),
        num_steps=train_config.time_steps,
        teacher_model_id=train_config.teacher_model,
    )
    student = BiSpikForCausalLM(student_config)
    teacher_dim = getattr(teacher_config, "hidden_size", student_config.hidden_size)
    return (
        student,
        SpADProjector(student_config.hidden_size, teacher_dim),
        SpADProjector(student_config.hidden_size, teacher_dim),
    )


def make_dummy_batch(vocab_size: int, batch_size: int, sequence_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(4, min(vocab_size, 128), (batch_size, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_dummy_batch(train_config: TrainingConfig, distill_config: SpADConfig | None = None) -> dict[str, float]:
    if torch is None:
        raise ImportError("torch is required for training")
    distill_config = distill_config or SpADConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(train_config.teacher_model, device)
    student, embedding_projector, hidden_projector = build_student_from_teacher(teacher, train_config)
    student = student.to(device)
    embedding_projector = embedding_projector.to(device)
    hidden_projector = hidden_projector.to(device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters()),
        lr=train_config.learning_rate,
    )
    warmup_steps = max(1, int(train_config.max_steps * train_config.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(train_config.max_steps, 1))

    batch = make_dummy_batch(teacher.config.vocab_size, min(train_config.batch_size, 2), train_config.sequence_length, device)
    with torch.no_grad():
        teacher_outputs_raw = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
    teacher_outputs = {
        "hidden_states": teacher_outputs_raw.hidden_states,
        "attentions": teacher_outputs_raw.attentions,
        "logits": teacher_outputs_raw.logits,
    }
    student_outputs = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        output_attentions=True,
        return_spike_stats=True,
    )
    losses = compute_multilevel_distillation(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        config=distill_config,
        labels=batch["labels"],
        embedding_projector=embedding_projector,
        hidden_projector=hidden_projector,
    )
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), train_config.gradient_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), "config": asdict(train_config)}, output_dir / "dummy-checkpoint.pt")
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BiSpikCLM with offline SpAD distillation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training plan.")
    parser.add_argument("--dummy-batch", action="store_true", help="Run one teacher/student SpAD backward step on a dummy batch.")
    parser.add_argument("--download-teachers", action="store_true", help="Cache teacher metadata and tokenizer assets.")
    parser.add_argument("--prepare-datasets", action="store_true", help="Write dataset manifests for smoke runs.")
    parser.add_argument("--teacher-model", default="facebook/opt-125m")
    parser.add_argument("--output-dir", default="output/v1-opt-sft")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = BiSpikConfig(teacher_model_id=args.teacher_model, num_steps=args.time_steps)
    distill_config = SpADConfig()
    if args.download_teachers:
        print({"teachers": download_teachers()})
    if args.prepare_datasets:
        print({"dataset_manifest_dir": str(prepare_dataset_manifests())})
    if args.dry_run:
        print(build_training_payload(model_config, distill_config))
    if args.dummy_batch:
        result = train_dummy_batch(
            TrainingConfig(
                teacher_model=args.teacher_model,
                output_dir=args.output_dir,
                sequence_length=args.sequence_length,
                time_steps=args.time_steps,
                max_steps=args.max_steps,
            ),
            distill_config,
        )
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

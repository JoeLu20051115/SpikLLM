from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from bispikclm.data.fineweb import build_fineweb_dataloader, download_teachers, prepare_dataset_manifests
from bispikclm.distill.spad import SpADConfig, SpADProjector, compute_multilevel_distillation, summarize_plan
from bispikclm.models import BiSpikConfig, BiSpikForCausalLM

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
except ImportError:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
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
    target_tokens: int | None = None
    warmup_ratio: float = 0.2
    gradient_clip: float = 0.7
    sequence_length: int = 2048
    time_steps: int = 2
    num_workers: int = 0
    checkpoint_interval: int = 1000
    precision: str = "bf16"
    use_wandb: bool = False
    wandb_project: str = "bispikclm"
    wandb_run_name: str | None = None
    log_interval: int = 1


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    split: str = "train"


@dataclass(slots=True)
class ExperimentConfig:
    model: BiSpikConfig
    distillation: SpADConfig
    training: TrainingConfig
    data: DataConfig


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    model = BiSpikConfig(**_section(raw, "model"))
    distillation = SpADConfig(**_section(raw, "distillation"))
    training = TrainingConfig(**_section(raw, "training"))
    data = DataConfig(**_section(raw, "data"))
    model.num_steps = training.time_steps
    model.teacher_model_id = training.teacher_model
    return ExperimentConfig(model=model, distillation=distillation, training=training, data=data)


def resolve_teacher_runtime(model_name: str) -> TeacherRuntime:
    if AutoModelForCausalLM is None:
        return TeacherRuntime(model_name=model_name, available=False, backend="unavailable")
    return TeacherRuntime(model_name=model_name, available=True, backend="transformers", teacher_model=AutoModelForCausalLM.__name__)


def resolve_max_steps(train_config: TrainingConfig, world_size: int = 1) -> int:
    if train_config.target_tokens is None:
        return train_config.max_steps
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * train_config.sequence_length
        * max(world_size, 1)
    )
    return max(1, math.ceil(train_config.target_tokens / tokens_per_step))


def build_training_payload(
    config: BiSpikConfig,
    distill_config: SpADConfig,
    train_config: TrainingConfig | None = None,
) -> dict[str, object]:
    teacher_runtime = resolve_teacher_runtime(config.teacher_model_id)
    resolved_steps = resolve_max_steps(train_config) if train_config is not None else None
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
            "target_tokens": train_config.target_tokens if train_config is not None else None,
            "resolved_max_steps": resolved_steps,
            "precision": train_config.precision if train_config is not None else None,
            "wandb": train_config.use_wandb if train_config is not None else None,
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


def build_student_config_from_teacher_config(
    teacher_config,
    train_config: TrainingConfig,
    model_config: BiSpikConfig | None = None,
) -> BiSpikConfig:
    base = model_config or BiSpikConfig()
    return BiSpikConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=getattr(teacher_config, "hidden_size", 768),
        intermediate_size=getattr(teacher_config, "ffn_dim", getattr(teacher_config, "intermediate_size", 3072)),
        num_attention_heads=getattr(teacher_config, "num_attention_heads", 12),
        num_hidden_layers=getattr(teacher_config, "num_hidden_layers", 12),
        max_position_embeddings=max(getattr(teacher_config, "max_position_embeddings", 2048), train_config.sequence_length),
        pad_token_id=getattr(teacher_config, "pad_token_id", 1) or 1,
        bos_token_id=getattr(teacher_config, "bos_token_id", 2) or 2,
        eos_token_id=getattr(teacher_config, "eos_token_id", 2) or 2,
        num_steps=train_config.time_steps,
        spike_threshold=base.spike_threshold,
        tau=base.tau,
        membrane_decay=base.membrane_decay,
        spike_surrogate=base.spike_surrogate,
        surrogate_alpha=base.surrogate_alpha,
        initializer_range=base.initializer_range,
        input_scale=base.input_scale,
        teacher_model_id=train_config.teacher_model,
    )


def build_student_from_teacher(
    teacher: nn.Module,
    train_config: TrainingConfig,
    model_config: BiSpikConfig | None = None,
) -> tuple[BiSpikForCausalLM, SpADProjector, SpADProjector]:
    teacher_config = teacher.config
    student_config = build_student_config_from_teacher_config(teacher_config, train_config, model_config)
    student = BiSpikForCausalLM(student_config)
    teacher_dim = getattr(teacher_config, "hidden_size", student_config.hidden_size)
    return (
        student,
        SpADProjector(student_config.hidden_size, teacher_dim),
        SpADProjector(student_config.hidden_size, teacher_dim),
    )


def compute_lm_monitoring_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    hard_loss: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid = shift_labels.ne(-100)
        if attention_mask is not None:
            valid = valid & attention_mask[..., 1:].to(dtype=torch.bool)
        if valid.any():
            detached_logits = shift_logits.detach()
            predictions = detached_logits.argmax(dim=-1)
            accuracy = predictions.eq(shift_labels).masked_select(valid).float().mean()
            logit_values = detached_logits.float()
            valid_float = valid.unsqueeze(-1).to(dtype=logit_values.dtype)
            value_count = valid.sum().to(dtype=logit_values.dtype) * logit_values.shape[-1]
            logit_sum = (logit_values * valid_float).sum()
            logit_mean = logit_sum / value_count.clamp_min(1.0)
            centered = (logit_values - logit_mean) * valid_float
            logit_std = (centered.square().sum() / value_count.clamp_min(1.0)).sqrt()
            logit_abs_max = logit_values.abs().masked_fill(~valid.unsqueeze(-1), 0.0).max()
            valid_tokens = valid.sum().detach().float()
        else:
            accuracy = hard_loss.new_zeros(())
            logit_mean = hard_loss.new_zeros(())
            logit_std = hard_loss.new_zeros(())
            logit_abs_max = hard_loss.new_zeros(())
            valid_tokens = hard_loss.new_zeros(())
        perplexity = torch.exp(hard_loss.detach().float().clamp(max=50.0))
        return {
            "train/token_accuracy": float(accuracy.detach().cpu()),
            "train/perplexity": float(perplexity.detach().cpu()),
            "train/logit_mean": float(logit_mean.detach().cpu()),
            "train/logit_std": float(logit_std.detach().cpu()),
            "train/logit_abs_max": float(logit_abs_max.detach().cpu()),
            "train/valid_tokens": float(valid_tokens.detach().cpu()),
        }


def average_loss_snapshots(snapshots: list[dict[str, float]]) -> dict[str, float]:
    if not snapshots:
        return {}
    names = snapshots[0].keys()
    return {name: sum(snapshot[name] for snapshot in snapshots) / len(snapshots) for name in names}


def _save_checkpoint(
    path: Path,
    student: nn.Module,
    embedding_projector: nn.Module,
    hidden_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: ExperimentConfig,
) -> None:
    def unwrap(module: nn.Module) -> nn.Module:
        return module.module if hasattr(module, "module") else module

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": unwrap(student).state_dict(),
            "embedding_projector": unwrap(embedding_projector).state_dict(),
            "hidden_projector": unwrap(hidden_projector).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": {
                "model": asdict(config.model),
                "distillation": asdict(config.distillation),
                "training": asdict(config.training),
                "data": asdict(config.data),
            },
        },
        path,
    )


def train(config: ExperimentConfig, resume_from: str | Path | None = None) -> dict[str, float]:
    if torch is None or AutoTokenizer is None or get_cosine_schedule_with_warmup is None:
        raise ImportError("torch and transformers are required for training")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    use_amp = device.type == "cuda" and config.training.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if config.training.precision == "bf16" else torch.float16
    wandb_run = None
    if local_rank == 0 and config.training.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is required when --wandb is enabled") from exc
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", config.training.wandb_project),
            name=os.environ.get("WANDB_RUN_NAME", config.training.wandb_run_name),
            config={
                "model": asdict(config.model),
                "distillation": asdict(config.distillation),
                "training": asdict(config.training),
                "data": asdict(config.data),
            },
        )

    teacher = load_teacher(config.training.teacher_model, device)
    student, embedding_projector, hidden_projector = build_student_from_teacher(teacher, config.training, config.model)
    config.model = student.config
    student = student.to(device)
    embedding_projector = embedding_projector.to(device)
    hidden_projector = hidden_projector.to(device)
    if distributed:
        ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
        student = torch.nn.parallel.DistributedDataParallel(student, **ddp_kwargs)
        embedding_projector = torch.nn.parallel.DistributedDataParallel(embedding_projector, **ddp_kwargs)
        hidden_projector = torch.nn.parallel.DistributedDataParallel(hidden_projector, **ddp_kwargs)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters()),
        lr=config.training.learning_rate,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config.training.max_steps = resolve_max_steps(config.training, world_size)
    warmup_steps = max(1, int(config.training.max_steps * config.training.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(config.training.max_steps, 1))
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        target_student = student.module if hasattr(student, "module") else student
        target_embedding_projector = embedding_projector.module if hasattr(embedding_projector, "module") else embedding_projector
        target_hidden_projector = hidden_projector.module if hasattr(hidden_projector, "module") else hidden_projector
        target_student.load_state_dict(checkpoint["student"])
        target_embedding_projector.load_state_dict(checkpoint["embedding_projector"])
        target_hidden_projector.load_state_dict(checkpoint["hidden_projector"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])

    tokenizer = AutoTokenizer.from_pretrained(config.training.teacher_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader = build_fineweb_dataloader(
        tokenizer,
        sequence_length=config.training.sequence_length,
        batch_size=config.training.batch_size,
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        num_workers=config.training.num_workers,
        rank=local_rank,
        world_size=world_size,
    )

    optimizer.zero_grad(set_to_none=True)
    last_losses: dict[str, float] = {}
    last_monitoring_metrics: dict[str, float] = {}
    accumulated_loss_snapshots: list[dict[str, float]] = []
    step = start_step
    output_dir = Path(config.training.output_dir)
    while step < config.training.max_steps:
        for batch_index, batch in enumerate(dataloader):
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
                    config=config.distillation,
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                    embedding_projector=embedding_projector,
                    hidden_projector=hidden_projector,
                    spike_threshold=config.model.spike_threshold,
                    membrane_decay=config.model.membrane_decay,
                )
            accumulated_loss_snapshots.append({name: float(value.detach().cpu()) for name, value in losses.items()})
            (losses["total_loss"] / config.training.gradient_accumulation_steps).backward()
            if (batch_index + 1) % config.training.gradient_accumulation_steps == 0:
                should_log = local_rank == 0 and wandb_run is not None and (step + 1) % config.training.log_interval == 0
                monitoring_metrics = (
                    compute_lm_monitoring_metrics(
                        logits=student_outputs["logits"],
                        labels=batch["labels"],
                        attention_mask=batch.get("attention_mask"),
                        hard_loss=losses["hard_loss"],
                    )
                    if should_log
                    else {}
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters()),
                    config.training.gradient_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last_losses = average_loss_snapshots(accumulated_loss_snapshots)
                accumulated_loss_snapshots = []
                last_monitoring_metrics = monitoring_metrics
                if should_log:
                    metrics = {
                        **{f"loss/{name}": value for name, value in last_losses.items()},
                        **last_monitoring_metrics,
                        "train/step": step,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                        "train/tokens_seen": step * config.training.batch_size * config.training.gradient_accumulation_steps * config.training.sequence_length * world_size,
                    }
                    if torch.cuda.is_available():
                        metrics["train/peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
                    wandb_run.log(metrics, step=step)
                if local_rank == 0 and step % config.training.checkpoint_interval == 0:
                    _save_checkpoint(output_dir / f"checkpoint-step-{step}.pt", student, embedding_projector, hidden_projector, optimizer, scheduler, step, config)
                if step >= config.training.max_steps:
                    break
    if local_rank == 0:
        _save_checkpoint(output_dir / "checkpoint-last.pt", student, embedding_projector, hidden_projector, optimizer, scheduler, step, config)
    if distributed:
        torch.distributed.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()
    return last_losses


def make_dummy_batch(vocab_size: int, batch_size: int, sequence_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(4, min(vocab_size, 128), (batch_size, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_dummy_batch(train_config: TrainingConfig, distill_config: SpADConfig | None = None) -> dict[str, float]:
    if torch is None:
        raise ImportError("torch is required for training")
    distill_config = distill_config or SpADConfig()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(train_config.teacher_model, device)
    student, embedding_projector, hidden_projector = build_student_from_teacher(teacher, train_config)
    student = student.to(device)
    embedding_projector = embedding_projector.to(device)
    hidden_projector = hidden_projector.to(device)
    trainable_parameters = list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters,
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
    snapshots: list[dict[str, float]] = []
    for _ in range(max(train_config.max_steps, 1)):
        optimizer.zero_grad(set_to_none=True)
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
            attention_mask=batch["attention_mask"],
            embedding_projector=embedding_projector,
            hidden_projector=hidden_projector,
            spike_threshold=student.config.spike_threshold,
            membrane_decay=student.config.membrane_decay,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, train_config.gradient_clip)
        optimizer.step()
        scheduler.step()
        snapshots.append({name: float(value.detach().cpu()) for name, value in losses.items()})

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), "config": asdict(train_config)}, output_dir / "dummy-checkpoint.pt")
    initial_losses = snapshots[0]
    final_losses = snapshots[-1]
    trend = {
        **final_losses,
        **{f"initial_{name}": value for name, value in initial_losses.items()},
        **{f"final_{name}": value for name, value in final_losses.items()},
        **{f"delta_{name}": final_losses[name] - initial_losses[name] for name in final_losses},
        "steps": float(len(snapshots)),
    }
    return trend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BiSpikCLM with offline SpAD distillation.")
    parser.add_argument("--config", default="configs/bispikclm_opt125m_spad.toml", help="TOML experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training plan.")
    parser.add_argument("--dummy-batch", action="store_true", help="Run fixed-batch SpAD overfit diagnostics.")
    parser.add_argument("--train", action="store_true", help="Run the real FineWeb-Edu streaming SpAD loop.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path for resume.")
    parser.add_argument("--download-teachers", action="store_true", help="Cache teacher metadata and tokenizer assets.")
    parser.add_argument("--prepare-datasets", action="store_true", help="Write dataset manifests for smoke runs.")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases on rank 0.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_config = load_experiment_config(args.config)
    if args.teacher_model is not None:
        experiment_config.training.teacher_model = args.teacher_model
        experiment_config.model.teacher_model_id = args.teacher_model
    if args.output_dir is not None:
        experiment_config.training.output_dir = args.output_dir
    if args.sequence_length is not None:
        experiment_config.training.sequence_length = args.sequence_length
    if args.time_steps is not None:
        experiment_config.training.time_steps = args.time_steps
        experiment_config.model.num_steps = args.time_steps
    if args.max_steps is not None:
        experiment_config.training.max_steps = args.max_steps
        experiment_config.training.target_tokens = None
    if args.batch_size is not None:
        experiment_config.training.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        experiment_config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        experiment_config.training.learning_rate = args.learning_rate
    if args.precision is not None:
        experiment_config.training.precision = args.precision
    if args.wandb:
        experiment_config.training.use_wandb = True
    if args.wandb_project is not None:
        experiment_config.training.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        experiment_config.training.wandb_run_name = args.wandb_run_name
    if args.download_teachers:
        print({"teachers": download_teachers()})
    if args.prepare_datasets:
        print({"dataset_manifest_dir": str(prepare_dataset_manifests())})
    if args.dry_run:
        print(build_training_payload(experiment_config.model, experiment_config.distillation, experiment_config.training))
    if args.train:
        print(train(experiment_config, resume_from=args.resume_from))
    if args.dummy_batch:
        result = train_dummy_batch(
            experiment_config.training,
            experiment_config.distillation,
        )
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

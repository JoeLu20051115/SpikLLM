from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - runtime dependency for training
    torch = None
    F = None
    nn = None


@dataclass(slots=True)
class SpADConfig:
    temperature: float = 2.0
    lambda_emb: float = 0.2
    lambda_attn: float = 0.1
    lambda_feat: float = 0.1
    lambda_soft: float = 0.3
    lambda_hard: float = 0.3


if nn is None:  # pragma: no cover
    class SpADProjector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("SpADProjector requires torch to be installed")

else:
    class SpADProjector(nn.Module):  # type: ignore[no-redef]
        def __init__(self, student_dim: int, teacher_dim: int) -> None:
            super().__init__()
            self.proj = nn.Identity() if student_dim == teacher_dim else nn.Linear(student_dim, teacher_dim, bias=False)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.proj(tensor)


def temporal_fusion(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("temporal_fusion expects a tensor with at least one dimension")
    return tensor.mean(dim=0)


def _last_hidden(outputs: dict[str, object]) -> torch.Tensor:
    hidden_states = outputs["hidden_states"]
    if not isinstance(hidden_states, tuple):
        raise TypeError("outputs['hidden_states'] must be a tuple")
    return hidden_states[-1]


def _align_sequence(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(student.shape[-2], teacher.shape[-2])
    return student[..., :length, :], teacher[..., :length, :]


def _align_attention(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(student.shape[-1], teacher.shape[-1])
    head_count = min(student.shape[-3], teacher.shape[-3])
    return (
        student[..., :head_count, :seq_len, :seq_len],
        teacher[..., :head_count, :seq_len, :seq_len],
    )


def _project_if_needed(projector: SpADProjector | None, tensor: torch.Tensor) -> torch.Tensor:
    return projector(tensor) if projector is not None else tensor


def compute_multilevel_distillation(
    student_outputs: dict[str, object],
    teacher_outputs: dict[str, object],
    config: SpADConfig,
    labels: torch.Tensor | None = None,
    hidden_projector: SpADProjector | None = None,
    embedding_projector: SpADProjector | None = None,
) -> dict[str, torch.Tensor]:
    if torch is None or F is None:
        raise ImportError("compute_multilevel_distillation requires torch")

    student_embedding = temporal_fusion(student_outputs["embedding_states"])
    teacher_hidden_states = teacher_outputs["hidden_states"]
    if not isinstance(teacher_hidden_states, tuple):
        raise TypeError("teacher_outputs['hidden_states'] must be a tuple")
    teacher_embedding = teacher_hidden_states[0].detach()
    student_embedding = _project_if_needed(embedding_projector, student_embedding)
    student_embedding, teacher_embedding = _align_sequence(student_embedding, teacher_embedding)
    emb_loss = F.mse_loss(student_embedding, teacher_embedding)

    student_attention = temporal_fusion(student_outputs["attentions"][-1])
    teacher_attentions = teacher_outputs["attentions"]
    if not isinstance(teacher_attentions, tuple):
        raise TypeError("teacher_outputs['attentions'] must be a tuple")
    teacher_attention = teacher_attentions[-1].detach()
    student_attention, teacher_attention = _align_attention(student_attention, teacher_attention)
    attn_loss = F.mse_loss(student_attention, teacher_attention)

    student_hidden = temporal_fusion(_last_hidden(student_outputs))
    teacher_hidden = _last_hidden(teacher_outputs).detach()
    student_hidden = _project_if_needed(hidden_projector, student_hidden)
    student_hidden, teacher_hidden = _align_sequence(student_hidden, teacher_hidden)
    feat_loss = F.mse_loss(student_hidden, teacher_hidden)

    student_logits = student_outputs["logits"]
    teacher_logits = teacher_outputs["logits"].detach()
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    seq_len = min(student_logits.shape[-2], teacher_logits.shape[-2])
    student_logits = student_logits[..., :seq_len, :vocab]
    teacher_logits = teacher_logits[..., :seq_len, :vocab]
    temperature = max(config.temperature, 1e-6)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)

    if labels is None:
        hard_loss = student_logits.new_zeros(())
    else:
        labels = labels[..., :seq_len].contiguous()
        hard_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    total = (
        config.lambda_emb * emb_loss
        + config.lambda_attn * attn_loss
        + config.lambda_feat * feat_loss
        + config.lambda_soft * soft_loss
        + config.lambda_hard * hard_loss
    )
    return {
        "embedding_loss": emb_loss,
        "attention_loss": attn_loss,
        "feature_loss": feat_loss,
        "soft_loss": soft_loss,
        "hard_loss": hard_loss,
        "total_loss": total,
    }


def summarize_plan(config: SpADConfig) -> dict[str, object]:
    return {"config": asdict(config), "loss_terms": ["EA", "SAA", "SFA", "STA", "HTA"]}

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
    gamma_attn: float = 0.5
    gamma_feat: float = 0.5


if nn is None:  # pragma: no cover
    class SpADProjector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("SpADProjector requires torch to be installed")

else:
    class SpADProjector(nn.Module):  # type: ignore[no-redef]
        def __init__(self, student_dim: int, teacher_dim: int) -> None:
            super().__init__()
            if student_dim == teacher_dim:
                self.proj = nn.Identity()
                self.norm = nn.Identity()
            else:
                hidden_dim = max(student_dim, teacher_dim)
                self.proj = nn.Sequential(
                    nn.Linear(student_dim, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim, teacher_dim, bias=False),
                )
                self.norm = nn.LayerNorm(teacher_dim)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.norm(self.proj(tensor))


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


def _layer_pairs(student_layers: tuple[torch.Tensor, ...], teacher_layers: tuple[torch.Tensor, ...]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not student_layers or not teacher_layers:
        raise ValueError("student and teacher layer tuples must not be empty")
    if len(student_layers) == 1:
        return [(student_layers[0], teacher_layers[-1])]
    pairs = []
    for student_idx, student_layer in enumerate(student_layers):
        teacher_idx = round(student_idx * (len(teacher_layers) - 1) / (len(student_layers) - 1))
        pairs.append((student_layer, teacher_layers[teacher_idx]))
    return pairs


def _replicate_teacher(tensor: torch.Tensor, time_steps: int) -> torch.Tensor:
    return tensor.detach().unsqueeze(0).expand(time_steps, *tensor.shape)


def _rate_encode(tensor: torch.Tensor, threshold: float, membrane_decay: float) -> torch.Tensor:
    membrane = torch.zeros_like(tensor[0])
    spikes = []
    for step in tensor:
        membrane = membrane * membrane_decay + step
        spike = (membrane >= threshold).to(step.dtype)
        membrane = membrane - spike.detach() * threshold
        spikes.append(spike)
    return torch.stack(spikes, dim=0).mean(dim=0)


def _attention_rate_drive(attention: torch.Tensor, threshold: float) -> torch.Tensor:
    del threshold
    return attention


def _attention_distribution(attention: torch.Tensor) -> torch.Tensor:
    row_sum = attention.sum(dim=-1, keepdim=True)
    return attention / row_sum.clamp_min(1e-6)


def _attention_distribution_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_distribution = _attention_distribution(student)
    teacher_distribution = _attention_distribution(teacher)
    return (student_distribution - teacher_distribution).square().sum(dim=-1).mean()


def _project_if_needed(projector: SpADProjector | None, tensor: torch.Tensor) -> torch.Tensor:
    return projector(tensor) if projector is not None else tensor


def _project_if_shape_mismatch(projector: SpADProjector | None, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if projector is not None and student.shape[-1] != teacher.shape[-1]:
        return projector(student)
    return student


def compute_multilevel_distillation(
    student_outputs: dict[str, object],
    teacher_outputs: dict[str, object],
    config: SpADConfig,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    hidden_projector: SpADProjector | None = None,
    embedding_projector: SpADProjector | None = None,
    spike_threshold: float = 1.0,
    membrane_decay: float = 0.9,
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

    student_attentions = student_outputs["attentions"]
    if not isinstance(student_attentions, tuple):
        raise TypeError("student_outputs['attentions'] must be a tuple")
    teacher_attentions = teacher_outputs["attentions"]
    if not isinstance(teacher_attentions, tuple):
        raise TypeError("teacher_outputs['attentions'] must be a tuple")
    attention_rate_losses = []
    attention_mse_losses = []
    for student_attention_steps, teacher_attention in _layer_pairs(student_attentions, teacher_attentions):
        student_attention = temporal_fusion(student_attention_steps)
        teacher_attention = teacher_attention.detach()
        student_attention, teacher_attention = _align_attention(student_attention, teacher_attention)
        teacher_rate = _rate_encode(
            _replicate_teacher(_attention_rate_drive(teacher_attention, spike_threshold), student_attention_steps.shape[0]),
            spike_threshold,
            membrane_decay,
        )
        student_rate, teacher_rate = _align_attention(student_attention, teacher_rate)
        attention_rate_losses.append(_attention_distribution_mse(student_rate, teacher_rate))
        attention_mse_losses.append(_attention_distribution_mse(student_attention, teacher_attention))
    attention_rate_loss = torch.stack(attention_rate_losses).mean()
    attention_mse_loss = torch.stack(attention_mse_losses).mean()
    attn_loss = config.gamma_attn * attention_rate_loss + (1.0 - config.gamma_attn) * attention_mse_loss

    student_hidden_states = student_outputs["hidden_states"]
    if not isinstance(student_hidden_states, tuple):
        raise TypeError("student_outputs['hidden_states'] must be a tuple")
    feature_rate_losses = []
    feature_mse_losses = []
    student_feature_layers = student_hidden_states[1:] if len(student_hidden_states) > 1 else student_hidden_states
    teacher_feature_layers = teacher_hidden_states[1:] if len(teacher_hidden_states) > 1 else teacher_hidden_states
    for student_hidden_steps, teacher_hidden in _layer_pairs(student_feature_layers, teacher_feature_layers):
        teacher_hidden = teacher_hidden.detach()
        student_fused = temporal_fusion(student_hidden_steps)
        student_hidden = _project_if_needed(hidden_projector, student_fused)
        student_hidden, teacher_hidden_for_mse = _align_sequence(student_hidden, teacher_hidden)
        teacher_rate = _rate_encode(
            _replicate_teacher(teacher_hidden, student_hidden_steps.shape[0]),
            spike_threshold,
            membrane_decay,
        )
        student_rate = _project_if_shape_mismatch(hidden_projector, student_fused, teacher_rate)
        student_rate, teacher_rate = _align_sequence(student_rate, teacher_rate)
        feature_rate_losses.append(F.mse_loss(student_rate, teacher_rate))
        feature_mse_losses.append(F.mse_loss(student_hidden, teacher_hidden_for_mse))
    feature_rate_loss = torch.stack(feature_rate_losses).mean()
    feature_mse_loss = torch.stack(feature_mse_losses).mean()
    feat_loss = config.gamma_feat * feature_rate_loss + (1.0 - config.gamma_feat) * feature_mse_loss

    student_logits = student_outputs["logits"]
    teacher_logits = teacher_outputs["logits"].detach()
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    seq_len = min(student_logits.shape[-2], teacher_logits.shape[-2])
    student_logits = student_logits[..., :seq_len, :vocab]
    teacher_logits = teacher_logits[..., :seq_len, :vocab]
    if labels is not None:
        labels = labels[..., :seq_len]
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask[..., :seq_len].to(dtype=torch.bool), -100)
    temperature = max(config.temperature, 1e-6)
    soft_student_logits = student_logits[..., :-1, :]
    soft_teacher_logits = teacher_logits[..., :-1, :]
    token_soft_loss = F.kl_div(
        F.log_softmax(soft_student_logits / temperature, dim=-1),
        F.softmax(soft_teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    if labels is not None:
        valid_soft_targets = labels[..., 1:seq_len].ne(-100)
        if not valid_soft_targets.any():
            soft_loss = student_logits.sum() * 0.0
        else:
            soft_loss = token_soft_loss.masked_select(valid_soft_targets).mean() * (temperature**2)
    elif attention_mask is not None:
        valid_soft_targets = attention_mask[..., 1:seq_len].to(dtype=torch.bool)
        if not valid_soft_targets.any():
            soft_loss = student_logits.sum() * 0.0
        else:
            soft_loss = token_soft_loss.masked_select(valid_soft_targets).mean() * (temperature**2)
    else:
        soft_loss = token_soft_loss.mean() * (temperature**2)

    if labels is None:
        hard_loss = student_logits.new_zeros(())
    else:
        shift_logits = student_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:seq_len].contiguous()
        if not labels.ne(-100).any():
            hard_loss = student_logits.sum() * 0.0
        else:
            hard_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
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
        "attention_rate_loss": attention_rate_loss,
        "attention_mse_loss": attention_mse_loss,
        "feature_loss": feat_loss,
        "feature_rate_loss": feature_rate_loss,
        "feature_mse_loss": feature_mse_loss,
        "soft_loss": soft_loss,
        "hard_loss": hard_loss,
        "total_loss": total,
    }


def summarize_plan(config: SpADConfig) -> dict[str, object]:
    return {"config": asdict(config), "loss_terms": ["EA", "SAA", "SFA", "STA", "HTA"]}

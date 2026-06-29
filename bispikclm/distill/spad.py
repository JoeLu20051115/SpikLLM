from dataclasses import asdict, dataclass

from .losses import kl_divergence, mean_squared_error


@dataclass(slots=True)
class SpADConfig:
    temperature: float = 1.0
    embedding_weight: float = 1.0
    attention_weight: float = 1.0
    hidden_weight: float = 1.0
    logit_weight: float = 1.0
    mse_weight: float = 1.0
    kl_weight: float = 1.0


def _weighted_mse(student: list[float], teacher: list[float], weight: float) -> float:
    return weight * mean_squared_error(student, teacher)


def compute_multilevel_distillation(
    student_states: dict[str, list[float]],
    teacher_states: dict[str, list[float]],
    config: SpADConfig,
) -> dict[str, float]:
    embedding = _weighted_mse(student_states["embedding"], teacher_states["embedding"], config.embedding_weight)
    attention = _weighted_mse(student_states["attention"], teacher_states["attention"], config.attention_weight)
    hidden = _weighted_mse(student_states["hidden"], teacher_states["hidden"], config.hidden_weight)
    logit = config.logit_weight * config.kl_weight * kl_divergence(
        student_states["logit"],
        teacher_states["logit"],
    )
    total = config.mse_weight * (embedding + attention + hidden) + logit / max(config.temperature, 1e-6)
    return {
        "embedding_loss": embedding,
        "attention_loss": attention,
        "hidden_loss": hidden,
        "logit_loss": logit,
        "total_loss": total,
    }


def summarize_plan(config: SpADConfig) -> dict[str, float | dict[str, float]]:
    reference_losses = compute_multilevel_distillation(
        student_states={
            "embedding": [0.0, 0.0],
            "attention": [0.0, 0.0],
            "hidden": [0.0, 0.0],
            "logit": [0.5, 0.5],
        },
        teacher_states={
            "embedding": [0.0, 0.0],
            "attention": [0.0, 0.0],
            "hidden": [0.0, 0.0],
            "logit": [0.5, 0.5],
        },
        config=config,
    )
    return {
        "config": asdict(config),
        "reference_losses": reference_losses,
        "reference_mse": mean_squared_error([0.0], [0.0]),
        "reference_kl": kl_divergence([1.0], [1.0]),
    }

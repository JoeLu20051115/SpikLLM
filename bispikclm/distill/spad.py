from dataclasses import dataclass

from .losses import kl_divergence, mean_squared_error


@dataclass(slots=True)
class SpADConfig:
    temperature: float = 1.0
    mse_weight: float = 1.0
    kl_weight: float = 1.0


def summarize_plan(config: SpADConfig) -> dict[str, float]:
    return {
        "temperature": config.temperature,
        "mse_weight": config.mse_weight,
        "kl_weight": config.kl_weight,
        "reference_mse": mean_squared_error([0.0], [0.0]),
        "reference_kl": kl_divergence([1.0], [1.0]),
    }


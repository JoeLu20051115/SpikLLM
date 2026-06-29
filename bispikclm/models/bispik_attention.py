from dataclasses import dataclass

from .bispik_config import BiSpikConfig


@dataclass(slots=True)
class BiSpikAttention:
    config: BiSpikConfig

    def forward(self, hidden_state: list[float]) -> list[float]:
        scale = 1.0 / max(self.config.num_attention_heads, 1)
        return [value * scale for value in hidden_state]


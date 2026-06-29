from dataclasses import dataclass

from .bispik_config import BiSpikConfig


@dataclass(slots=True)
class BiSpikMLP:
    config: BiSpikConfig

    def forward(self, hidden_state: list[float]) -> list[float]:
        gain = min(self.config.intermediate_size / max(self.config.hidden_size, 1), 4.0)
        return [value * gain for value in hidden_state]


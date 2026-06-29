from dataclasses import dataclass

from .bispik_attention import BiSpikAttention
from .bispik_config import BiSpikConfig
from .bispik_mlp import BiSpikMLP


@dataclass(slots=True)
class BiSpikBlock:
    attention: BiSpikAttention
    mlp: BiSpikMLP

    @classmethod
    def from_config(cls, config: BiSpikConfig) -> "BiSpikBlock":
        return cls(attention=BiSpikAttention(config), mlp=BiSpikMLP(config))

    def forward(self, hidden_state: list[float]) -> list[float]:
        attended = self.attention.forward(hidden_state)
        transformed = self.mlp.forward(attended)
        return [left + right for left, right in zip(hidden_state, transformed, strict=False)]


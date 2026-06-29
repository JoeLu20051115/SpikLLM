from dataclasses import dataclass
from typing import Any

from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


@dataclass(slots=True)
class BiSpikMLP:
    config: BiSpikConfig
    fc1: Any = None
    fc2: Any = None

    def __post_init__(self) -> None:
        if nn is not None:
            self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
            self.fc2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

    def forward(self, hidden_state: list[float]) -> list[float]:
        if torch is not None and isinstance(hidden_state, torch.Tensor):
            activated = torch.relu(self.fc1(hidden_state))
            return self.fc2(activated)
        gain = min(self.config.intermediate_size / max(self.config.hidden_size, 1), 4.0)
        return [value * gain for value in hidden_state]

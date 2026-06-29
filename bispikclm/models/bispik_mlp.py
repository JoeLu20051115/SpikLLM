from __future__ import annotations

from typing import Any

from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikMLP:  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            self.config = config
            self.fc1: Any = None
            self.fc2: Any = None

        def forward(self, hidden_state: list[float]) -> list[float]:
            gain = min(self.config.intermediate_size / max(self.config.hidden_size, 1), 4.0)
            return [value * gain for value in hidden_state]

else:
    class BiSpikMLP(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.config = config
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        def forward(self, hidden_state: torch.Tensor | list[float]) -> torch.Tensor | list[float]:
            if isinstance(hidden_state, torch.Tensor):
                activated = torch.relu(self.fc1(hidden_state))
                return self.fc2(activated)
            gain = min(self.config.intermediate_size / max(self.config.hidden_size, 1), 4.0)
            return [value * gain for value in hidden_state]

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
class BiSpikAttention:
    config: BiSpikConfig
    q_proj: Any = None
    k_proj: Any = None
    v_proj: Any = None
    out_proj: Any = None

    def __post_init__(self) -> None:
        if nn is not None:
            self.q_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
            self.k_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
            self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
            self.out_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(self, hidden_state: list[float]) -> list[float]:
        if torch is not None and isinstance(hidden_state, torch.Tensor):
            query = self.q_proj(hidden_state)
            key = self.k_proj(hidden_state)
            value = self.v_proj(hidden_state)
            scale = max(query.shape[-1], 1) ** -0.5
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            weights = torch.softmax(scores, dim=-1)
            return self.out_proj(torch.matmul(weights, value))
        scale = 1.0 / max(self.config.num_attention_heads, 1)
        return [value * scale for value in hidden_state]

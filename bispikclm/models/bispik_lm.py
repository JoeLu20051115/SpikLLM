from dataclasses import dataclass, field
from typing import Any

from .bispik_config import BiSpikConfig
from .bispik_model import BiSpikModel

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


@dataclass(slots=True)
class BiSpikForCausalLM:
    config: BiSpikConfig
    model: BiSpikModel = field(init=False)
    lm_head: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model = BiSpikModel(self.config)
        if nn is not None:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, token_values: list[float]) -> dict[str, list[float]]:
        hidden_state = self.model.forward(token_values)
        if torch is not None and isinstance(hidden_state, torch.Tensor):
            logits = self.lm_head(hidden_state)
            return {"logits": logits}
        logits = hidden_state[: self.config.vocab_size]
        if len(logits) < self.config.vocab_size:
            logits = logits + [0.0] * (self.config.vocab_size - len(logits))
        return {"logits": logits}

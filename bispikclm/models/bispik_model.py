from dataclasses import dataclass, field
from typing import Any

from .bispik_block import BiSpikBlock
from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


@dataclass(slots=True)
class BiSpikModel:
    config: BiSpikConfig
    blocks: list[BiSpikBlock] = field(init=False)
    token_embedding: Any = field(init=False, default=None)
    position_embeddings: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.blocks = [BiSpikBlock.from_config(self.config) for _ in range(self.config.num_hidden_layers)]
        if nn is not None:
            self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)

    def forward(self, token_values: list[float]) -> list[float]:
        if torch is not None and isinstance(token_values, torch.Tensor):
            if token_values.dtype in (torch.int32, torch.int64):
                positions = torch.arange(token_values.shape[-1], device=token_values.device)
                hidden_state = self.token_embedding(token_values) + self.position_embeddings(positions)
            else:
                hidden_state = token_values
            for block in self.blocks:
                hidden_state = block.forward(hidden_state)
            return hidden_state
        hidden_state = token_values
        for block in self.blocks:
            hidden_state = block.forward(hidden_state)
        return hidden_state

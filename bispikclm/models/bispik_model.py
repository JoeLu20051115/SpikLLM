from dataclasses import dataclass, field

from .bispik_block import BiSpikBlock
from .bispik_config import BiSpikConfig


@dataclass(slots=True)
class BiSpikModel:
    config: BiSpikConfig
    blocks: list[BiSpikBlock] = field(init=False)

    def __post_init__(self) -> None:
        self.blocks = [BiSpikBlock.from_config(self.config) for _ in range(self.config.num_hidden_layers)]

    def forward(self, token_values: list[float]) -> list[float]:
        hidden_state = token_values
        for block in self.blocks:
            hidden_state = block.forward(hidden_state)
        return hidden_state


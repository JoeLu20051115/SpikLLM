from dataclasses import dataclass, field

from .bispik_config import BiSpikConfig
from .bispik_model import BiSpikModel


@dataclass(slots=True)
class BiSpikForCausalLM:
    config: BiSpikConfig
    model: BiSpikModel = field(init=False)

    def __post_init__(self) -> None:
        self.model = BiSpikModel(self.config)

    def forward(self, token_values: list[float]) -> dict[str, list[float]]:
        hidden_state = self.model.forward(token_values)
        logits = hidden_state[: self.config.vocab_size]
        return {"logits": logits}


from __future__ import annotations

from .bispik_attention import BiSpikAttention
from .bispik_config import BiSpikConfig
from .bispik_mlp import BiSpikMLP

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


def _call_forward(module, *args, **kwargs):
    if callable(module):
        return module(*args, **kwargs)
    return module.forward(*args, **kwargs)


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikBlock:  # type: ignore[no-redef]
        def __init__(self, attention: BiSpikAttention, mlp: BiSpikMLP, config: BiSpikConfig | None = None) -> None:
            self.attention = attention
            self.mlp = mlp
            self.config = config or attention.config

        @classmethod
        def from_config(cls, config: BiSpikConfig) -> "BiSpikBlock":
            return cls(attention=BiSpikAttention(config), mlp=BiSpikMLP(config), config=config)

        def forward(
            self,
            hidden_state,
            attention_mask=None,
            return_attention: bool = False,
            return_spike_stats: bool = False,
        ):
            del attention_mask, return_attention, return_spike_stats
            attended = _call_forward(self.attention, hidden_state)
            transformed = _call_forward(self.mlp, attended)
            if len(hidden_state) != len(transformed):
                raise ValueError("hidden_state and transformed outputs must have the same length")
            return [left + right for left, right in zip(hidden_state, transformed, strict=True)]

else:
    class BiSpikBlock(nn.Module):  # type: ignore[no-redef]
        def __init__(self, attention: BiSpikAttention, mlp: BiSpikMLP, config: BiSpikConfig | None = None) -> None:
            super().__init__()
            self.attention = attention
            self.mlp = mlp
            self.config = config or attention.config
            self.spike_threshold = self.config.spike_threshold
            self.membrane_decay = self.config.membrane_decay

        @classmethod
        def from_config(cls, config: BiSpikConfig) -> "BiSpikBlock":
            return cls(attention=BiSpikAttention(config), mlp=BiSpikMLP(config), config=config)

        def forward(
            self,
            hidden_state: torch.Tensor | list[float],
            attention_mask: torch.Tensor | None = None,
            return_attention: bool = False,
            return_spike_stats: bool = False,
        ):
            attention_result = _call_forward(
                self.attention,
                hidden_state,
                attention_mask=attention_mask,
                return_weights=return_attention,
            )
            if return_attention:
                attended, attention_weights = attention_result
            else:
                attended = attention_result
                attention_weights = None
            transformed = _call_forward(self.mlp, attended)
            if isinstance(hidden_state, torch.Tensor):
                hidden_state = hidden_state + transformed
                spike_stats = None
                if return_spike_stats:
                    spike_tensor = (hidden_state > self.spike_threshold).to(hidden_state.dtype)
                    spike_stats = {
                        "spike_rate": spike_tensor.mean(),
                        "membrane_mean": (hidden_state * self.membrane_decay).mean(),
                    }
                if return_attention or return_spike_stats:
                    return hidden_state, attention_weights, spike_stats
                return hidden_state

            if len(hidden_state) != len(transformed):
                raise ValueError("hidden_state and transformed outputs must have the same length")
            output = [left + right for left, right in zip(hidden_state, transformed, strict=True)]
            if return_attention or return_spike_stats:
                return output, attention_weights, None
            return output

from __future__ import annotations

from .bispik_attention import BiSpikAttention
from .bispik_config import BiSpikConfig
from .bispik_mlp import BiSpikMLP

try:
    import torch
    from torch import nn
    from spikingjelly.activation_based import neuron, surrogate
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    neuron = None
    surrogate = None


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
            self.attention_norm = nn.LayerNorm(self.config.hidden_size)
            self.mlp_norm = nn.LayerNorm(self.config.hidden_size)
            if neuron is None or surrogate is None:
                raise ImportError("BiSpikBlock requires spikingjelly for LIF activation")
            self.out_lif = neuron.LIFNode(
                tau=1.0 / max(1.0 - self.config.membrane_decay, 1e-6),
                v_threshold=self.config.spike_threshold,
                surrogate_function=surrogate.ATan(alpha=self.config.surrogate_alpha),
                detach_reset=True,
                decay_input=False,
            )

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
            attention_input = self.attention_norm(hidden_state) if isinstance(hidden_state, torch.Tensor) else hidden_state
            attention_result = _call_forward(
                self.attention,
                attention_input,
                attention_mask=attention_mask,
                return_weights=return_attention,
            )
            if isinstance(attention_result, dict):
                attended = attention_result["context"]
                attention_weights = attention_result["attention_spikes"]
            elif return_attention:
                attended, attention_weights = attention_result
            else:
                attended = attention_result
                attention_weights = None
            if isinstance(hidden_state, torch.Tensor):
                attention_residual = hidden_state + attended
                transformed = _call_forward(self.mlp, self.mlp_norm(attention_residual))
                hidden_state = self.out_lif(attention_residual + transformed)
                spike_stats = None
                if return_spike_stats:
                    spike_stats = {
                        "spike_rate": hidden_state.mean(),
                        "membrane_mean": (hidden_state * self.membrane_decay).mean(),
                    }
                if return_attention or return_spike_stats:
                    return hidden_state, attention_weights, spike_stats
                return hidden_state

            transformed = _call_forward(self.mlp, attended)
            if len(hidden_state) != len(transformed):
                raise ValueError("hidden_state and transformed outputs must have the same length")
            output = [left + right for left, right in zip(hidden_state, transformed, strict=True)]
            if return_attention or return_spike_stats:
                return output, attention_weights, None
            return output

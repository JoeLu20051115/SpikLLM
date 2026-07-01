from __future__ import annotations

from typing import Any

from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
    from spikingjelly.activation_based import neuron, surrogate
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    neuron = None
    surrogate = None


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikAttention:  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            self.config = config
            self.q_proj: Any = None
            self.k_proj: Any = None
            self.v_proj: Any = None
            self.out_proj: Any = None

        def forward(
            self,
            hidden_state: list[float],
            attention_mask: Any = None,
            return_weights: bool = False,
        ) -> Any:
            del attention_mask
            scale = 1.0 / max(self.config.num_attention_heads, 1)
            attended = [value * scale for value in hidden_state]
            if return_weights:
                return attended, None
            return attended

else:
    class BiSpikAttention(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.config = config
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                projection.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if neuron is None or surrogate is None:
                raise ImportError("BiSpikAttention requires spikingjelly for LIF activation")
            lif_kwargs = {
                "tau": 1.0 / max(1.0 - config.membrane_decay, 1e-6),
                "v_threshold": config.spike_threshold,
                "surrogate_function": surrogate.ATan(alpha=config.surrogate_alpha),
                "detach_reset": True,
                "decay_input": False,
            }
            self.q_lif = neuron.LIFNode(**lif_kwargs)
            self.k_lif = neuron.LIFNode(**lif_kwargs)
            self.v_lif = neuron.LIFNode(**lif_kwargs)
            self.attn_lif = neuron.LIFNode(**lif_kwargs)
            self.attn_out_lif = neuron.LIFNode(**lif_kwargs)
            self.out_lif = neuron.LIFNode(**lif_kwargs)

        def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, _ = tensor.shape
            return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
            batch_size, _, sequence_length, _ = tensor.shape
            return tensor.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.config.hidden_size)

        def forward(
            self,
            hidden_state: torch.Tensor | list[float],
            attention_mask: torch.Tensor | None = None,
            return_weights: bool = False,
        ) -> dict[str, torch.Tensor] | list[float] | tuple[list[float], None]:
            if isinstance(hidden_state, torch.Tensor):
                query = self._split_heads(self.q_proj(hidden_state))
                key = self._split_heads(self.k_proj(hidden_state))
                value = self._split_heads(self.v_proj(hidden_state))
                query_spikes = self.q_lif(query)
                key_spikes = self.k_lif(key)
                value_spikes = self.v_lif(value)
                attention_int = torch.matmul(query_spikes, key_spikes.transpose(-1, -2))
                sequence_length = hidden_state.shape[-2]
                causal_mask = torch.ones(
                    (sequence_length, sequence_length),
                    dtype=attention_int.dtype,
                    device=hidden_state.device,
                ).tril()
                valid_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                if attention_mask is not None:
                    key_mask = attention_mask[:, None, None, :].to(dtype=attention_int.dtype, device=hidden_state.device)
                    valid_mask = valid_mask * key_mask

                masked_attention_int = attention_int * valid_mask
                attention_spikes = self.attn_lif(masked_attention_int) * valid_mask
                attention_out = torch.matmul(attention_spikes, value_spikes)
                attention_out_spikes = self.attn_out_lif(attention_out)
                projected = self.out_proj(self._merge_heads(attention_out_spikes))
                attended = self.out_lif(projected)
                del return_weights
                return {
                    "context": attended,
                    "attention_scores": masked_attention_int,
                    "attention_int": attention_int,
                    "query_spikes": query_spikes,
                    "key_spikes": key_spikes,
                    "value_spikes": value_spikes,
                    "attention_spikes": attention_spikes,
                }

            scale = 1.0 / max(self.config.num_attention_heads, 1)
            attended = [value * scale for value in hidden_state]
            if return_weights:
                return attended, None
            return attended

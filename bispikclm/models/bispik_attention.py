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
        ) -> torch.Tensor | list[float] | tuple[torch.Tensor, torch.Tensor | None]:
            if isinstance(hidden_state, torch.Tensor):
                query = self._split_heads(self.q_proj(hidden_state))
                key = self._split_heads(self.k_proj(hidden_state))
                value = self._split_heads(self.v_proj(hidden_state))
                scale = max(self.head_dim, 1) ** -0.5
                scores = torch.matmul(query, key.transpose(-1, -2)) * scale
                sequence_length = hidden_state.shape[-2]
                causal_mask = torch.ones(
                    (sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=hidden_state.device,
                ).tril()
                valid_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                if attention_mask is not None:
                    key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool, device=hidden_state.device)
                    valid_mask = valid_mask & key_mask

                masked_scores = scores.masked_fill(~valid_mask, 0.0)
                spikes = (masked_scores > self.config.spike_threshold).to(scores.dtype) * valid_mask.to(scores.dtype)
                support = spikes.sum(dim=-1, keepdim=True).clamp_min(1.0)
                weights = spikes / support
                attended = self.out_proj(self._merge_heads(torch.matmul(weights, value)))
                if return_weights:
                    return attended, weights
                return {
                    "context": attended,
                    "attention_scores": masked_scores,
                    "attention_spikes": spikes,
                }

            scale = 1.0 / max(self.config.num_attention_heads, 1)
            attended = [value * scale for value in hidden_state]
            if return_weights:
                return attended, None
            return attended

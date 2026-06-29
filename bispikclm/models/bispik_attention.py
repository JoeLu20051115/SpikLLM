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
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        def forward(
            self,
            hidden_state: torch.Tensor | list[float],
            attention_mask: torch.Tensor | None = None,
            return_weights: bool = False,
        ) -> torch.Tensor | list[float] | tuple[torch.Tensor, torch.Tensor | None]:
            if isinstance(hidden_state, torch.Tensor):
                query = self.q_proj(hidden_state)
                key = self.k_proj(hidden_state)
                value = self.v_proj(hidden_state)
                scale = max(query.shape[-1], 1) ** -0.5
                scores = torch.matmul(query, key.transpose(-1, -2)) * scale
                sequence_length = hidden_state.shape[-2]
                causal_mask = torch.ones(
                    (sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=hidden_state.device,
                ).tril()
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
                if attention_mask is not None:
                    key_mask = attention_mask[:, None, :].to(dtype=torch.bool, device=hidden_state.device)
                    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=-1)
                attended = self.out_proj(torch.matmul(weights, value))
                if return_weights:
                    return attended, weights
                return attended

            scale = 1.0 / max(self.config.num_attention_heads, 1)
            attended = [value * scale for value in hidden_state]
            if return_weights:
                return attended, None
            return attended

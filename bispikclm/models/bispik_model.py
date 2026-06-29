from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None

from .bispik_config import BiSpikConfig


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikModel:  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            self.config = config

        def forward(self, *args, **kwargs):
            raise ImportError("BiSpikModel requires torch to be installed")

else:
    class _BiSpikLayer(nn.Module):
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.spike_threshold = config.spike_threshold
            self.membrane_decay = config.membrane_decay
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            query = self.q_proj(hidden_state)
            key = self.k_proj(hidden_state)
            value = self.v_proj(hidden_state)
            scale = max(query.shape[-1], 1) ** -0.5
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            if attention_mask is not None:
                key_mask = attention_mask[:, None, :].to(dtype=torch.bool, device=hidden_state.device)
                scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            attended = self.out_proj(torch.matmul(weights, value))
            transformed = self.fc2(torch.relu(self.fc1(attended)))
            hidden_state = hidden_state + transformed
            spike_tensor = (hidden_state > self.spike_threshold).to(hidden_state.dtype)
            spike_stats = {
                "spike_rate": spike_tensor.mean(),
                "membrane_mean": (hidden_state * self.membrane_decay).mean(),
            }
            return hidden_state, weights, spike_stats


    class BiSpikModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.layers = nn.ModuleList(_BiSpikLayer(config) for _ in range(config.num_hidden_layers))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            return_spike_stats: bool = False,
        ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[dict[str, torch.Tensor]] | None]:
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("input_ids must be an integer tensor")
            sequence_length = input_ids.shape[-1]
            positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
            hidden_state = self.token_embedding(input_ids) + self.position_embeddings(positions)
            if attention_mask is not None:
                hidden_state = hidden_state * attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            embedding_states = hidden_state
            hidden_states = [hidden_state] if output_hidden_states else None
            attentions = [] if output_attentions else None
            spike_stats = [] if return_spike_stats else None

            for layer in self.layers:
                hidden_state, attention_weights, layer_spike_stats = layer(hidden_state, attention_mask)
                if attention_mask is not None:
                    hidden_state = hidden_state * attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                if hidden_states is not None:
                    hidden_states.append(hidden_state)
                if attentions is not None:
                    attentions.append(attention_weights)
                if spike_stats is not None:
                    spike_stats.append(layer_spike_stats)

            return {
                "last_hidden_state": hidden_state,
                "hidden_states": tuple(hidden_states) if hidden_states is not None else None,
                "attentions": tuple(attentions) if attentions is not None else None,
                "spike_stats": spike_stats,
                "embedding_states": embedding_states,
            }

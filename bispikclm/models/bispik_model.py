from __future__ import annotations

from .bispik_block import BiSpikBlock
from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikModel:  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            self.config = config

        def forward(self, *args, **kwargs):
            raise ImportError("BiSpikModel requires torch to be installed")

else:
    class BiSpikModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.layers = nn.ModuleList(
                BiSpikBlock.from_config(config) for _ in range(config.num_hidden_layers)
            )

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
                block_output = layer(
                    hidden_state,
                    attention_mask=attention_mask,
                    return_attention=output_attentions,
                    return_spike_stats=return_spike_stats,
                )
                if output_attentions or return_spike_stats:
                    hidden_state, attention_weights, layer_spike_stats = block_output
                else:
                    hidden_state = block_output
                    attention_weights = None
                    layer_spike_stats = None
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

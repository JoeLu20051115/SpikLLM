from __future__ import annotations

from .bispik_block import BiSpikBlock
from .bispik_config import BiSpikConfig

try:
    import torch
    from torch import nn
    from spikingjelly.activation_based import functional
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    functional = None


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
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.layers = nn.ModuleList(
                BiSpikBlock.from_config(config) for _ in range(config.num_hidden_layers)
            )
            self.token_embedding.weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if self.token_embedding.padding_idx is not None:
                self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

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
            if functional is not None:
                functional.reset_net(self)

            sequence_length = input_ids.shape[-1]
            positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
            base_embedding = self.token_embedding(input_ids) + self.position_embeddings(positions)
            input_scale = self.config.input_scale
            if input_scale is None:
                input_scale = 1.0 / max(self.config.initializer_range, 1e-6)
            base_embedding = base_embedding * input_scale
            if attention_mask is not None:
                base_embedding = base_embedding * attention_mask.unsqueeze(-1).to(base_embedding.dtype)

            step_embeddings = []
            step_last_hidden_states = []
            step_hidden_states = [] if output_hidden_states else None
            step_attentions = [] if output_attentions else None
            step_spike_stats = [] if return_spike_stats else None

            for step in range(self.config.num_steps):
                step_scale = float(step + 1) / float(self.config.num_steps)
                hidden_state = base_embedding * step_scale
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

                step_embeddings.append(hidden_states[0] if hidden_states is not None else base_embedding * step_scale)
                step_last_hidden_states.append(hidden_state)
                if step_hidden_states is not None:
                    step_hidden_states.append(tuple(hidden_states))
                if step_attentions is not None:
                    step_attentions.append(tuple(attentions))
                if step_spike_stats is not None:
                    step_spike_stats.append(spike_stats)

            hidden_state = torch.stack(step_last_hidden_states, dim=0).mean(dim=0)
            embedding_states = torch.stack(step_embeddings, dim=0)
            hidden_states = (
                tuple(torch.stack([step[layer_idx] for step in step_hidden_states], dim=0) for layer_idx in range(len(step_hidden_states[0])))
                if step_hidden_states is not None
                else None
            )
            attentions = (
                tuple(torch.stack([step[layer_idx] for step in step_attentions], dim=0) for layer_idx in range(len(step_attentions[0])))
                if step_attentions is not None
                else None
            )
            spike_stats = step_spike_stats[-1] if step_spike_stats is not None else None

            return {
                "last_hidden_state": hidden_state,
                "hidden_states": tuple(hidden_states) if hidden_states is not None else None,
                "attentions": tuple(attentions) if attentions is not None else None,
                "spike_stats": spike_stats,
                "embedding_states": embedding_states,
            }

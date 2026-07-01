from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    F = None
    nn = None

from .bispik_config import BiSpikConfig
from .bispik_model import BiSpikModel


if nn is None:  # pragma: no cover - import-time fallback when torch is unavailable
    class BiSpikForCausalLM:  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            self.config = config
            self.model = BiSpikModel(config)

        def forward(self, *args, **kwargs):
            raise ImportError("BiSpikForCausalLM requires torch to be installed")

else:
    class BiSpikForCausalLM(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: BiSpikConfig) -> None:
            super().__init__()
            self.config = config
            self.model = BiSpikModel(config)
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.token_embedding.weight
            self.readout_log_scale = nn.Parameter(
                torch.tensor(math.log(max(config.readout_scale, 1e-6)), dtype=torch.float32)
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            return_spike_stats: bool = False,
        ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[dict[str, torch.Tensor]] | None]:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_spike_stats=return_spike_stats,
            )
            hidden_states = model_output.get("hidden_states")
            if not isinstance(hidden_states, tuple) or not hidden_states:
                raise TypeError("BiSpikModel must return temporal hidden states for LM readout")
            last_hidden_steps = hidden_states[-1]
            assert isinstance(last_hidden_steps, torch.Tensor)
            if last_hidden_steps.ndim == 4:
                step_logits = self.lm_head(self.final_layer_norm(last_hidden_steps))
                logits = step_logits.mean(dim=0) * self.readout_log_scale.exp()
            else:
                logits = self.lm_head(self.final_layer_norm(last_hidden_steps)) * self.readout_log_scale.exp()
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous().to(dtype=torch.bool)
                    shift_labels = shift_labels.masked_fill(~shift_mask, -100)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            return {
                "logits": logits,
                "hidden_states": hidden_states if output_hidden_states else None,
                "attentions": model_output.get("attentions"),
                "spike_stats": model_output.get("spike_stats"),
                "embedding_states": model_output.get("embedding_states"),
                "loss": loss,
            }

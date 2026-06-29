from dataclasses import dataclass


@dataclass(slots=True)
class BiSpikConfig:
    vocab_size: int = 50272
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048
    spike_surrogate: str = "arctan"
    teacher_model_id: str = "facebook/opt-125m"


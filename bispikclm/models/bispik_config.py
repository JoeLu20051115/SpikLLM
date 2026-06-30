from dataclasses import dataclass


@dataclass(slots=True)
class BiSpikConfig:
    vocab_size: int = 50272
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 2
    num_steps: int = 4
    spike_threshold: float = 0.7
    tau: float = 0.9
    membrane_decay: float = 0.9
    spike_surrogate: str = "arctan"
    surrogate_alpha: float = 2.0
    initializer_range: float = 0.02
    input_scale: float | None = None
    readout_scale: float = 2.0
    teacher_model_id: str = "facebook/opt-125m"

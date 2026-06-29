from pathlib import Path
import json

from bispikclm.distill.spad import SpADConfig
from bispikclm.models.bispik_attention import BiSpikAttention
from bispikclm.models.bispik_block import BiSpikBlock
from bispikclm.models.bispik_config import BiSpikConfig
from bispikclm.models.bispik_lm import BiSpikForCausalLM
from bispikclm.train.eval_lm import main as eval_main
from bispikclm.train.train_spad import build_training_payload


def test_scaffold_smoke() -> None:
    config = BiSpikConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2)
    model = BiSpikForCausalLM(config)

    assert model.config.hidden_size == 16
    assert eval_main(["--smoke-datasets"]) == 0
    assert not Path("bispikclm/cache").exists()


def test_lm_forward_returns_tensor_features() -> None:
    import torch

    config = BiSpikConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=16,
        num_steps=2,
    )
    model = BiSpikForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=True,
        return_spike_stats=True,
    )

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert len(output["hidden_states"]) == config.num_hidden_layers + 1
    assert len(output["attentions"]) == config.num_hidden_layers
    assert len(output["spike_stats"]) == config.num_hidden_layers


def test_block_forward_rejects_shape_mismatch() -> None:
    class ShortMLP:
        def forward(self, hidden_state: list[float]) -> list[float]:
            return hidden_state[:-1]

    config = BiSpikConfig(hidden_size=4)
    block = BiSpikBlock(attention=BiSpikAttention(config), mlp=ShortMLP())

    try:
        block.forward([1.0, 2.0, 3.0, 4.0])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched block outputs")


def test_training_payload_exposes_teacher_and_multilevel_spad_plan() -> None:
    payload = build_training_payload(BiSpikConfig(), distill_config=SpADConfig())

    serialized = json.loads(json.dumps(payload, sort_keys=True))

    assert "teacher_runtime" in serialized
    assert "distillation" in serialized
    assert "reference_losses" in serialized["distillation"]
    assert "train_loop" in serialized
    assert "runtime_requirements" in serialized

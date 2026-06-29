from pathlib import Path

from bispikclm.models.bispik_attention import BiSpikAttention
from bispikclm.models.bispik_block import BiSpikBlock
from bispikclm.models.bispik_config import BiSpikConfig
from bispikclm.models.bispik_lm import BiSpikForCausalLM
from bispikclm.train.eval_lm import main as eval_main


def test_scaffold_smoke() -> None:
    config = BiSpikConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2)
    model = BiSpikForCausalLM(config)

    assert model.config.hidden_size == 16
    assert eval_main(["--smoke-datasets"]) == 0
    assert not Path("bispikclm/cache").exists()


def test_lm_forward_returns_vocab_sized_logits() -> None:
    config = BiSpikConfig(vocab_size=6, hidden_size=4, num_hidden_layers=1)
    model = BiSpikForCausalLM(config)

    output = model.forward([1.0, 2.0, 3.0, 4.0])

    assert len(output["logits"]) == config.vocab_size
    assert output["logits"][-2:] == [0.0, 0.0]


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

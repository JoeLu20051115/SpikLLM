from __future__ import annotations

from pathlib import Path


class TinyTokenizer:
    pad_token_id = 1
    bos_token_id = 2
    eos_token_id = 2
    vocab_size = 64

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [3 + (ord(char) % 20) for char in text]}


def test_table3_config_file_is_loaded_and_propagated() -> None:
    from bispikclm.train.train_spad import load_experiment_config

    config = load_experiment_config(Path("configs/bispikclm_opt125m_spad.toml"))

    assert config.training.learning_rate == 5e-4
    assert config.training.batch_size == 16
    assert config.training.gradient_accumulation_steps == 16
    assert config.training.warmup_ratio == 0.2
    assert config.training.gradient_clip == 0.7
    assert config.distillation.temperature == 2.0
    assert config.distillation.lambda_emb == 0.2
    assert config.model.spike_threshold == 0.7
    assert config.model.surrogate_alpha == 2.0
    assert config.model.readout_scale == 1.0
    assert config.training.time_steps in (2, 4)
    assert config.training.target_tokens == 1_000_000_000


def test_three_opt_scale_configs_are_ready_for_training() -> None:
    from bispikclm.train.train_spad import load_experiment_config, resolve_max_steps

    expected = {
        "configs/bispikclm_opt125m_spad.toml": ("facebook/opt-125m", 768, 3072, 12, 12),
        "configs/bispikclm_opt350m_spad.toml": ("facebook/opt-350m", 1024, 4096, 16, 24),
        "configs/bispikclm_opt13b_spad.toml": ("facebook/opt-1.3b", 2048, 8192, 32, 24),
    }

    for path, (teacher, hidden, ffn, heads, layers) in expected.items():
        config = load_experiment_config(Path(path))
        assert config.training.teacher_model == teacher
        assert config.model.hidden_size == hidden
        assert config.model.intermediate_size == ffn
        assert config.model.num_attention_heads == heads
        assert config.model.num_hidden_layers == layers
        assert config.training.learning_rate == 5e-4
        assert config.training.warmup_ratio == 0.2
        assert config.training.gradient_clip == 0.7
        assert config.model.spike_threshold == 0.7
        assert config.model.readout_scale == 1.0
        assert resolve_max_steps(config.training, world_size=8) == 239


def test_lm_head_is_tied_to_token_embedding_for_opt_parameter_budget() -> None:
    from bispikclm.models import BiSpikConfig, BiSpikForCausalLM

    model = BiSpikForCausalLM(BiSpikConfig())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert model.lm_head.weight.data_ptr() == model.model.token_embedding.weight.data_ptr()
    assert 120_000_000 <= parameter_count <= 130_000_000


def test_lm_head_applies_configured_readout_scale() -> None:
    import torch

    from bispikclm.models import BiSpikConfig, BiSpikForCausalLM

    torch.manual_seed(0)
    input_ids = torch.tensor([[2, 3, 4, 5]])
    base_config = BiSpikConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
        readout_scale=1.0,
    )
    scaled_config = BiSpikConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
        readout_scale=2.0,
    )
    base = BiSpikForCausalLM(base_config)
    scaled = BiSpikForCausalLM(scaled_config)
    scaled.load_state_dict(base.state_dict(), strict=False)
    scaled.readout_log_scale.data.fill_(torch.log(torch.tensor(2.0)))

    base_logits = base(input_ids)["logits"]
    scaled_logits = scaled(input_ids)["logits"]

    assert torch.allclose(scaled_logits, base_logits * 2.0)


def test_fineweb_streaming_sequence_packing_builds_real_batches() -> None:
    from bispikclm.data.fineweb import SequencePackingIterableDataset, collate_packed_sequences

    dataset = SequencePackingIterableDataset(
        rows=iter([{"text": "abcdefghij"}, {"text": "klmnopqrst"}]),
        tokenizer=TinyTokenizer(),
        sequence_length=6,
    )
    examples = list(dataset)
    batch = collate_packed_sequences(examples[:2], pad_token_id=TinyTokenizer.pad_token_id)

    assert len(examples) >= 2
    assert examples[0]["input_ids"].shape[0] == 6
    assert examples[0]["labels"].equal(examples[0]["input_ids"])
    assert batch["input_ids"].shape == (2, 6)
    assert batch["attention_mask"].shape == (2, 6)


def test_student_config_inherits_opt_dimensions_without_debug_truncation() -> None:
    from types import SimpleNamespace

    from bispikclm.train.train_spad import TrainingConfig, build_student_config_from_teacher_config

    teacher_config = SimpleNamespace(
        vocab_size=50272,
        hidden_size=768,
        ffn_dim=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=2048,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
    )

    student_config = build_student_config_from_teacher_config(
        teacher_config,
        TrainingConfig(time_steps=4, teacher_model="facebook/opt-125m"),
    )

    assert student_config.hidden_size == 768
    assert student_config.intermediate_size == 3072
    assert student_config.num_attention_heads == 12
    assert student_config.num_hidden_layers == 12
    assert student_config.vocab_size == 50272
    assert student_config.num_steps == 4


def test_student_embeddings_initialize_from_teacher_opt_space() -> None:
    from types import SimpleNamespace

    import torch

    from bispikclm.train.train_spad import TrainingConfig, build_student_from_teacher

    hidden_size = 4
    vocab_size = 8
    max_positions = 6
    teacher = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            ffn_dim=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=max_positions,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=2,
        ),
        model=SimpleNamespace(
            decoder=SimpleNamespace(
                embed_tokens=torch.nn.Embedding(vocab_size, hidden_size),
                embed_positions=torch.nn.Embedding(max_positions + 2, hidden_size),
            )
        ),
    )
    with torch.no_grad():
        teacher.model.decoder.embed_tokens.weight.copy_(
            torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size)
        )
        teacher.model.decoder.embed_positions.weight.copy_(
            torch.arange((max_positions + 2) * hidden_size, dtype=torch.float32).view(max_positions + 2, hidden_size)
        )

    student, _, _ = build_student_from_teacher(
        teacher,
        TrainingConfig(time_steps=2, teacher_model="facebook/opt-125m", sequence_length=max_positions),
    )

    assert torch.equal(student.model.token_embedding.weight, teacher.model.decoder.embed_tokens.weight)
    assert torch.equal(
        student.model.position_embeddings.weight,
        teacher.model.decoder.embed_positions.weight[2 : max_positions + 2],
    )
    assert student.lm_head.weight.data_ptr() == student.model.token_embedding.weight.data_ptr()


def test_sfsa_exposes_binary_qkv_and_uses_spike_domain_attention() -> None:
    import torch

    from bispikclm.models import BiSpikConfig
    from bispikclm.models.bispik_attention import BiSpikAttention

    config = BiSpikConfig(hidden_size=8, num_attention_heads=2, spike_threshold=0.2)
    attention = BiSpikAttention(config)
    hidden_state = torch.randn(2, 5, 8)

    output = attention(hidden_state, return_weights=True)

    for key in ("query_spikes", "key_spikes", "value_spikes", "attention_spikes", "context"):
        assert key in output
    for key in ("query_spikes", "key_spikes", "value_spikes", "attention_spikes"):
        tensor = output[key]
        assert torch.equal(tensor, tensor.bool().to(tensor.dtype))
    expected_int = torch.matmul(output["query_spikes"], output["key_spikes"].transpose(-1, -2))
    assert torch.equal(output["attention_int"], expected_int)


def test_sffn_lif_uses_configured_spike_threshold() -> None:
    from bispikclm.models import BiSpikConfig
    from bispikclm.models.bispik_mlp import BiSpikMLP

    config = BiSpikConfig(hidden_size=8, intermediate_size=16, spike_threshold=0.7)
    mlp = BiSpikMLP(config)

    assert float(mlp.lif.v_threshold) == 0.7


def test_spad_attention_and_feature_losses_include_rate_mse_branches() -> None:
    import torch

    from bispikclm.distill.spad import SpADConfig, SpADProjector, compute_multilevel_distillation

    config = SpADConfig(gamma_attn=0.5, gamma_feat=0.5)
    student_outputs = {
        "embedding_states": torch.randn(2, 2, 4, 8, requires_grad=True),
        "hidden_states": (
            torch.randn(2, 2, 4, 8, requires_grad=True),
            torch.randn(2, 2, 4, 8, requires_grad=True),
            torch.randn(2, 2, 4, 8, requires_grad=True),
        ),
        "attentions": (
            torch.randint(0, 2, (2, 2, 2, 4, 4), dtype=torch.float32).requires_grad_(),
            torch.randint(0, 2, (2, 2, 2, 4, 4), dtype=torch.float32).requires_grad_(),
        ),
        "logits": torch.randn(2, 4, 16, requires_grad=True),
    }
    teacher_outputs = {
        "hidden_states": (
            torch.randn(2, 4, 12),
            torch.randn(2, 4, 12),
            torch.randn(2, 4, 12),
            torch.randn(2, 4, 12),
        ),
        "attentions": (
            torch.rand(2, 2, 4, 4),
            torch.rand(2, 2, 4, 4),
            torch.rand(2, 2, 4, 4),
        ),
        "logits": torch.randn(2, 4, 16),
    }

    losses = compute_multilevel_distillation(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        config=config,
        labels=torch.randint(0, 16, (2, 4)),
        embedding_projector=SpADProjector(8, 12),
        hidden_projector=SpADProjector(8, 12),
        spike_threshold=1.0,
        membrane_decay=0.9,
    )

    assert "attention_rate_loss" in losses
    assert "attention_mse_loss" in losses
    assert "feature_rate_loss" in losses
    assert "feature_mse_loss" in losses
    losses["total_loss"].backward()
    assert student_outputs["logits"].grad is not None


def test_same_dimension_spad_projector_is_identity() -> None:
    import torch

    from bispikclm.distill.spad import SpADProjector

    projector = SpADProjector(4, 4)
    tensor = torch.randn(2, 3, 4)

    assert torch.equal(projector(tensor), tensor)
    assert list(projector.parameters()) == []


def test_trainable_parameter_detection_skips_identity_projectors() -> None:
    from bispikclm.distill.spad import SpADProjector
    from bispikclm.train.train_spad import _has_trainable_parameters

    assert not _has_trainable_parameters(SpADProjector(4, 4))
    assert _has_trainable_parameters(SpADProjector(4, 8))


def test_identity_projector_checkpoint_round_trip(tmp_path) -> None:
    import torch

    from bispikclm.distill.spad import SpADConfig, SpADProjector
    from bispikclm.models import BiSpikConfig, BiSpikForCausalLM
    from bispikclm.train.train_spad import DataConfig, ExperimentConfig, TrainingConfig, _save_checkpoint

    config = BiSpikConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
    )
    training = TrainingConfig(output_dir=str(tmp_path), sequence_length=8, time_steps=config.num_steps)
    student = BiSpikForCausalLM(config)
    embedding_projector = SpADProjector(config.hidden_size, config.hidden_size)
    hidden_projector = SpADProjector(config.hidden_size, config.hidden_size)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters()),
        lr=training.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"

    _save_checkpoint(
        path,
        student,
        embedding_projector,
        hidden_projector,
        optimizer,
        scheduler,
        3,
        ExperimentConfig(config, SpADConfig(), training, DataConfig()),
    )

    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["embedding_projector"] == {}
    assert checkpoint["hidden_projector"] == {}

    restored_embedding_projector = SpADProjector(config.hidden_size, config.hidden_size)
    restored_hidden_projector = SpADProjector(config.hidden_size, config.hidden_size)
    restored_embedding_projector.load_state_dict(checkpoint["embedding_projector"])
    restored_hidden_projector.load_state_dict(checkpoint["hidden_projector"])


def test_spad_attention_loss_penalizes_zero_student_attention_distribution() -> None:
    import torch

    from bispikclm.distill.spad import SpADConfig, compute_multilevel_distillation

    batch_size = 1
    heads = 2
    sequence_length = 64
    hidden_size = 8
    vocab_size = 16
    teacher_attention = torch.tril(torch.ones(sequence_length, sequence_length))
    teacher_attention = teacher_attention / teacher_attention.sum(dim=-1, keepdim=True)
    student_attention = torch.zeros(2, batch_size, heads, sequence_length, sequence_length, requires_grad=True)
    student_outputs = {
        "embedding_states": torch.zeros(2, batch_size, sequence_length, hidden_size, requires_grad=True),
        "hidden_states": (torch.zeros(2, batch_size, sequence_length, hidden_size, requires_grad=True),),
        "attentions": (student_attention,),
        "logits": torch.zeros(batch_size, sequence_length, vocab_size, requires_grad=True),
    }
    teacher_outputs = {
        "hidden_states": (torch.zeros(batch_size, sequence_length, hidden_size),),
        "attentions": (teacher_attention.expand(batch_size, heads, sequence_length, sequence_length),),
        "logits": torch.zeros(batch_size, sequence_length, vocab_size),
    }

    losses = compute_multilevel_distillation(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        config=SpADConfig(),
        labels=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        spike_threshold=1.0,
        membrane_decay=0.9,
    )

    assert losses["attention_loss"] > 0.05


def test_stable_gradient_clip_preserves_huge_finite_gradient_direction() -> None:
    import pytest
    import torch

    from bispikclm.train.train_spad import clip_grad_norm_stable

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    parameter.grad = torch.tensor([1e20, -1e20], dtype=torch.float32)

    total_norm = clip_grad_norm_stable([parameter], 0.7)

    assert torch.isfinite(total_norm)
    assert float(total_norm) == pytest.approx(2**0.5 * 1e20, rel=1e-6)
    assert torch.isfinite(parameter.grad).all()
    assert float(parameter.grad.norm()) == pytest.approx(0.7, rel=1e-5)
    assert parameter.grad[0] > 0
    assert parameter.grad[1] < 0

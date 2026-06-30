from pathlib import Path
import json

from bispikclm.distill.spad import SpADConfig, SpADProjector, compute_multilevel_distillation
from bispikclm.models.bispik_attention import BiSpikAttention
from bispikclm.models.bispik_block import BiSpikBlock
from bispikclm.models.bispik_config import BiSpikConfig
from bispikclm.models.bispik_lm import BiSpikForCausalLM
from bispikclm.train.eval_lm import main as eval_main
from bispikclm.train.train_spad import build_training_payload, freeze_teacher


def test_scaffold_smoke() -> None:
    config = BiSpikConfig(vocab_size=64, hidden_size=16, num_attention_heads=4, num_hidden_layers=2)
    model = BiSpikForCausalLM(config)

    assert model.config.hidden_size == 16
    assert eval_main(["--smoke-datasets"]) == 0
    assert not Path("bispikclm/cache").exists()


def test_zero_shot_eval_registry_contains_paper_tasks() -> None:
    from bispikclm.train.eval_lm import EVAL_TASKS

    assert set(EVAL_TASKS) == {
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "boolq",
        "piqa",
        "hellaswag",
        "openbookqa",
        "headqa",
    }


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
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        output_attentions=True,
        return_spike_stats=True,
    )
    minimal_output = model(input_ids=input_ids)

    assert isinstance(model.final_layer_norm, torch.nn.LayerNorm)
    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"] is not None
    assert output["loss"].ndim == 0
    assert len(output["hidden_states"]) == config.num_hidden_layers + 1
    assert output["hidden_states"][0].shape == (config.num_steps, 2, 8, config.hidden_size)
    assert not torch.allclose(output["hidden_states"][-1][0], output["hidden_states"][-1][1])
    assert len(output["attentions"]) == config.num_hidden_layers
    assert output["attentions"][0].shape == (config.num_steps, 2, config.num_attention_heads, 8, 8)
    assert torch.equal(output["attentions"][0], output["attentions"][0].bool().to(output["attentions"][0].dtype))
    assert len(output["spike_stats"]) == config.num_hidden_layers
    assert output["embedding_states"].shape == (config.num_steps, 2, 8, config.hidden_size)
    assert not torch.allclose(output["embedding_states"][0], output["embedding_states"][1])
    assert torch.count_nonzero(output["embedding_states"][:, :, 6:, :]) == 0

    assert minimal_output["hidden_states"] is None
    assert minimal_output["attentions"] is None
    assert minimal_output["spike_stats"] is None


def test_lm_uses_transformer_scale_embedding_initialization() -> None:
    config = BiSpikConfig(vocab_size=1024, hidden_size=64, num_attention_heads=4, num_hidden_layers=1)
    model = BiSpikForCausalLM(config)

    embedding_std = float(model.model.token_embedding.weight.detach().std())

    assert model.lm_head.weight.data_ptr() == model.model.token_embedding.weight.data_ptr()
    assert 0.01 < embedding_std < 0.08


def test_attention_path_is_tensor_native_and_softmax_free() -> None:
    import torch

    config = BiSpikConfig(hidden_size=16, num_attention_heads=4, num_steps=2)
    attention = BiSpikAttention(config)
    hidden_state = torch.randn(2, 8, 16)

    original_softmax = torch.softmax

    def fail_softmax(*args, **kwargs):
        raise AssertionError("BiSpikAttention must not call torch.softmax")

    torch.softmax = fail_softmax
    try:
        output = attention(hidden_state)
    finally:
        torch.softmax = original_softmax

    assert output["context"].shape == hidden_state.shape
    assert output["attention_scores"].shape == (2, config.num_attention_heads, 8, 8)
    assert output["attention_spikes"].shape == output["attention_scores"].shape
    assert torch.isfinite(output["context"]).all()


def test_attention_is_causal_and_respects_padding_mask() -> None:
    import torch

    config = BiSpikConfig(hidden_size=2, num_attention_heads=1)
    attention = BiSpikAttention(config)
    with torch.no_grad():
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj):
            projection.weight.copy_(torch.eye(config.hidden_size))

    hidden_state = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 0.0]]])
    causal_output = attention(hidden_state, return_weights=True)
    masked_output = attention(
        hidden_state,
        attention_mask=torch.tensor([[1, 0, 1]], dtype=torch.long),
        return_weights=True,
    )
    causal_weights = causal_output["attention_spikes"]
    masked_weights = masked_output["attention_spikes"]

    assert torch.allclose(causal_weights[0, 0].triu(diagonal=1), torch.zeros_like(causal_weights[0, 0].triu(diagonal=1)))
    assert torch.allclose(masked_weights[0, :, :, 1], torch.zeros_like(masked_weights[0, :, :, 1]))


def test_lm_loss_ignores_masked_positions() -> None:
    import torch
    import torch.nn.functional as F

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
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    shift_logits = output["logits"][..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().to(dtype=torch.bool)
    expected_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.masked_fill(~shift_mask, -100).reshape(-1),
        ignore_index=-100,
    )
    unmasked_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    assert output["loss"] is not None
    assert torch.isclose(output["loss"], expected_loss)
    assert not torch.isclose(output["loss"], unmasked_loss)


def test_lm_model_uses_bispik_block_stack() -> None:
    config = BiSpikConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4, num_hidden_layers=2)
    model = BiSpikForCausalLM(config)

    assert len(model.model.layers) == config.num_hidden_layers
    assert all(isinstance(layer, BiSpikBlock) for layer in model.model.layers)
    assert all(hasattr(layer, "attention_norm") for layer in model.model.layers)
    assert all(hasattr(layer, "mlp_norm") for layer in model.model.layers)


def test_block_tensor_path_uses_attention_then_mlp_residuals() -> None:
    import torch

    class AddOneAttention:
        config = BiSpikConfig(hidden_size=4, num_attention_heads=1)

        def __call__(self, hidden_state, **kwargs):
            del kwargs
            return {"context": hidden_state + 1.0, "attention_spikes": None}

    class AddTwoMLP:
        def __call__(self, hidden_state):
            return hidden_state + 2.0

    block = BiSpikBlock(AddOneAttention(), AddTwoMLP(), config=BiSpikConfig(hidden_size=4, num_attention_heads=1))
    block.attention_norm = torch.nn.Identity()
    block.mlp_norm = torch.nn.Identity()
    block.out_lif = torch.nn.Identity()
    hidden_state = torch.zeros(1, 2, 4)

    output = block(hidden_state)

    assert torch.equal(output, torch.full_like(hidden_state, 4.0))


def test_block_forward_rejects_shape_mismatch() -> None:
    class ShortMLP:
        def forward(self, hidden_state: list[float]) -> list[float]:
            return hidden_state[:-1]

    config = BiSpikConfig(hidden_size=4, num_attention_heads=1)
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
    assert serialized["distillation"]["loss_terms"] == ["EA", "SAA", "SFA", "STA", "HTA"]
    assert "train_loop" in serialized
    assert serialized["train_loop"]["optimizer"] == "torch.optim.Adam"
    assert serialized["train_loop"]["scheduler"] == "cosine_decay"
    assert "runtime_requirements" in serialized


def test_spad_five_loss_backward_with_temporal_fusion() -> None:
    import torch

    config = SpADConfig()
    student_outputs = {
        "embedding_states": torch.randn(2, 2, 4, 8, requires_grad=True),
        "hidden_states": (torch.randn(2, 2, 4, 8, requires_grad=True), torch.randn(2, 2, 4, 8, requires_grad=True)),
        "attentions": (torch.randint(0, 2, (2, 2, 2, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": torch.randn(2, 4, 16, requires_grad=True),
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(2, 4, 12), torch.randn(2, 4, 12)),
        "attentions": (torch.rand(2, 2, 4, 4),),
        "logits": torch.randn(2, 4, 16),
    }
    labels = torch.randint(0, 16, (2, 4))

    losses = compute_multilevel_distillation(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        config=config,
        labels=labels,
        embedding_projector=SpADProjector(8, 12),
        hidden_projector=SpADProjector(8, 12),
    )
    losses["total_loss"].backward()

    assert set(losses) >= {"embedding_loss", "attention_loss", "feature_loss", "soft_loss", "hard_loss", "total_loss"}
    assert losses["total_loss"].ndim == 0
    assert student_outputs["logits"].grad is not None


def test_spad_hard_loss_uses_next_token_shift() -> None:
    import torch
    import torch.nn.functional as F

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=0.0, lambda_hard=1.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    student_outputs = {
        "embedding_states": torch.randn(2, 1, 4, 4, requires_grad=True),
        "hidden_states": (torch.randn(2, 1, 4, 4, requires_grad=True),),
        "attentions": (torch.randint(0, 2, (2, 1, 1, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": student_logits,
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(1, 4, 4),),
        "attentions": (torch.rand(1, 1, 4, 4),),
        "logits": torch.randn(1, 4, 8),
    }
    labels = torch.tensor([[1, 2, 3, 4]])

    losses = compute_multilevel_distillation(student_outputs, teacher_outputs, config, labels=labels)
    shifted = F.cross_entropy(student_logits[..., :-1, :].reshape(-1, 8), labels[..., 1:].reshape(-1), ignore_index=-100)
    unshifted = F.cross_entropy(student_logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100)

    assert torch.isclose(losses["hard_loss"], shifted)
    assert not torch.isclose(losses["hard_loss"], unshifted)


def test_spad_soft_loss_is_token_averaged() -> None:
    import torch

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=1.0, lambda_hard=0.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    teacher_logits = torch.randn(1, 4, 8)

    def outputs(logits: torch.Tensor) -> dict[str, object]:
        return {
            "embedding_states": torch.randn(2, logits.shape[0], logits.shape[1], 4, requires_grad=True),
            "hidden_states": (torch.randn(2, logits.shape[0], logits.shape[1], 4, requires_grad=True),),
            "attentions": (
                torch.randint(0, 2, (2, logits.shape[0], 1, logits.shape[1], logits.shape[1]), dtype=torch.float32).requires_grad_(),
            ),
            "logits": logits,
        }

    def teacher(logits: torch.Tensor) -> dict[str, object]:
        return {
            "hidden_states": (torch.randn(logits.shape[0], logits.shape[1], 4),),
            "attentions": (torch.rand(logits.shape[0], 1, logits.shape[1], logits.shape[1]),),
            "logits": logits,
        }

    base = compute_multilevel_distillation(
        outputs(student_logits),
        teacher(teacher_logits),
        config,
        labels=torch.ones(1, 4, dtype=torch.long),
    )["soft_loss"]
    repeated = compute_multilevel_distillation(
        outputs(torch.cat([student_logits, student_logits], dim=1)),
        teacher(torch.cat([teacher_logits, teacher_logits], dim=1)),
        config,
        labels=torch.tensor([[1, 1, 1, 1, -100, 1, 1, 1]]),
    )["soft_loss"]

    assert torch.isclose(repeated, base, rtol=1e-5, atol=1e-5)


def test_spad_soft_loss_uses_next_token_mask() -> None:
    import torch
    import torch.nn.functional as F

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=1.0, lambda_hard=0.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    teacher_logits = torch.randn(1, 4, 8)

    student_outputs = {
        "embedding_states": torch.randn(2, 1, 4, 4, requires_grad=True),
        "hidden_states": (torch.randn(2, 1, 4, 4, requires_grad=True),),
        "attentions": (torch.randint(0, 2, (2, 1, 1, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": student_logits,
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(1, 4, 4),),
        "attentions": (torch.rand(1, 1, 4, 4),),
        "logits": teacher_logits,
    }
    labels = torch.tensor([[1, 2, -100, 4]])

    losses = compute_multilevel_distillation(student_outputs, teacher_outputs, config, labels=labels)
    token_kl = F.kl_div(
        F.log_softmax(student_logits[..., :-1, :] / config.temperature, dim=-1),
        F.softmax(teacher_logits[..., :-1, :] / config.temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = token_kl.masked_select(labels[..., 1:].ne(-100)).mean() * (config.temperature**2)

    assert torch.isclose(losses["soft_loss"], expected)


def test_spad_soft_only_loss_respects_attention_mask() -> None:
    import torch
    import torch.nn.functional as F

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=1.0, lambda_hard=0.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    teacher_logits = torch.randn(1, 4, 8)
    student_outputs = {
        "embedding_states": torch.randn(2, 1, 4, 4, requires_grad=True),
        "hidden_states": (torch.randn(2, 1, 4, 4, requires_grad=True),),
        "attentions": (torch.randint(0, 2, (2, 1, 1, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": student_logits,
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(1, 4, 4),),
        "attentions": (torch.rand(1, 1, 4, 4),),
        "logits": teacher_logits,
    }
    attention_mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.long)

    losses = compute_multilevel_distillation(
        student_outputs,
        teacher_outputs,
        config,
        labels=None,
        attention_mask=attention_mask,
    )
    token_kl = F.kl_div(
        F.log_softmax(student_logits[..., :-1, :] / config.temperature, dim=-1),
        F.softmax(teacher_logits[..., :-1, :] / config.temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = token_kl.masked_select(attention_mask[..., 1:].bool()).mean() * (config.temperature**2)

    assert torch.isclose(losses["soft_loss"], expected)


def test_spad_soft_loss_is_zero_when_all_targets_are_ignored() -> None:
    import torch

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=1.0, lambda_hard=0.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    student_outputs = {
        "embedding_states": torch.randn(2, 1, 4, 4, requires_grad=True),
        "hidden_states": (torch.randn(2, 1, 4, 4, requires_grad=True),),
        "attentions": (torch.randint(0, 2, (2, 1, 1, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": student_logits,
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(1, 4, 4),),
        "attentions": (torch.rand(1, 1, 4, 4),),
        "logits": torch.randn(1, 4, 8),
    }

    losses = compute_multilevel_distillation(
        student_outputs,
        teacher_outputs,
        config,
        labels=torch.full((1, 4), -100),
    )

    assert losses["soft_loss"].item() == 0.0
    assert losses["hard_loss"].item() == 0.0
    assert losses["total_loss"].item() == 0.0


def test_spad_attention_mask_ignores_padding_targets() -> None:
    import torch

    config = SpADConfig(lambda_emb=0.0, lambda_attn=0.0, lambda_feat=0.0, lambda_soft=1.0, lambda_hard=1.0)
    student_logits = torch.randn(1, 4, 8, requires_grad=True)
    student_outputs = {
        "embedding_states": torch.randn(2, 1, 4, 4, requires_grad=True),
        "hidden_states": (torch.randn(2, 1, 4, 4, requires_grad=True),),
        "attentions": (torch.randint(0, 2, (2, 1, 1, 4, 4), dtype=torch.float32).requires_grad_(),),
        "logits": student_logits,
    }
    teacher_outputs = {
        "hidden_states": (torch.randn(1, 4, 4),),
        "attentions": (torch.rand(1, 1, 4, 4),),
        "logits": torch.randn(1, 4, 8),
    }

    losses = compute_multilevel_distillation(
        student_outputs,
        teacher_outputs,
        config,
        labels=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.zeros(1, 4, dtype=torch.long),
    )

    assert losses["soft_loss"].item() == 0.0
    assert losses["hard_loss"].item() == 0.0


def test_training_logs_average_accumulated_microbatches() -> None:
    from bispikclm.train.train_spad import average_loss_snapshots

    averaged = average_loss_snapshots(
        [
            {"total_loss": 12.0, "hard_loss": 10.0},
            {"total_loss": 6.0, "hard_loss": 8.0},
            {"total_loss": 3.0, "hard_loss": 5.0},
        ]
    )

    assert averaged == {"total_loss": 7.0, "hard_loss": 23.0 / 3.0}


def test_lm_monitoring_metrics_include_logit_scale_and_valid_tokens() -> None:
    import torch

    from bispikclm.train.train_spad import compute_lm_monitoring_metrics

    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, -100]])
    attention_mask = torch.tensor([[1, 1, 0]])
    hard_loss = torch.tensor(2.0)

    metrics = compute_lm_monitoring_metrics(logits, labels, attention_mask, hard_loss)

    assert metrics["train/valid_tokens"] == 1.0
    assert metrics["train/logit_abs_max"] == 2.0
    assert "train/logit_mean" in metrics
    assert "train/logit_std" in metrics


def test_lm_monitoring_metrics_report_alignment_diagnostics() -> None:
    import torch

    from bispikclm.train.train_spad import compute_lm_monitoring_metrics

    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0, 2.0],
                [4.0, 1.0, 3.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    teacher_logits = torch.tensor(
        [
            [
                [0.0, 5.0, 1.0, 2.0],
                [6.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, 2]])
    attention_mask = torch.ones_like(labels)
    hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [0.0, 0.0]]])
    spike_stats = [{"spike_rate": torch.tensor(0.25)}, {"spike_rate": torch.tensor(0.75)}]

    metrics = compute_lm_monitoring_metrics(
        logits,
        labels,
        attention_mask,
        torch.tensor(2.0),
        teacher_logits=teacher_logits,
        hidden_state=hidden_state,
        spike_stats=spike_stats,
        readout_scale=torch.tensor(2.0),
    )

    assert metrics["train/target_rank_mean"] == 1.5
    assert metrics["train/target_margin_mean"] == 0.0
    assert metrics["train/top5_accuracy"] == 1.0
    assert metrics["train/teacher_top1_agreement"] == 1.0
    assert metrics["train/spike_rate_mean"] == 0.5
    assert metrics["train/readout_scale"] == 2.0
    assert "train/hidden_std" in metrics


def test_freeze_teacher_disables_gradients() -> None:
    import torch

    teacher = torch.nn.Linear(4, 4)

    frozen = freeze_teacher(teacher)

    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_readme_training_command_matches_cli() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m bispikclm.train.train_spad" in readme
    assert "--teacher-model" in readme
    assert "--dummy-batch" in readme
    assert "--learning-rate" in readme
    assert "scripts/run_sft.sh" in readme


def test_run_sft_script_does_not_override_config_defaults() -> None:
    import os
    import subprocess

    env = os.environ.copy()
    env["CONFIG"] = "configs/bispikclm_opt350m_spad.toml"
    env["TRAIN_MODE"] = "--dry-run"
    env["NPROC_PER_NODE"] = "1"
    env["PATH"] = f"{Path('.venv/bin').resolve()}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", "scripts/run_sft.sh"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "'teacher_model_id': 'facebook/opt-350m'" in result.stdout
    assert "'model_name': 'facebook/opt-350m'" in result.stdout

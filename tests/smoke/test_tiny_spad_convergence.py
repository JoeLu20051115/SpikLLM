from __future__ import annotations


def test_tiny_spad_overfits_fixed_batch() -> None:
    import torch

    from bispikclm.distill.spad import SpADConfig, SpADProjector, compute_multilevel_distillation
    from bispikclm.models import BiSpikConfig, BiSpikForCausalLM

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BiSpikConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=32,
        num_steps=2,
        spike_threshold=0.5,
        membrane_decay=0.9,
    )
    student = BiSpikForCausalLM(config).to(device)
    embedding_projector = SpADProjector(config.hidden_size, config.hidden_size).to(device)
    hidden_projector = SpADProjector(config.hidden_size, config.hidden_size).to(device)
    trainable_parameters = list(student.parameters()) + list(embedding_projector.parameters()) + list(hidden_projector.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=3e-3)
    distill_config = SpADConfig()

    input_ids = torch.randint(4, config.vocab_size, (2, 12), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    teacher_logits = torch.full((2, 12, config.vocab_size), -6.0, device=device)
    teacher_logits[..., :-1, :].scatter_(2, labels[..., 1:].unsqueeze(-1), 6.0)
    teacher_logits[..., -1, :] = 0.0
    causal_attention = torch.tril(torch.ones(12, 12, device=device))
    causal_attention = causal_attention / causal_attention.sum(dim=-1, keepdim=True)
    teacher_outputs = {
        "hidden_states": tuple(torch.randn(2, 12, config.hidden_size, device=device) * 0.2 for _ in range(3)),
        "attentions": tuple(causal_attention.expand(2, config.num_attention_heads, 12, 12).clone() for _ in range(3)),
        "logits": teacher_logits,
    }

    history: list[dict[str, float]] = []
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        losses = compute_multilevel_distillation(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            config=distill_config,
            labels=labels,
            attention_mask=attention_mask,
            embedding_projector=embedding_projector,
            hidden_projector=hidden_projector,
            spike_threshold=config.spike_threshold,
            membrane_decay=config.membrane_decay,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        history.append({name: float(value.detach().cpu()) for name, value in losses.items()})

    tracked = ("attention_loss", "soft_loss", "hard_loss", "total_loss")
    assert history[-1]["hard_loss"] < 4.0
    assert history[-1]["soft_loss"] < 5.0
    assert all(history[-1][name] < history[0][name] for name in tracked), history


def test_dummy_batch_reports_initial_final_and_delta(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    import torch
    from torch import nn

    from bispikclm.train import train_spad

    class TinyTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                vocab_size=64,
                hidden_size=16,
                ffn_dim=32,
                num_attention_heads=4,
                num_hidden_layers=1,
                max_position_embeddings=32,
                pad_token_id=1,
                bos_token_id=2,
                eos_token_id=2,
            )

        def forward(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            batch, length = input_ids.shape
            hidden = torch.zeros(batch, length, self.config.hidden_size, device=input_ids.device)
            attention = torch.tril(torch.ones(length, length, device=input_ids.device))
            attention = attention / attention.sum(dim=-1, keepdim=True)
            logits = torch.zeros(batch, length, self.config.vocab_size, device=input_ids.device)
            logits[..., :-1, :].scatter_(2, input_ids[..., 1:].unsqueeze(-1), 6.0)
            return SimpleNamespace(
                hidden_states=(hidden, hidden),
                attentions=(attention.expand(batch, self.config.num_attention_heads, length, length),),
                logits=logits,
            )

    monkeypatch.setattr(train_spad, "load_teacher", lambda model_name, device: TinyTeacher().to(device))

    result = train_spad.train_dummy_batch(
        train_spad.TrainingConfig(
            teacher_model="tiny",
            output_dir=str(tmp_path),
            learning_rate=1e-3,
            batch_size=2,
            max_steps=3,
            sequence_length=8,
            time_steps=2,
            gradient_accumulation_steps=1,
            precision="fp32",
        )
    )

    assert "initial_total_loss" in result
    assert "final_total_loss" in result
    assert "delta_total_loss" in result
    assert result["steps"] == 3.0
    for name in ("embedding_loss", "attention_loss", "feature_loss", "soft_loss", "hard_loss", "total_loss"):
        assert f"initial_{name}" in result
        assert f"final_{name}" in result
        assert f"delta_{name}" in result
        assert result[f"final_{name}"] == result[name]
        assert result[f"delta_{name}"] == result[f"final_{name}"] - result[f"initial_{name}"]

# BiSpikCLM Training Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current `SpikLLM` scaffold into a paper-faithful, directly trainable offline `BiSpikCLM` distillation implementation that launches with single-node multi-GPU `torchrun`.

**Architecture:** Keep the existing package layout, but replace placeholder internals with a real tensor-native student model, a tensor-native `SpAD` loss path, a real dataset loader, and a native PyTorch DDP trainer. Reuse the current file paths where practical so the repo stays easy to diff against the scaffold.

**Tech Stack:** `torch`, `torch.distributed`, `transformers`, `datasets`, `huggingface_hub`, `pytest`

## Global Constraints

- Implement the `BiSpikCLM` paper body only.
- Do not implement `SA-TOPD` or any online post-training stage.
- Target single-node multi-GPU training as the minimum acceptance bar.
- Use native PyTorch DDP first; do not add `accelerate`, `deepspeed`, or `lightning` unless DDP blocks launch.
- Keep the training path tensor-native; remove placeholder `list[float]` logic from the critical path.
- Keep review gates aligned to the design spec: model interface, `SFSA`, `SpAD`, data, DDP trainer, verification/docs.

---

## File Structure Map

- `bispikclm/models/bispik_config.py`
  Owns model hyperparameters and training-surface flags shared across the student.
- `bispikclm/models/bispik_attention.py`
  Owns softmax-free spiking attention and attention-side distillation outputs.
- `bispikclm/models/bispik_mlp.py`
  Owns the block feed-forward path.
- `bispikclm/models/bispik_block.py`
  Owns one transformer block and block-level feature packaging.
- `bispikclm/models/bispik_model.py`
  Owns embeddings, block stack, hidden-state collection, and output assembly.
- `bispikclm/models/bispik_lm.py`
  Owns LM head, causal-LM loss, and the public forward contract.
- `bispikclm/distill/spad.py`
  Owns tensor-native multi-level distillation loss.
- `bispikclm/distill/hooks.py`
  Owns teacher/student feature extraction helpers.
- `bispikclm/data/fineweb.py`
  Owns dataset loading, tokenization, packing, and collator helpers for the first trainable corpus path.
- `bispikclm/train/train_spad.py`
  Owns CLI parsing, DDP setup, teacher/student initialization, train loop, checkpointing, and smoke launch path.
- `tests/smoke/test_scaffold.py`
  Replace scaffold assertions with real tensor-path smoke coverage.
- `README.md`
  Owns user-facing setup and direct training command.
- `pyproject.toml`
  Owns runtime dependencies.

### Task 1: Repair the model interface and tensor-return contract

**Files:**
- Modify: `bispikclm/models/bispik_config.py`
- Modify: `bispikclm/models/bispik_model.py`
- Modify: `bispikclm/models/bispik_lm.py`
- Modify: `bispikclm/models/__init__.py`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `BiSpikConfig`
- Produces: `BiSpikForCausalLM.forward(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, output_hidden_states: bool = False, output_attentions: bool = False, return_spike_stats: bool = False) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[dict[str, torch.Tensor]] | None]`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features -v`
Expected: FAIL because the current model does not implement a callable tensor-native forward contract with hidden-state and attention outputs.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class BiSpikConfig:
    vocab_size: int = 50272
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048
    num_steps: int = 4
    spike_threshold: float = 1.0
    membrane_decay: float = 0.9
    teacher_model_id: str = "facebook/opt-125m"

class BiSpikForCausalLM(nn.Module):
    def forward(...):
        model_output = self.model(...)
        logits = self.lm_head(model_output["last_hidden_state"])
        return {
            "logits": logits,
            "hidden_states": model_output.get("hidden_states"),
            "attentions": model_output.get("attentions"),
            "spike_stats": model_output.get("spike_stats"),
            "embedding_states": model_output.get("embedding_states"),
            "loss": loss,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_lm_forward_returns_tensor_features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add bispikclm/models/bispik_config.py bispikclm/models/bispik_model.py bispikclm/models/bispik_lm.py bispikclm/models/__init__.py tests/smoke/test_scaffold.py
git -C SpikLLM commit -m "feat: expose tensor-native bispikclm outputs"
```

### Task 2: Replace placeholder attention and block logic with softmax-free spiking execution

**Files:**
- Modify: `bispikclm/models/bispik_attention.py`
- Modify: `bispikclm/models/bispik_block.py`
- Modify: `bispikclm/models/bispik_mlp.py`
- Modify: `bispikclm/models/bispik_model.py`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `BiSpikConfig`, `BiSpikForCausalLM.forward(...)`
- Produces: `BiSpikAttention.forward(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write the failing test**

```python
def test_attention_path_is_tensor_native_and_softmax_free() -> None:
    import torch

    config = BiSpikConfig(hidden_size=16, num_attention_heads=4, num_steps=2)
    attention = BiSpikAttention(config)
    hidden_state = torch.randn(2, 8, 16)

    output = attention(hidden_state)

    assert output["context"].shape == hidden_state.shape
    assert output["attention_scores"].shape[:3] == (2, config.num_attention_heads, 8)
    assert torch.isfinite(output["context"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_attention_path_is_tensor_native_and_softmax_free -v`
Expected: FAIL because the current attention class is not an `nn.Module` and still runs `torch.softmax(...)`.

- [ ] **Step 3: Write minimal implementation**

```python
class BiSpikAttention(nn.Module):
    def forward(self, hidden_state, attention_mask=None):
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / max(self.head_dim, 1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, 0.0)
        spikes = (scores > self.config.spike_threshold).to(scores.dtype)
        context = torch.matmul(spikes, value)
        return {
            "context": self.out_proj(context),
            "attention_scores": scores,
            "attention_spikes": spikes,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_attention_path_is_tensor_native_and_softmax_free -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add bispikclm/models/bispik_attention.py bispikclm/models/bispik_block.py bispikclm/models/bispik_mlp.py bispikclm/models/bispik_model.py tests/smoke/test_scaffold.py
git -C SpikLLM commit -m "feat: add softmax-free spiking block path"
```

### Task 3: Implement tensor-native Spike-Aware Alignment Distillation

**Files:**
- Modify: `bispikclm/distill/spad.py`
- Modify: `bispikclm/distill/hooks.py`
- Modify: `bispikclm/models/bispik_lm.py`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `teacher_outputs: dict[str, torch.Tensor | tuple[torch.Tensor, ...]]`, `student_outputs: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[dict[str, torch.Tensor]]]`
- Produces: `compute_multilevel_distillation(student_outputs: dict[str, object], teacher_outputs: dict[str, object], config: SpADConfig) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write the failing test**

```python
def test_multilevel_spad_returns_tensor_losses() -> None:
    import torch

    config = SpADConfig()
    teacher_outputs = {
        "embedding_states": torch.randn(2, 8, 16),
        "hidden_states": (torch.randn(2, 8, 16), torch.randn(2, 8, 16)),
        "attentions": (torch.randn(2, 4, 8, 8),),
        "logits": torch.randn(2, 8, 32),
    }
    student_outputs = {
        "embedding_states": torch.randn(2, 8, 16),
        "hidden_states": (torch.randn(2, 8, 16), torch.randn(2, 8, 16)),
        "attentions": (torch.randn(2, 4, 8, 8),),
        "logits": torch.randn(2, 8, 32),
    }

    losses = compute_multilevel_distillation(student_outputs, teacher_outputs, config)

    assert set(losses) >= {"embedding_loss", "attention_loss", "hidden_loss", "logit_loss", "total_loss"}
    assert losses["total_loss"].ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_multilevel_spad_returns_tensor_losses -v`
Expected: FAIL because the current loss path only accepts `list[float]`.

- [ ] **Step 3: Write minimal implementation**

```python
def compute_multilevel_distillation(student_outputs, teacher_outputs, config):
    embedding_loss = F.mse_loss(student_outputs["embedding_states"], teacher_outputs["embedding_states"])
    hidden_loss = sum(F.mse_loss(s, t) for s, t in zip(student_outputs["hidden_states"], teacher_outputs["hidden_states"]))
    attention_loss = sum(F.mse_loss(s, t) for s, t in zip(student_outputs["attentions"], teacher_outputs["attentions"]))
    student_log_probs = F.log_softmax(student_outputs["logits"] / config.temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs["logits"] / config.temperature, dim=-1)
    logit_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    total_loss = (
        config.embedding_weight * embedding_loss
        + config.hidden_weight * hidden_loss
        + config.attention_weight * attention_loss
        + config.logit_weight * logit_loss
    )
    return {...}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_multilevel_spad_returns_tensor_losses -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add bispikclm/distill/spad.py bispikclm/distill/hooks.py bispikclm/models/bispik_lm.py tests/smoke/test_scaffold.py
git -C SpikLLM commit -m "feat: implement tensor-native spad losses"
```

### Task 4: Replace manifest stubs with a real dataset pipeline

**Files:**
- Modify: `bispikclm/data/fineweb.py`
- Modify: `bispikclm/train/train_spad.py`
- Modify: `pyproject.toml`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `teacher_model_id: str`, `dataset_name: str`, `dataset_config: str | None`, `sequence_length: int`
- Produces: `build_language_modeling_datasets(dataset_name: str, dataset_config: str | None, tokenizer_name: str, sequence_length: int, train_split: str = "train", validation_split: str | None = None) -> dict[str, Dataset]`

- [ ] **Step 1: Write the failing test**

```python
def test_dataset_pipeline_builds_token_blocks(monkeypatch) -> None:
    from bispikclm.data.fineweb import build_language_modeling_datasets

    datasets = build_language_modeling_datasets(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        tokenizer_name="facebook/opt-125m",
        sequence_length=8,
    )

    sample = datasets["train"][0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert len(sample["input_ids"]) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_dataset_pipeline_builds_token_blocks -v`
Expected: FAIL because the current data module only writes JSON manifests.

- [ ] **Step 3: Write minimal implementation**

```python
def build_language_modeling_datasets(...):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    dataset = load_dataset(dataset_name, dataset_config)
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)
    packed = tokenized.map(pack_blocks, batched=True)
    return {"train": packed["train"], "validation": packed.get("validation")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_dataset_pipeline_builds_token_blocks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add bispikclm/data/fineweb.py bispikclm/train/train_spad.py pyproject.toml tests/smoke/test_scaffold.py
git -C SpikLLM commit -m "feat: add real tokenized dataset pipeline"
```

### Task 5: Convert the dry-run entrypoint into a real single-node multi-GPU trainer

**Files:**
- Modify: `bispikclm/train/train_spad.py`
- Modify: `bispikclm/models/__init__.py`
- Modify: `bispikclm/distill/__init__.py`
- Test: `tests/smoke/test_scaffold.py`

**Interfaces:**
- Consumes: `build_language_modeling_datasets(...)`, `BiSpikForCausalLM`, `compute_multilevel_distillation(...)`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

```python
def test_training_main_runs_single_process_smoke(tmp_path) -> None:
    from bispikclm.train.train_spad import main

    exit_code = main(
        [
            "--dataset-name", "wikitext",
            "--dataset-config", "wikitext-2-raw-v1",
            "--teacher-model", "facebook/opt-125m",
            "--output-dir", str(tmp_path),
            "--max-steps", "1",
            "--per-device-batch-size", "1",
            "--sequence-length", "8",
            "--smoke-run",
        ]
    )

    assert exit_code == 0
    assert any(path.name.endswith(".pt") for path in tmp_path.iterdir())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_training_main_runs_single_process_smoke -v`
Expected: FAIL because the current entrypoint only builds a JSON payload.

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_distributed_if_needed(args)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model).eval()
    freeze_module(teacher)
    student = build_student_from_args(args).to(device)
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank])
    train_loader = build_train_dataloader(args)
    train_one_epoch(...)
    save_checkpoint(args.output_dir, student, optimizer, global_step)
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_training_main_runs_single_process_smoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add bispikclm/train/train_spad.py bispikclm/models/__init__.py bispikclm/distill/__init__.py tests/smoke/test_scaffold.py
git -C SpikLLM commit -m "feat: add ddp spad training entrypoint"
```

### Task 6: Final verification, README repair, and smoke coverage cleanup

**Files:**
- Modify: `README.md`
- Modify: `tests/smoke/test_scaffold.py`
- Modify: `bispikclm/train/eval_lm.py`

**Interfaces:**
- Consumes: `main(argv: list[str] | None = None) -> int`
- Produces: user-facing launch commands and final smoke assertions

- [ ] **Step 1: Write the failing test**

```python
def test_readme_training_command_matches_cli() -> None:
    from pathlib import Path

    readme = Path("SpikLLM/README.md").read_text(encoding="utf-8")

    assert "torchrun --nproc_per_node=" in readme
    assert "--dataset-name" in readme
    assert "--teacher-model" in readme
    assert "--output-dir" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py::test_readme_training_command_matches_cli -v`
Expected: FAIL because the current README only describes a prelaunch scaffold.

- [ ] **Step 3: Write minimal implementation**

```markdown
# SpikLLM

## Train BiSpikCLM

```bash
torchrun --nproc_per_node=4 -m bispikclm.train.train_spad \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --teacher-model facebook/opt-125m \
  --output-dir checkpoints/bispikclm \
  --sequence-length 128 \
  --per-device-batch-size 2 \
  --max-steps 100
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest SpikLLM/tests/smoke/test_scaffold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C SpikLLM add README.md tests/smoke/test_scaffold.py bispikclm/train/eval_lm.py
git -C SpikLLM commit -m "docs: document direct bispikclm training"
```

## Self-Review

### Spec coverage

- Paper-faithful student model surface: Task 1
- Softmax-free spiking attention and block repair: Task 2
- Offline `SpAD`: Task 3
- Real data loading: Task 4
- Single-node multi-GPU `torchrun` path: Task 5
- Smoke verification and README launch instructions: Task 6

No spec requirement is left without a task.

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Every task includes explicit files, interface names, test commands, and a concrete code direction.

### Type consistency

- `BiSpikForCausalLM.forward(...)` is the single public student entrypoint referenced by later tasks.
- `compute_multilevel_distillation(...)` is introduced in Task 3 and consumed unchanged in Task 5.
- `build_language_modeling_datasets(...)` is introduced in Task 4 and consumed unchanged in Task 5.


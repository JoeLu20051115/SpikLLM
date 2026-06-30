# BiSpikCLM Paper-Faithful Training Design

## Goal

Repair the current `SpikLLM` scaffold into a paper-faithful, directly trainable implementation of `BiSpikCLM: A Spiking Language Model integrating Softmax-Free Spiking Attention and Spike-Aware Alignment Distillation`.

This scope is intentionally strict:

- Implement the `BiSpikCLM` paper body only.
- Do not implement `SA-TOPD` or any online post-training stage.
- Target single-node multi-GPU training as the minimum acceptance bar.

## Non-Goals

- No second-stage online distillation.
- No reproduction of every paper experiment before training works.
- No extra training framework unless plain `torchrun` becomes a blocker.
- No speculative abstractions for future methods.

## Current State

The repository is a smoke-tested bootstrap scaffold, not a training-ready implementation.

Known gaps:

- `bispikclm/models/bispik_attention.py` still uses standard softmax attention in the tensor path.
- `bispikclm/distill/spad.py` is a placeholder list-based loss calculator, not a tensor distillation module.
- `bispikclm/train/train_spad.py` is a dry-run planner, not a train loop.
- `bispikclm/data/fineweb.py` only writes manifests and caches limited teacher metadata.
- The student LM interface is not yet a real language-model training surface.

## Acceptance Criteria

The implementation is considered complete when all of the following are true:

1. `torchrun --nproc_per_node=N -m bispikclm.train.train_spad ...` initializes successfully on a single node with multiple GPUs.
2. The training path uses real `torch` tensors, a real Hugging Face causal-LM teacher, and a real dataset loader.
3. One or more training steps complete with forward pass, SpAD loss computation, backward pass, optimizer step, and checkpoint save.
4. The repository contains at least one smoke test that exercises the real tensor training path.
5. The README contains a direct training command and minimal setup notes.

## Architecture

### Models

`bispikclm/models/` remains the home of the student implementation.

Required modules:

- `BiSpikConfig`: expand to include training-relevant model hyperparameters needed by the paper-faithful student.
- `BiSpikModel`: embedding stack, position handling, block stack, and output collection for distillation.
- `BiSpikBlock`: one spiking transformer block with paper-aligned attention and MLP submodules.
- `BiSpikAttention`: replace the current softmax-based path with a softmax-free spiking attention implementation.
- `BiSpikMLP`: preserve or repair as needed so the block remains trainable under real tensor execution.
- `BiSpikForCausalLM`: expose logits plus optional distillation features in a single forward pass.

The student forward interface should support:

- `input_ids`
- optional `attention_mask`
- `labels`
- `output_hidden_states`
- `output_attentions`
- `return_spike_stats`

The returned object may be a dataclass or a dict, but it must consistently expose the tensors required by `SpAD`.

### Distillation

`bispikclm/distill/` will implement the paper's offline `Spike-Aware Alignment Distillation`.

Required pieces:

- `SpADConfig`: scalar weights and temperatures for each loss term.
- Feature collectors or explicit outputs for:
  - embedding alignment
  - attention alignment
  - hidden-state alignment
  - logit distillation
- A tensor-native `compute_multilevel_distillation(...)` that consumes teacher and student outputs and returns per-term losses plus total loss.

The distillation module should operate on explicit tensor dictionaries instead of implicit Python lists.

### Data

`bispikclm/data/` will stop pretending that manifests equal readiness.

Required pieces:

- Hugging Face dataset loading through `datasets`
- teacher tokenizer loading through `transformers`
- tokenization and sequence blocking
- a collator for causal-LM batches
- minimal configuration for train and optional validation splits

The first version should optimize for reliability over completeness. A single training corpus path that can launch real training is enough.

### Training

`bispikclm/train/train_spad.py` becomes the real offline distillation entrypoint.

Required behavior:

- initialize distributed training from `torchrun`
- load and freeze the teacher
- build the student
- build dataloaders
- run mixed-precision training when requested
- compute teacher outputs and student outputs on the same batch
- compute SpAD loss and optional next-token CE
- backpropagate student gradients only
- save checkpoints
- log minimal training progress

The initial implementation should use native PyTorch DDP. Do not introduce `accelerate`, `deepspeed`, or `lightning` unless DDP blocks the acceptance criteria.

## Training Step Design

One training step follows this exact shape:

1. Load a batch of tokenized text blocks.
2. Feed the batch into the frozen ANN teacher.
3. Feed the same batch into the trainable `BiSpikCLM` student.
4. Collect teacher and student tensors for:
   - embedding-level alignment
   - attention-level alignment
   - hidden-level alignment
   - final logits
5. Compute `SpAD` total loss.
6. Optionally add standard next-token cross-entropy as a stabilizer.
7. Backpropagate through the student only.
8. Apply optimizer step, scheduler step if configured, and zero gradients.

## Multi-GPU Design

The minimum supported launch mode is single-node multi-GPU.

Implementation rules:

- Launch with `torchrun`.
- Use `torch.distributed.init_process_group`.
- Wrap the student in `DistributedDataParallel`.
- Use `DistributedSampler` for training data.
- Ensure checkpoint writes happen only on rank 0.
- Keep teacher loading deterministic across ranks.

This is enough for the current goal. Multi-node support is out of scope.

## Dependency Policy

The implementation may add dependencies, but only if they directly shorten the path to a trainable paper-faithful result.

Expected dependency set:

- `torch`
- `transformers`
- `datasets`
- `huggingface_hub`

Avoid adding orchestration frameworks during this repair pass.

## File-Level Change Plan

Primary files expected to change:

- `bispikclm/models/bispik_config.py`
- `bispikclm/models/bispik_attention.py`
- `bispikclm/models/bispik_block.py`
- `bispikclm/models/bispik_model.py`
- `bispikclm/models/bispik_lm.py`
- `bispikclm/models/bispik_mlp.py`
- `bispikclm/distill/spad.py`
- `bispikclm/distill/hooks.py`
- `bispikclm/data/fineweb.py` or a renamed data module if the current filename becomes misleading
- `bispikclm/train/train_spad.py`
- `README.md`
- `pyproject.toml`
- smoke tests under `tests/`

New files are acceptable when they remove complexity from the main training entrypoint, but the default should be to reuse existing paths and replace placeholder internals.

## Risks

- The paper may define attention or spike-state details that are only partially recoverable from the current scaffold.
- Teacher-to-student feature matching can fail if the student forward contract is not designed first.
- Multi-GPU bugs can hide behind rank-local success if the smoke test is too shallow.
- Overbuilding the training stack would slow delivery and make paper-faithful review harder.

## Review Gates

Implementation must proceed in small checkpoints, each followed by a code review:

1. Model interface and tensor-return contract
2. Softmax-free spiking attention and block repair
3. Tensor-native `SpAD`
4. Real data pipeline
5. DDP training entrypoint
6. README and smoke verification

Important review rule:

- Do not proceed past a checkpoint while important review findings remain unresolved.

## Verification Plan

Before claiming completion:

- run unit or smoke tests for the repaired tensor path
- run a local training smoke launch
- verify checkpoint writing
- verify the README command matches the actual CLI

## Success Statement

This repair pass succeeds when the repository stops being a `dry-run scaffold` and becomes a `paper-faithful BiSpikCLM offline distillation trainer` that can be launched directly on a single node with multiple GPUs.

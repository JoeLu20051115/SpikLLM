# SpikLLM

Paper-faithful BiSpikCLM training scaffold for offline Spike-Aware Alignment Distillation.

## Layout

```text
data/                 # static datasets, no code
models/               # offline teacher weights, e.g. models/opt-125m
scripts/              # launch scripts
output/               # generated checkpoints and logs, ignored by git
bispikclm/            # PyTorch implementation
```

## Environment

```bash
pip install -e .
```

Core runtime dependencies are PyTorch, SpikingJelly, Hugging Face `transformers`, and `datasets`.

## Dummy Sanity Step

```bash
python -m bispikclm.train.train_spad \
  --teacher-model facebook/opt-125m \
  --output-dir output/v1-opt-sft \
  --sequence-length 16 \
  --time-steps 2 \
  --max-steps 1 \
  --dummy-batch
```

This runs one frozen OPT teacher forward pass, one BiSpikCLM student multi-step forward pass, the five-term SpAD loss, `loss.backward()`, gradient clipping, `optimizer.step()`, and checkpoint save.

## Script Launch

```bash
NPROC_PER_NODE=1 TEACHER_MODEL=models/opt-125m bash scripts/run_sft.sh
```

Use a local `models/opt-125m/` directory for offline teacher loading, or set `TEACHER_MODEL=facebook/opt-125m` to load from Hugging Face.

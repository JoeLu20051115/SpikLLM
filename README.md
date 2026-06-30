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

## Dummy Fixed-Batch Diagnostic

```bash
python -m bispikclm.train.train_spad \
  --teacher-model facebook/opt-125m \
  --output-dir output/v1-opt-sft \
  --sequence-length 16 \
  --time-steps 2 \
  --max-steps 80 \
  --learning-rate 3e-3 \
  --dummy-batch
```

This reuses one fixed batch and one frozen OPT teacher target to report `initial_*`, `final_*`, and `delta_*` values for the five-term SpAD loss before saving a checkpoint.

## Script Launch

```bash
NPROC_PER_NODE=1 TEACHER_MODEL=models/opt-125m bash scripts/run_sft.sh
```

Use a local `models/opt-125m/` directory for offline teacher loading, or set `TEACHER_MODEL=facebook/opt-125m` to load from Hugging Face.

# Code Review

## Findings

1. Important: [bispikclm/data/fineweb.py](/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/bispikclm-bootstrap/bispikclm/data/fineweb.py) and [bispikclm/train/train_spad.py](/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/bispikclm-bootstrap/bispikclm/train/train_spad.py) only cache teacher metadata/tokenizer files and dataset-card metadata. That is fine for smoke/bootstrap, but it is not sufficient for a future real distillation run. The current naming is mostly explicit for teachers, but dataset preparation still looks closer to "ready" than it really is.

2. Minor: [bispikclm/models/bispik_lm.py](/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/bispikclm-bootstrap/bispikclm/models/bispik_lm.py) returns `hidden_state[:vocab_size]` as logits. For typical inputs that means the logits length matches sequence length, not vocabulary size, so downstream code could get a misleading LM interface.

## Resolved During Review

- Smoke evaluation no longer writes generated manifests into the repo tree. [bispikclm/data/fineweb.py](/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/bispikclm-bootstrap/bispikclm/data/fineweb.py) now uses `/tmp/bispikclm-datasets` by default, and [tests/smoke/test_scaffold.py](/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/bispikclm-bootstrap/tests/smoke/test_scaffold.py) guards that behavior.

## Assessment

Bootstrap acceptance is met for scaffold, teacher cache presence, smoke tests, and review artifact creation. The retained codebase is suitable as a smoke-tested bootstrap scaffold, not as a training-ready implementation.

#!/usr/bin/env bash
set -euo pipefail

python -m bispikclm.train.eval_lm --smoke-datasets

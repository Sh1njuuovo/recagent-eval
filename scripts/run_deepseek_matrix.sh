#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before running formal evaluation}"

for experiment_config in baseline structured_memory full; do
  uv run recagent-eval evaluate \
    --config "configs/${experiment_config}.yaml" \
    --cases cases/fixed_cases.json \
    --data-dir data/raw/ml-1m \
    --output "artifacts/runs/${experiment_config}-deepseek" \
    --provider deepseek
done

uv run recagent-eval evaluate \
  --config configs/full.yaml \
  --cases cases/stability_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/full-deepseek-stability-repeat \
  --provider deepseek

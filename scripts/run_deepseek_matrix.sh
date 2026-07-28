#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before running formal evaluation}"

experiment_configs=(
  "configs/baseline.yaml"
  "configs/structured_memory.yaml"
  "configs/full_constraint_aware.yaml"
)
experiment_outputs=(
  "artifacts/runs/baseline-deepseek-constraint-aware"
  "artifacts/runs/structured-deepseek-constraint-aware"
  "artifacts/runs/full-deepseek-constraint-aware"
)

for index in "${!experiment_configs[@]}"; do
  uv run recagent-eval evaluate \
    --config "${experiment_configs[$index]}" \
    --cases cases/fixed_cases.json \
    --data-dir data/raw/ml-1m \
    --output "${experiment_outputs[$index]}" \
    --provider deepseek
done

uv run recagent-eval evaluate \
  --config configs/full_constraint_aware.yaml \
  --cases cases/stability_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/full-deepseek-constraint-aware-stability \
  --provider deepseek

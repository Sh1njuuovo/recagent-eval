#!/usr/bin/env bash
set -euo pipefail

qwen_model="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
qwen_port="${VLLM_PORT:-8000}"
qwen_case_count="${QWEN_CASE_COUNT:-20}"

mkdir -p artifacts/runs/qwen-smoke
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > artifacts/runs/qwen-smoke/gpu-before.csv

uv run --with vllm vllm serve "$qwen_model" \
  --host 127.0.0.1 \
  --port "$qwen_port" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  > artifacts/runs/qwen-smoke/vllm.log 2>&1 &
qwen_server_pid=$!
trap 'kill "$qwen_server_pid" 2>/dev/null || true' EXIT

for attempt in $(seq 1 60); do
  if curl --fail --silent "http://127.0.0.1:${qwen_port}/health" >/dev/null; then
    break
  fi
  if [ "$attempt" -eq 60 ]; then
    echo "vLLM did not become healthy" >&2
    exit 1
  fi
  sleep 2
done

export VLLM_BASE_URL="http://127.0.0.1:${qwen_port}/v1"
export VLLM_MODEL="$qwen_model"

uv run recagent-eval evaluate \
  --config configs/full.yaml \
  --cases cases/qwen_smoke_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/qwen-smoke \
  --provider vllm

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > artifacts/runs/qwen-smoke/gpu-after.csv
echo "Completed ${qwen_case_count}-case Qwen smoke test."

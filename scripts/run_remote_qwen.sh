#!/usr/bin/env bash
set -euo pipefail

: "${VIRTUAL_ENV:?Activate the server virtualenv before running this script}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY in the environment}"
: "${VLLM_MODEL:?Set VLLM_MODEL=Qwen/Qwen3-8B}"
: "${RUN_TIMEOUT_SECONDS:?Set a bounded RUN_TIMEOUT_SECONDS budget}"

if [[ "$VLLM_MODEL" != "Qwen/Qwen3-8B" ]]; then
  echo "VLLM_MODEL must be Qwen/Qwen3-8B for the documented smoke run" >&2
  exit 2
fi

vllm_port="${VLLM_PORT:-8000}"
output_dir="${QWEN_OUTPUT_DIR:-artifacts/runs/qwen-smoke}"
python_bin="$VIRTUAL_ENV/bin/python"
cli_bin="$VIRTUAL_ENV/bin/recagent-eval"

for executable in "$python_bin" "$cli_bin" nvidia-smi curl timeout; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable unavailable: $executable" >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > "$output_dir/gpu-before.csv"

cat > "$output_dir/commands.txt" <<'COMMANDS'
# API key is supplied through VLLM_API_KEY and intentionally redacted here.
python -m vllm.entrypoints.openai.api_server Qwen/Qwen3-8B --host 127.0.0.1 --api-key <redacted>
recagent-eval evaluate --config configs/full.yaml --cases cases/qwen_smoke_cases.json --provider vllm
COMMANDS

"$python_bin" - "$output_dir/environment.json" <<'PY'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

destination = sys.argv[1]
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()
packages = {}
for name in ("recagent-eval", "vllm", "torch", "transformers"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
payload = {
    "hardware": gpu,
    "platform": platform.platform(),
    "python": platform.python_version(),
    "packages": packages,
    "model": os.environ["VLLM_MODEL"],
    "provider": "vllm",
}
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

"$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --host 127.0.0.1 \
  --port "$vllm_port" \
  --api-key "$VLLM_API_KEY" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  > "$output_dir/vllm.log" 2>&1 &
qwen_server_pid=$!
cleanup() {
  kill "$qwen_server_pid" 2>/dev/null || true
  wait "$qwen_server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

healthy=false
for attempt in $(seq 1 60); do
  if curl --fail --silent "http://127.0.0.1:${vllm_port}/health" >/dev/null; then
    healthy=true
    break
  fi
  if ! kill -0 "$qwen_server_pid" 2>/dev/null; then
    echo "vLLM exited before becoming healthy; inspect $output_dir/vllm.log" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$healthy" != true ]]; then
  echo "vLLM health check exceeded its bounded wait" >&2
  exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${vllm_port}/v1"
timeout "$RUN_TIMEOUT_SECONDS" "$cli_bin" evaluate \
  --config configs/full.yaml \
  --cases cases/qwen_smoke_cases.json \
  --data-dir data/raw/ml-1m \
  --output "$output_dir" \
  --provider vllm

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > "$output_dir/gpu-after.csv"
echo "Completed the bounded 10-case Qwen smoke test; inspect recorded metrics before matrices."

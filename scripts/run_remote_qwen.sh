#!/usr/bin/env bash
set -euo pipefail

output_root="${QWEN_OUTPUT_ROOT:-artifacts/runs/qwen-smoke}"
run_id="${QWEN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "QWEN_RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi
mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd -P)
output_dir="$output_root/$run_id"
if ! mkdir "$output_dir"; then
  echo "Evidence run directory already exists; refusing reuse: $output_dir" >&2
  exit 2
fi
started_epoch=$(date +%s)
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
qwen_server_pid=""

exec 3>&1 4>&2
exec >> "$output_dir/run.stdout.log" 2>> "$output_dir/run.stderr.log"

finalize() {
  exit_code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$qwen_server_pid" ]]; then
    kill "$qwen_server_pid" 2>/dev/null
    wait "$qwen_server_pid" 2>/dev/null
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
      --format=csv,noheader > "$output_dir/gpu-after.csv" 2>> "$output_dir/run.stderr.log"
  else
    : > "$output_dir/gpu-after.csv"
  fi
  finished_epoch=$(date +%s)
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  duration_seconds=$((finished_epoch - started_epoch))
  final_status="success"
  if [[ "$exit_code" -ne 0 ]]; then
    final_status="failed"
  fi
  printf '{\n  "status": "%s",\n  "exit_code": %d,\n  "started_at": "%s",\n  "finished_at": "%s",\n  "duration_seconds": %d,\n  "run_id": "%s",\n  "run_dir": "%s"\n}\n' \
    "$final_status" "$exit_code" "$started_at" "$finished_at" "$duration_seconds" \
    "$run_id" "$output_dir" \
    > "$output_dir/status.json"
  if [[ ! -f "$output_dir/environment.json" ]]; then
    printf '{"capture_status":"unavailable","exit_code":%d}\n' "$exit_code" \
      > "$output_dir/environment.json"
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    tail -n 50 "$output_dir/run.stderr.log" >&4
  fi
  exit "$exit_code"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_env() {
  variable_name=$1
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    return 2
  fi
}
require_env VIRTUAL_ENV
require_env VLLM_API_KEY
require_env VLLM_MODEL
require_env QWEN_MODEL_REVISION
require_env RUN_TIMEOUT_SECONDS

if [[ "$VLLM_MODEL" != "Qwen/Qwen3-8B" ]]; then
  echo "VLLM_MODEL must be Qwen/Qwen3-8B for the documented smoke run" >&2
  exit 2
fi
if [[ ! "$QWEN_MODEL_REVISION" =~ ^([0-9a-f]{40,64}|[A-Za-z][A-Za-z0-9._-]*[0-9][A-Za-z0-9._-]*)$ ]]; then
  echo "QWEN_MODEL_REVISION must be an immutable commit SHA or versioned tag" >&2
  exit 2
fi

vllm_port="${VLLM_PORT:-8000}"
python_bin="$VIRTUAL_ENV/bin/python"
cli_bin="$VIRTUAL_ENV/bin/recagent-eval"
config_path="${QWEN_CONFIG_PATH:-configs/full.yaml}"
cases_path="${QWEN_CASES_PATH:-cases/qwen_smoke_cases.json}"
data_dir="${QWEN_DATA_DIR:-data/raw/ml-1m}"
vllm_dtype="${VLLM_DTYPE:-auto}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

for executable in "$python_bin" "$cli_bin" nvidia-smi curl timeout; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable unavailable: $executable" >&2
    exit 2
  fi
done

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > "$output_dir/gpu-before.csv"

{
  printf '# Set VLLM_API_KEY securely before replay; its value is never recorded.\n'
  printf ': "${VLLM_API_KEY:?Set VLLM_API_KEY before replay}"\n'
  printf 'export VLLM_MODEL=%q\n' "$VLLM_MODEL"
  printf 'export QWEN_MODEL_REVISION=%q\n' "$QWEN_MODEL_REVISION"
  printf 'export VLLM_BASE_URL=%q\n' "http://127.0.0.1:${vllm_port}/v1"
  printf 'provider_extra_body=%q\n' '{"chat_template_kwargs":{"enable_thinking":false}}'
  printf '%q ' "$python_bin" -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --revision "$QWEN_MODEL_REVISION" \
    --served-model-name "$VLLM_MODEL" \
    --host 127.0.0.1 \
    --port "$vllm_port"
  printf '%s ' '--api-key "${VLLM_API_KEY}"'
  printf '%q ' --dtype "$vllm_dtype" --gpu-memory-utilization "$gpu_memory_utilization"
  printf '\n'
  printf 'timeout %q ' "$RUN_TIMEOUT_SECONDS"
  printf '%q ' "$cli_bin" evaluate \
    --config "$config_path" \
    --cases "$cases_path" \
    --data-dir "$data_dir" \
    --output "$output_dir" \
    --provider vllm
  printf '\n'
} > "$output_dir/commands.txt"

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
    "model_id": os.environ["VLLM_MODEL"],
    "model_revision": os.environ["QWEN_MODEL_REVISION"],
    "api_model": os.environ["VLLM_MODEL"],
    "provider": "vllm",
    "provider_extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

"$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --revision "$QWEN_MODEL_REVISION" \
  --served-model-name "$VLLM_MODEL" \
  --host 127.0.0.1 \
  --port "$vllm_port" \
  --api-key "$VLLM_API_KEY" \
  --dtype "$vllm_dtype" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  > "$output_dir/vllm.log" 2>&1 &
qwen_server_pid=$!

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
  --config "$config_path" \
  --cases "$cases_path" \
  --data-dir "$data_dir" \
  --output "$output_dir" \
  --provider vllm

echo "Completed the bounded 10-case Qwen smoke test; inspect recorded metrics before matrices."

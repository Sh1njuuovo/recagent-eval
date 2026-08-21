#!/usr/bin/env bash
set -euo pipefail
umask 077

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

python_bin="$VIRTUAL_ENV/bin/python"
cli_bin="$VIRTUAL_ENV/bin/recagent-eval"
vllm_port_raw="${VLLM_PORT:-8000}"
if [[ ! "$vllm_port_raw" =~ ^[0-9]+$ ]]; then
  echo "VLLM_PORT must be an integer from 1 through 65535" >&2
  exit 2
fi
vllm_port=$((10#$vllm_port_raw))
if ((vllm_port < 1 || vllm_port > 65535)); then
  echo "VLLM_PORT must be an integer from 1 through 65535" >&2
  exit 2
fi
if [[ ! "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || ((10#$RUN_TIMEOUT_SECONDS < 1)); then
  echo "RUN_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
run_timeout_seconds=$((10#$RUN_TIMEOUT_SECONDS))
if [[ "$VLLM_MODEL" != "Qwen/Qwen3-8B" ]]; then
  echo "VLLM_MODEL must be Qwen/Qwen3-8B for the documented smoke run" >&2
  exit 2
fi
if [[ ! "$QWEN_MODEL_REVISION" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "QWEN_MODEL_REVISION must be an immutable 40- or 64-character lowercase SHA" >&2
  exit 2
fi

output_root="${QWEN_OUTPUT_ROOT:-artifacts/runs/qwen-smoke}"
run_id="${QWEN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "QWEN_RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi
config_path="${QWEN_CONFIG_PATH:-configs/full.yaml}"
cases_path="${QWEN_CASES_PATH:-cases/qwen_smoke_cases.json}"
data_dir="${QWEN_DATA_DIR:-data/raw/ml-1m}"
vllm_dtype="${VLLM_DTYPE:-auto}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
lock_root="${QWEN_LOCK_ROOT:-${TMPDIR:-/tmp}/recagent-qwen-locks}"

for executable in "$python_bin" "$cli_bin" nvidia-smi curl timeout flock; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable unavailable: $executable" >&2
    exit 2
  fi
done
curl_bin=$(command -v curl)
timeout_bin=$(command -v timeout)
flock_bin=$(command -v flock)

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

write_status() {
  status_name=$1
  status_code=$2
  status_finished_at=$3
  status_duration=$4
  STATUS_NAME="$status_name" STATUS_CODE="$status_code" \
    STATUS_STARTED_AT="$started_at" STATUS_FINISHED_AT="$status_finished_at" \
    STATUS_DURATION="$status_duration" STATUS_RUN_ID="$run_id" STATUS_RUN_DIR="$output_dir" \
    "$python_bin" - "$output_dir/status.json" <<'PY'
import json
import os
import sys

payload = {
    "status": os.environ["STATUS_NAME"],
    "exit_code": int(os.environ["STATUS_CODE"]),
    "started_at": os.environ["STATUS_STARTED_AT"],
    "finished_at": os.environ["STATUS_FINISHED_AT"],
    "duration_seconds": int(os.environ["STATUS_DURATION"]),
    "run_id": os.environ["STATUS_RUN_ID"],
    "run_dir": os.environ["STATUS_RUN_DIR"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

write_unavailable_environment() {
  ENV_EXIT_CODE=$1 "$python_bin" - "$output_dir/environment.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {"capture_status": "unavailable", "exit_code": int(os.environ["ENV_EXIT_CODE"])},
        handle,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

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
  write_status "$final_status" "$exit_code" "$finished_at" "$duration_seconds"
  if [[ ! -f "$output_dir/environment.json" ]]; then
    write_unavailable_environment "$exit_code"
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    tail -n 50 "$output_dir/run.stderr.log" >&4
  fi
  exit "$exit_code"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$lock_root"
chmod 700 "$lock_root"
lock_root=$(cd "$lock_root" && pwd -P)
lock_path="$lock_root/127.0.0.1-${vllm_port}.lock"
exec 9> "$lock_path"
if ! flock -n 9; then
  echo "Endpoint lock is already held for 127.0.0.1:${vllm_port}" >&2
  exit 3
fi

if ! "$python_bin" - --port-check "$vllm_port" <<'PY'
import socket
import sys

port = int(sys.argv[2])
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    probe.close()
PY
then
  echo "Loopback endpoint 127.0.0.1:${vllm_port} is occupied" >&2
  exit 3
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > "$output_dir/gpu-before.csv"

"$python_bin" - \
  "$output_dir/commands.txt" "$output_dir/replay.sh" "$python_bin" "$cli_bin" \
  "$curl_bin" "$timeout_bin" "$flock_bin" "$lock_path" "$VLLM_MODEL" \
  "$QWEN_MODEL_REVISION" "$vllm_port" "$vllm_dtype" "$gpu_memory_utilization" \
  "$run_timeout_seconds" "$config_path" "$cases_path" "$data_dir" <<'PY'
import shlex
import sys

(
    commands_path,
    replay_path,
    python_bin,
    cli_bin,
    curl_bin,
    timeout_bin,
    flock_bin,
    lock_path,
    model,
    revision,
    port,
    dtype,
    gpu_memory,
    timeout_seconds,
    config_path,
    cases_path,
    data_dir,
) = sys.argv[1:]
server = [
    python_bin,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    model,
    "--revision",
    revision,
    "--served-model-name",
    model,
    "--host",
    "127.0.0.1",
    "--port",
    port,
    "--api-key",
    "${VLLM_API_KEY}",
    "--dtype",
    dtype,
    "--gpu-memory-utilization",
    gpu_memory,
]
evaluation = [
    timeout_bin,
    timeout_seconds,
    cli_bin,
    "evaluate",
    "--config",
    config_path,
    "--cases",
    cases_path,
    "--data-dir",
    data_dir,
    "--output",
    "${REPLAY_OUTPUT_DIR}",
    "--provider",
    "vllm",
]

def command_with_variables(parts):
    rendered = []
    for part in parts:
        if part in {"${VLLM_API_KEY}", "${REPLAY_OUTPUT_DIR}"}:
            rendered.append(f'"{part}"')
        else:
            rendered.append(shlex.quote(part))
    return " ".join(rendered)

server_command = command_with_variables(server)
evaluate_command = command_with_variables(evaluation)
with open(commands_path, "w", encoding="utf-8") as handle:
    handle.write("# Exact server command; VLLM_API_KEY is intentionally environment-only.\n")
    handle.write(server_command + "\n")
    handle.write("# Exact evaluation command; choose a fresh REPLAY_OUTPUT_DIR.\n")
    handle.write(evaluate_command + "\n")
    handle.write('provider_extra_body=\'{"chat_template_kwargs":{"enable_thinking":false}}\'\n')

model_check = (
    "import json,sys; data=json.load(sys.stdin); expected=sys.argv[1]; "
    "ids=[item.get('id') for item in data.get('data', [])]; "
    "raise SystemExit(0 if ids == [expected] else 1)"
)
replay = f'''#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${{VLLM_API_KEY:?Set VLLM_API_KEY before replay}}"
: "${{REPLAY_OUTPUT_DIR:?Set a fresh REPLAY_OUTPUT_DIR}}"
if ! mkdir "$REPLAY_OUTPUT_DIR"; then echo "REPLAY_OUTPUT_DIR already exists" >&2; exit 2; fi
exec 9>{shlex.quote(lock_path)}
{shlex.quote(flock_bin)} -n 9 || {{ echo "endpoint lock unavailable" >&2; exit 3; }}
if ! {shlex.quote(python_bin)} - --port-check {shlex.quote(port)} <<'PYPORT'
import socket
import sys

port = int(sys.argv[2])
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    probe.close()
PYPORT
then
  echo "Loopback endpoint 127.0.0.1:{port} is occupied" >&2
  exit 3
fi
{server_command} > "$REPLAY_OUTPUT_DIR/vllm.log" 2>&1 &
server_pid=$!
cleanup() {{ kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }}
trap cleanup EXIT INT TERM
healthy=false
for attempt in $(seq 1 60); do
  kill -0 "$server_pid" 2>/dev/null || {{ echo "server exited" >&2; exit 1; }}
  if {shlex.quote(curl_bin)} --fail --silent http://127.0.0.1:{port}/health >/dev/null; then healthy=true; break; fi
  sleep 2
done
[[ "$healthy" == true ]] || {{ echo "health timeout" >&2; exit 1; }}
models=$({shlex.quote(curl_bin)} --fail --silent --header "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:{port}/v1/models)
printf '%s' "$models" | {shlex.quote(python_bin)} -c {shlex.quote(model_check)} {shlex.quote(model)} || {{ echo "model identity mismatch" >&2; exit 1; }}
kill -0 "$server_pid" 2>/dev/null || {{ echo "server exited" >&2; exit 1; }}
export VLLM_BASE_URL=http://127.0.0.1:{port}/v1
{evaluate_command}
'''
with open(replay_path, "w", encoding="utf-8") as handle:
    handle.write(replay)
PY
chmod 700 "$output_dir/replay.sh"

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
  if ! kill -0 "$qwen_server_pid" 2>/dev/null; then
    echo "vLLM exited before becoming healthy; inspect $output_dir/vllm.log" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${vllm_port}/health" >/dev/null; then
    healthy=true
    break
  fi
  sleep 2
done
if [[ "$healthy" != true ]]; then
  echo "vLLM health check exceeded its bounded wait" >&2
  exit 1
fi

models_json=$(curl --fail --silent --header "Authorization: Bearer $VLLM_API_KEY" \
  "http://127.0.0.1:${vllm_port}/v1/models")
if ! printf '%s' "$models_json" | "$python_bin" -c \
  'import json,sys; data=json.load(sys.stdin); ids=[item.get("id") for item in data.get("data", [])]; raise SystemExit(0 if ids == [sys.argv[1]] else 1)' \
  "$VLLM_MODEL"; then
  echo "vLLM model identity does not match expected served model" >&2
  exit 1
fi
if ! kill -0 "$qwen_server_pid" 2>/dev/null; then
  echo "vLLM exited after model identity verification" >&2
  exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${vllm_port}/v1"
timeout "$run_timeout_seconds" "$cli_bin" evaluate \
  --config "$config_path" \
  --cases "$cases_path" \
  --data-dir "$data_dir" \
  --output "$output_dir" \
  --provider vllm

echo "Completed the bounded 10-case Qwen smoke test; inspect recorded metrics before matrices."

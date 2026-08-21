# RTX 4090 remote Qwen runbook

This runbook serves `Qwen/Qwen3-8B` with vLLM on one RTX 4090. The Qwen3-8B
model repository identifies the model license as Apache-2.0; record the exact
model revision used with the run artifacts.

Remote results are **pending** until the commands below run successfully.
Missing GPU access, an unhealthy endpoint, or an interrupted connection stays
recorded as pending/failed evidence. Do not fill metrics from estimates.

## Preconditions and budget

- Linux server with an RTX 4090, compatible NVIDIA driver, `nvidia-smi`, Git,
  `curl`, GNU `timeout`, Python 3.11+, and enough model-cache disk.
- A platform time/cost limit and `RUN_TIMEOUT_SECONDS` chosen before launch.
- A dedicated virtualenv. Every Python, vLLM, and project command below runs
  inside that environment; never install into the server's global Python.
- The repository and MovieLens data are already present on the server.

Record the server/provider budget in the job notes. If the budget expires,
retain logs and mark remaining 10-case or 50+20 runs pending.

## Create and activate the environment

```bash
cd recagent-eval
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev,ml]' vllm
recagent-eval download-data --output data/raw
```

Confirm the interpreter is inside the virtualenv, then capture initial facts:

```bash
test -n "${VIRTUAL_ENV:-}"
python --version
python -m pip freeze
nvidia-smi
uname -a
```

## Configure without exposing the API key

Read the key silently so it is not saved in shell history. The script never
prints it and writes a redacted command record.

```bash
read -rsp 'Temporary vLLM API key: ' VLLM_API_KEY && printf '\n'
export VLLM_API_KEY
export VLLM_MODEL='Qwen/Qwen3-8B'
# Pin the exact immutable Hugging Face commit SHA (or an immutable release tag).
export QWEN_MODEL_REVISION='<immutable-model-commit-sha>'
export VLLM_PORT=8000
export RUN_TIMEOUT_SECONDS=1800
```

The server binds only to `127.0.0.1`. From the laptop, open a separate terminal
and forward a local port; replace the SSH destination with the real host:

```bash
ssh -L 8000:127.0.0.1:8000 -N user@gpu-host
```

A client on the laptop can then use `http://127.0.0.1:8000/v1`. Keep the API
key in that client's environment as well.

## Required 10-case smoke first

`cases/qwen_smoke_cases.json` contains exactly 10 fixed cases. The script starts
vLLM, checks `http://127.0.0.1:8000/health`, runs the existing `evaluate` CLI,
captures the environment and GPU snapshots, and kills vLLM on exit.

```bash
scripts/run_remote_qwen.sh
```

Do not start either matrix until all of these exist and are internally
consistent:

- `artifacts/runs/qwen-smoke/environment.json`, `commands.txt`, `status.json`,
  `run.stdout.log`, `run.stderr.log`, `vllm.log`, `gpu-before.csv`, and
  `gpu-after.csv`;
- `episodes.jsonl`, `metrics.json`, and `run_manifest.json`;
- exactly 10 episodes, with provider/model identity and no credential text;
- measured `plan_valid_rate`, `tool_success_rate`,
  `constraint_satisfaction_rate`, `fallback_rate`, `latency_p50_ms`,
  `latency_p95_ms`, and `total_tokens`.

The replayable `commands.txt` captures the effective executable, immutable
revision, API model, host/port, dtype, GPU-memory utilization, timeout,
config/case/data/output paths, and the Qwen non-thinking provider setting. It
references `VLLM_API_KEY` without recording its value. `status.json` records
success/failure, exit code, start/end timestamps, and duration even when the
health check or evaluation fails. The manifest and environment sidecar capture
package/runtime versions, model, configuration and case fingerprints. vLLM
logs are the source for serving throughput details. If tokens/s is absent from
the log, report it as unavailable.

## 50+20 matrices after smoke approval

Only after reviewing the 10-case artifacts, keep the same healthy loopback
server running (or restart it with the same revision) and execute the existing
evaluation CLI on the 50 fixed cases and 20 stability cases:

```bash
export VLLM_BASE_URL='http://127.0.0.1:8000/v1'
timeout "$RUN_TIMEOUT_SECONDS" recagent-eval evaluate \
  --config configs/full.yaml \
  --cases cases/fixed_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/qwen-formal-50 \
  --provider vllm

timeout "$RUN_TIMEOUT_SECONDS" recagent-eval evaluate \
  --config configs/full.yaml \
  --cases cases/stability_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/qwen-stability-20 \
  --provider vllm
```

These commands deliberately reuse generic `evaluate`; it continues to reject
LambdaMART configs before provider or dataset execution. Learned-ranker frozen
evaluation remains available only through its gated command.

## Failure handling and cleanup

- Missing GPU/driver or connection: preserve the error and mark all unrun work
  pending. Never fabricate GPU, latency, token, or quality metrics.
- Out of memory: capture `vllm.log` and GPU state. Adjust the declared memory or
  context budget and record the revised command; do not silently swap models.
- Timeout or malformed JSON: preserve episode errors. The agent performs its
  bounded repair and deterministic fallback, reflected in `fallback_rate`.
- Health-check failure: inspect the log and stop; matrices remain pending.
- End of job: copy artifacts off the host, unset `VLLM_API_KEY`, stop vLLM, and
  release the server. The script's trap handles its own vLLM child process.

```bash
unset VLLM_API_KEY
```

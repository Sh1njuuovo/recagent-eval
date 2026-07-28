# RTX 4090 remote runbook

## Preconditions

- One NVIDIA RTX 4090 with at least 20 GB free VRAM.
- Linux, a compatible NVIDIA driver, Python 3.11, `uv`, Git, and `curl`.
- At least 20 GB free disk for environment and model cache.
- A maximum runtime/budget set on the remote platform.

Record environment before the run:

```bash
nvidia-smi
python --version
uv --version
uname -a
```

## Setup

```bash
git clone <your-recagent-eval-repository>
cd recagent-eval
uv sync --extra dev
uv run recagent-eval download-data --output data/raw
uv run recagent-eval prepare-cases \
  --data-dir data/raw/ml-1m \
  --output cases/fixed_cases.json
```

Create the 20-case smoke subset without changing the formal 50-case file:

```bash
uv run recagent-eval subset-cases \
  --source cases/fixed_cases.json \
  --output cases/qwen_smoke_cases.json \
  --single-turn-count 16 \
  --multi-turn-count 4
```

Then run:

```bash
scripts/run_remote_qwen.sh
```

## Evidence to preserve

- `gpu-before.csv`, `gpu-after.csv`, `vllm.log`.
- `episodes.jsonl`, `metrics.json`, `run_manifest.json`.
- Model ID and revision, CUDA/driver version, wall-clock time and platform cost.
- p50/p95 episode latency, plan validity, tool success, tokens/s from vLLM logs,
  and peak GPU memory from `nvidia-smi`.

Run only 10–20 cases. Qwen is a compatibility and performance smoke test; its
results must not be merged into the formal DeepSeek matrix.

## Failure recovery

- Out of memory: reduce `--gpu-memory-utilization` or model context length;
  do not silently switch to a different model.
- Server unhealthy: inspect `vllm.log`, record the model/driver error, then
  terminate the process.
- Invalid plans: preserve episode output; the Agent repairs once and records a
  deterministic fallback.
- Before releasing the machine, copy artifacts off the host and stop vLLM.

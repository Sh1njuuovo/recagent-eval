import json
import subprocess
import sys
from pathlib import Path


def test_deepseek_matrix_uses_constraint_aware_frozen_config() -> None:
    script = Path("scripts/run_deepseek_matrix.sh").read_text()

    assert "configs/full_constraint_aware.yaml" in script
    assert "full-deepseek-constraint-aware" in script
    assert "full-deepseek-constraint-aware-stability" in script


def test_remote_qwen_script_is_loopback_secret_safe_and_venv_only() -> None:
    path = Path("scripts/run_remote_qwen.sh")
    script = path.read_text()

    syntax = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    assert "set -euo pipefail" in script
    assert "VIRTUAL_ENV" in script
    assert "require_env VLLM_API_KEY" in script
    assert "require_env VLLM_MODEL" in script
    assert "require_env QWEN_MODEL_REVISION" in script
    assert "Qwen/Qwen3-8B" in script
    assert "--host 127.0.0.1" in script
    assert "VLLM_BASE_URL=\"http://127.0.0.1:" in script
    assert "cases/qwen_smoke_cases.json" in script
    assert "--provider vllm" in script
    assert "timeout" in script
    assert "trap" in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script
    assert "environment.json" in script
    assert "commands.txt" in script
    assert "status.json" in script
    assert "run.stdout.log" in script
    assert "run.stderr.log" in script
    assert "uv run --with" not in script
    assert "echo \"$VLLM_API_KEY\"" not in script
    assert "sk-" not in script


def test_qwen_smoke_fixture_has_exactly_ten_cases() -> None:
    cases = json.loads(Path("cases/qwen_smoke_cases.json").read_text())
    assert len(cases) == 10


def test_remote_runbook_documents_forwarding_evidence_and_pending_results() -> None:
    runbook = Path("docs/remote-4090.md").read_text()

    for required in (
        "ssh -L",
        "127.0.0.1",
        "Qwen/Qwen3-8B",
        "QWEN_MODEL_REVISION",
        "Apache-2.0",
        "10-case",
        "50+20",
        "plan_valid",
        "tool_success",
        "constraint",
        "fallback",
        "p50",
        "p95",
        "tokens",
        "pending",
        "status.json",
        "commands.txt",
    ):
        assert required in runbook


def test_remote_script_exits_without_required_environment(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(Path("scripts/run_remote_qwen.sh").resolve())],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "VIRTUAL_ENV" in result.stderr
    assert "API_KEY" not in result.stdout


def test_remote_script_failure_records_exact_args_and_final_evidence(tmp_path: Path) -> None:
    fake_venv = tmp_path / "venv"
    bin_dir = fake_venv / "bin"
    bin_dir.mkdir(parents=True)
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    logs = tmp_path / "stub-logs"
    logs.mkdir()
    output = tmp_path / "run output"

    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" ]]; then
  printf '%s\\n' "$@" > "$STUB_LOG/serve.args"
  trap 'exit 0' TERM INT
  while true; do /bin/sleep 1; done
fi
exec "$REAL_PYTHON" "$@"
""",
    )
    _write_executable(
        bin_dir / "recagent-eval",
        """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_LOG/evaluate.args"
echo simulated-evaluation-failure >&2
exit 7
""",
    )
    _write_executable(
        stub_dir / "nvidia-smi",
        """#!/usr/bin/env bash
echo 'Stub RTX 4090, 555.1, 24564 MiB, 10 MiB'
""",
    )
    _write_executable(stub_dir / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        stub_dir / "timeout",
        "#!/usr/bin/env bash\nshift\nexec \"$@\"\n",
    )

    env = {
        "PATH": f"{stub_dir}:/usr/bin:/bin",
        "VIRTUAL_ENV": str(fake_venv),
        "VLLM_API_KEY": "must-not-appear",
        "VLLM_MODEL": "Qwen/Qwen3-8B",
        "QWEN_MODEL_REVISION": "0123456789abcdef0123456789abcdef01234567",
        "RUN_TIMEOUT_SECONDS": "91",
        "VLLM_PORT": "8123",
        "QWEN_OUTPUT_DIR": str(output),
        "REAL_PYTHON": sys.executable,
        "STUB_LOG": str(logs),
    }
    invalid_revision_output = tmp_path / "invalid revision"
    invalid_revision = subprocess.run(
        ["bash", str(Path("scripts/run_remote_qwen.sh").resolve())],
        cwd=Path.cwd(),
        env={
            **env,
            "QWEN_MODEL_REVISION": "main",
            "QWEN_OUTPUT_DIR": str(invalid_revision_output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_revision.returncode == 2
    assert json.loads((invalid_revision_output / "status.json").read_text())["status"] == "failed"

    result = subprocess.run(
        ["bash", str(Path("scripts/run_remote_qwen.sh").resolve())],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    serve_args = (logs / "serve.args").read_text().splitlines()
    evaluate_args = (logs / "evaluate.args").read_text().splitlines()
    assert serve_args == [
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen3-8B",
        "--revision",
        "0123456789abcdef0123456789abcdef01234567",
        "--served-model-name",
        "Qwen/Qwen3-8B",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--api-key",
        "must-not-appear",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.85",
    ]
    assert evaluate_args == [
        "evaluate",
        "--config",
        "configs/full.yaml",
        "--cases",
        "cases/qwen_smoke_cases.json",
        "--data-dir",
        "data/raw/ml-1m",
        "--output",
        str(output),
        "--provider",
        "vllm",
    ]
    commands = (output / "commands.txt").read_text()
    assert "must-not-appear" not in commands
    assert "QWEN_MODEL_REVISION" in commands
    assert "enable_thinking" in commands
    assert subprocess.run(
        ["bash", "-n", str(output / "commands.txt")], check=False
    ).returncode == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exit_code"] == 7
    assert status["duration_seconds"] >= 0
    assert status["started_at"]
    assert status["finished_at"]
    assert (output / "gpu-after.csv").exists()
    assert "simulated-evaluation-failure" in (output / "run.stderr.log").read_text()
    assert "must-not-appear" not in (output / "run.stderr.log").read_text()

    _write_executable(
        bin_dir / "recagent-eval",
        """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_LOG/evaluate.args"
echo simulated-evaluation-success
exit 0
""",
    )
    success_output = tmp_path / "successful run"
    success_env = {**env, "QWEN_OUTPUT_DIR": str(success_output)}
    success = subprocess.run(
        ["bash", str(Path("scripts/run_remote_qwen.sh").resolve())],
        cwd=Path.cwd(),
        env=success_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert success.returncode == 0
    success_status = json.loads((success_output / "status.json").read_text())
    assert success_status["status"] == "success"
    assert success_status["exit_code"] == 0
    assert (success_output / "gpu-after.csv").exists()
    assert (success_output / "environment.json").exists()
    assert "simulated-evaluation-success" in (
        success_output / "run.stdout.log"
    ).read_text()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_candidate_score_script_runs_portably_without_taste(tmp_path: Path) -> None:
    script = Path("scripts/candidate_score.py").resolve()
    jd = Path("reports/profile/jd.txt").resolve()
    candidates = Path("reports/profile/candidates.json").resolve()
    output = tmp_path / "ranking"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--jd",
            str(jd),
            "--candidates",
            str(candidates),
            "--out",
            str(output),
            "--as-of",
            "2026-08-20",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ranked = json.loads((output / "candidate_score.json").read_text())["candidates"]
    markdown = (output / "candidate_score.md").read_text()

    assert [item["name"] for item in ranked] == [
        "RecAI/InteRecAgent",
        "RecBole",
        "OpenOneRec",
    ]
    assert all(item["max_raw_score"] == 104 for item in ranked)
    assert all("user_preference" not in item["score_breakdown"] for item in ranked)
    assert all(
        not {
            "taste_tags",
            "avoid_tags",
            "project_taste_notes",
            "taste_matches",
            "taste_mismatches",
            "user_preference_notes",
        }
        & item.keys()
        for item in ranked
    )
    assert "Taste Fit" not in markdown
    assert "Apache-2.0" in markdown
    assert "license unclear" not in markdown
    assert "documentation incomplete" in markdown
    assert "1 risk note" in ranked[1]["score_reasons"]

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
    assert ': "${VLLM_API_KEY:?' in script
    assert ': "${VLLM_MODEL:?' in script
    assert "Qwen/Qwen3-8B" in script
    assert "--host 127.0.0.1" in script
    assert "VLLM_BASE_URL=\"http://127.0.0.1:" in script
    assert "cases/qwen_smoke_cases.json" in script
    assert "--provider vllm" in script
    assert "timeout" in script
    assert "trap" in script
    assert "environment.json" in script
    assert "commands.txt" in script
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

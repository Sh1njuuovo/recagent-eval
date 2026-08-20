import json
import subprocess
import sys
from pathlib import Path


def test_deepseek_matrix_uses_constraint_aware_frozen_config() -> None:
    script = Path("scripts/run_deepseek_matrix.sh").read_text()

    assert "configs/full_constraint_aware.yaml" in script
    assert "full-deepseek-constraint-aware" in script
    assert "full-deepseek-constraint-aware-stability" in script


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

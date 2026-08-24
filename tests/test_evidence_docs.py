from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_a_is_development_evidence_only() -> None:
    text = (ROOT / "reports/experiments/v2-strong-baselines-confirmation-a.md").read_text()
    lowered = text.lower()
    assert "development/debugging/replication evidence" in lowered
    assert "not final certification" in lowered
    assert "met on confirmation-a" not in lowered
    assert "untouched" not in lowered


def test_confirmation_b_is_the_sole_final_certification() -> None:
    text = (ROOT / "reports/experiments/v2-strong-baselines-confirmation-b.md").read_text()
    lowered = text.lower()
    assert "sole final certification cohort" in lowered
    assert "both cohorts" not in lowered
    assert "both untouched" not in lowered


def test_peak_rss_correction_invalidates_legacy_field() -> None:
    text = (ROOT / "reports/experiments/v2-baseline-evidence-corrections.md").read_text()
    assert "invalid_due_to_platform_unit_bug" in text
    assert "process_peak_rss_mib" in text
    assert "independent process" in text.lower()


def test_current_project_docs_report_confirmation_b_certification() -> None:
    paths = [
        "README.md",
        "docs/HANDOFF-2026-08-22.md",
        "docs/project-methodology.md",
        "docs/demo-script.md",
        "reports/interview-pack/resume_star.md",
        "reports/interview-pack/interview-pack.md",
        "reports/interview-pack/interview_qa.md",
        "reports/interview-pack/ppt_prompt.md",
        "reports/interview-pack/application_checklist.md",
    ]
    for relative in paths:
        text = (ROOT / relative).read_text()
        assert "Confirmation-B" in text, relative
        assert "0.0555" in text, relative
        assert "0.118" in text, relative


def test_current_docs_keep_frozen_unconsumed_and_qwen_pending() -> None:
    combined = "\n".join(
        (ROOT / relative).read_text()
        for relative in (
            "README.md",
            "docs/HANDOFF-2026-08-22.md",
            "docs/project-methodology.md",
            "docs/demo-script.md",
            "reports/interview-pack/interview-pack.md",
        )
    )
    assert "frozen test remains unconsumed" in combined.lower()
    assert "qwen/4090 remains pending" in combined.lower()
    assert "certified on both" not in combined.lower()
    assert "both untouched" not in combined.lower()

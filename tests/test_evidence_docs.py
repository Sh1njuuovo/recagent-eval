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

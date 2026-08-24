from __future__ import annotations

import pytest

from recagent_eval.baseline_summary import summarize_baselines, summary_to_markdown
from recagent_eval.evidence import artifact_fingerprint


def _artifact(
    method: str,
    rows: list[dict[str, object]],
    *,
    cohort: str = "confirmation_a",
) -> dict[str, object]:
    artifact = {
        "schema_version": "baseline-evaluation/v1",
        "method": method,
        "cohort": cohort,
        "config_fingerprint": "c" * 64,
        "dataset_fingerprint": "d" * 64,
        "model_fingerprint": "m" * 64,
        "user_count": len(rows),
        "per_user_rows": rows,
        "aggregates": {},
    }
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    return artifact


def _rows(users: list[int], ndcgs: list[float]) -> list[dict[str, object]]:
    return [
        {
            "user_id": user,
            "recall_at_10": 1.0 if ndcg > 0 else 0.0,
            "ndcg_at_10": ndcg,
            "mrr_at_10": 0.0,
            "candidate_recall": 1.0,
            "constraint_satisfied": True,
            "latency_ms": 1.0,
        }
        for user, ndcg in zip(users, ndcgs, strict=True)
    ]


def test_summarize_reports_aggregates_and_pairwise_bootstrap() -> None:
    users = list(range(1, 101))
    itemcf = _artifact(
        "itemcf_direct", _rows(users, [1.0 if i % 2 else 0.0 for i in range(100)])
    )
    learned = _artifact(
        "current_v2b", _rows(users, [1.0 if i % 3 else 0.0 for i in range(100)])
    )
    summary = summarize_baselines(
        {"itemcf_direct": itemcf, "current_v2b": learned},
        cohort="confirmation_a",
        expected_user_ids=users,
        cohort_ledger_fingerprint="l" * 64,
    )
    assert summary["schema_version"] == "baseline-summary/v2"
    assert summary["source_schema_version"] == "baseline-evaluation/v1"
    assert summary["cohort_ledger_fingerprint"] == "l" * 64
    assert summary["user_count"] == 100
    assert summary["aggregates"]["itemcf_direct"]["ndcg_at_10"] == 0.5
    assert "itemcf_direct_vs_current_v2b" in summary["pairwise_ndcg_bootstrap"]
    pair = summary["pairwise_ndcg_bootstrap"]["itemcf_direct_vs_current_v2b"]
    assert pair["resamples"] == 2000
    assert pair["lower"] <= pair["upper"]
    assert abs(pair["mean_delta"] - (34 / 100 - 0.5)) < 1e-9


def test_summarize_rejects_misaligned_users() -> None:
    itemcf = _artifact("itemcf_direct", _rows([1, 2, 3], [1.0, 0.0, 1.0]))
    learned = _artifact("current_v2b", _rows([1, 2], [1.0, 0.0]))
    try:
        summarize_baselines(
            {"itemcf_direct": itemcf, "current_v2b": learned},
            cohort="confirmation_a",
            expected_user_ids=[1, 2, 3],
            cohort_ledger_fingerprint="l" * 64,
        )
    except ValueError as exc:
        assert "missing, extra, or out of ledger order" in str(exc)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.parametrize("rows", [None, [], "not-a-list"])
def test_summarize_rejects_missing_rows(rows: object) -> None:
    with pytest.raises(ValueError, match="no per_user_rows"):
        summarize_baselines(
            {
                "itemcf_direct": {
                    "schema_version": "baseline-evaluation/v1",
                    "method": "itemcf_direct",
                    "cohort": "confirmation_a",
                    "per_user_rows": rows,
                }
            },
            cohort="confirmation_a",
            expected_user_ids=[1],
            cohort_ledger_fingerprint="l" * 64,
        )


def test_summary_to_markdown_serializes_aggregates_and_pairwise_rows() -> None:
    summary = summarize_baselines(
        {
            "itemcf_direct": _artifact(
                "itemcf_direct",
                _rows([1, 2], [0.0, 1.0]),
                cohort="confirmation_b",
            ),
            "current_v2b": _artifact(
                "current_v2b",
                _rows([1, 2], [1.0, 1.0]),
                cohort="confirmation_b",
            ),
        },
        cohort="confirmation_b",
        expected_user_ids=[1, 2],
        cohort_ledger_fingerprint="l" * 64,
    )
    markdown = summary_to_markdown(summary)
    assert "cohort `confirmation_b`" in markdown
    assert "| current_v2b | 1.0000 | 1.0000" in markdown
    assert "itemcf_direct_vs_current_v2b" in markdown
    assert markdown.endswith("\n")

from __future__ import annotations

from recagent_eval.baseline_summary import summarize_baselines


def _artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "baseline-evaluation/v1",
        "per_user_rows": rows,
        "aggregates": {},
    }


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
    itemcf = _artifact(_rows(users, [1.0 if i % 2 else 0.0 for i in range(100)]))
    learned = _artifact(_rows(users, [1.0 if i % 3 else 0.0 for i in range(100)]))
    summary = summarize_baselines(
        {"itemcf_direct": itemcf, "current_v2b": learned}, cohort="confirmation_a"
    )
    assert summary["user_count"] == 100
    assert summary["aggregates"]["itemcf_direct"]["ndcg_at_10"] == 0.5
    assert "itemcf_direct_vs_current_v2b" in summary["pairwise_ndcg_bootstrap"]
    pair = summary["pairwise_ndcg_bootstrap"]["itemcf_direct_vs_current_v2b"]
    assert pair["resamples"] == 2000
    assert pair["lower"] <= pair["upper"]
    assert abs(pair["mean_delta"] - (34 / 100 - 0.5)) < 1e-9


def test_summarize_rejects_misaligned_users() -> None:
    itemcf = _artifact(_rows([1, 2, 3], [1.0, 0.0, 1.0]))
    learned = _artifact(_rows([1, 2], [1.0, 0.0]))
    try:
        summarize_baselines(
            {"itemcf_direct": itemcf, "current_v2b": learned},
            cohort="confirmation_a",
        )
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("expected ValueError")

from __future__ import annotations

import math

import pytest

import recagent_eval.baseline_eval as module
from recagent_eval.baseline_eval import (
    MetricRow,
    metric_json,
    paired_bootstrap_deltas,
    score_ranking,
)


def _ranked() -> list[int]:
    return [5, 3, 1, 2, 4, 6, 7, 8, 9, 10]


def test_score_ranking_metrics() -> None:
    row = score_ranking(
        ranked_ids=_ranked(),
        target=3,
        allowed={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        history={11, 12},
        candidate_recall=1.0,
        latency_ms=1.0,
    )
    assert row.recall_at_10 == 1.0
    assert abs(row.ndcg_at_10 - (1.0 / math.log2(3))) < 1e-9
    assert row.mrr_at_10 == 1.0 / 2
    assert row.constraint_satisfied is True
    assert row.candidate_recall == 1.0


def test_score_ranking_missing_target_and_constraint_violation() -> None:
    row = score_ranking(
        ranked_ids=[10, 11],
        target=3,
        allowed={1, 2, 3},
        history=set(),
        candidate_recall=0.0,
        latency_ms=0.5,
    )
    assert row.recall_at_10 == 0.0
    assert row.ndcg_at_10 == 0.0
    assert row.mrr_at_10 == 0.0
    assert row.constraint_satisfied is False


def test_paired_bootstrap_is_deterministic_and_reports_ci() -> None:
    first = [1.0, 0.0, 1.0, 0.0] * 25
    second = [0.0, 0.0, 1.0, 0.0] * 25
    a = paired_bootstrap_deltas(first, second, seed=42, resamples=2000)
    b = paired_bootstrap_deltas(first, second, seed=42, resamples=2000)
    assert a == b
    assert a["resamples"] == 2000
    assert a["lower"] <= a["upper"]
    assert abs(a["mean_delta"] - (-0.25)) < 1e-9


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([], [], "equal non-empty"),
        ([0.0], [0.0, 1.0], "equal non-empty"),
        ([float("nan")], [0.0], "finite"),
        ([0.0], [float("inf")], "finite"),
    ],
)
def test_paired_bootstrap_rejects_invalid_values(
    baseline: list[float], candidate: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_deltas(baseline, candidate, seed=42)


def test_register_baseline_rejects_duplicate_name() -> None:
    name = "test-only-duplicate"

    def scorer() -> None:
        return None

    try:
        assert module.register_baseline(name)(scorer) is scorer
        with pytest.raises(ValueError, match="already registered"):
            module.register_baseline(name)(scorer)
    finally:
        module.BASELINE_SCORERS.pop(name, None)


def test_metric_json_reports_aggregates_and_coverage() -> None:
    rows = [
        MetricRow(
            user_id=1,
            recall_at_10=1.0,
            ndcg_at_10=0.5,
            mrr_at_10=1.0,
            candidate_recall=1.0,
            constraint_satisfied=True,
            latency_ms=2.0,
            recommended_ids=(1, 2),
        ),
        MetricRow(
            user_id=2,
            recall_at_10=0.0,
            ndcg_at_10=0.0,
            mrr_at_10=0.0,
            candidate_recall=1.0,
            constraint_satisfied=True,
            latency_ms=4.0,
            recommended_ids=(2, 3),
        ),
    ]
    artifact = metric_json(
        rows,
        method="popularity",
        cohort="confirmation_a",
        universe_size=10,
        config_fingerprint="config",
        dataset_fingerprint="dataset",
        model_fingerprint="model",
        cohort_ledger_fingerprint="ledger",
        selected_params={},
        parameter_grid=[],
        seed=42,
        dependency_versions={"python": "3.13"},
        hardware={"platform": "Darwin", "cpu_count": 8},
        training_seconds=1.0,
        resource_usage={
            "metric_name": "process_peak_rss_mib",
            "normalized_mib": 100.0,
        },
        model_size_bytes=1024,
        environment={"python": "3.13"},
    )
    agg = artifact["aggregates"]
    assert agg["recall_at_10"] == 0.5
    assert agg["ndcg_at_10"] == 0.25
    assert agg["coverage"] == 0.3  # {1,2,3} / 10
    assert agg["latency_ms_p95"] == 4.0
    assert artifact["fingerprint"]
    assert artifact["schema_version"] == "baseline-evaluation/v2"
    assert artifact["ordered_user_ids"] == [1, 2]
    assert artifact["selected_params"]["source"] == "observed"
    assert artifact["resource_usage"]["normalized_mib"] == 100.0
    assert "peak_memory_mb" not in artifact


def test_metric_json_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        metric_json(
            [],
            method="popularity",
            cohort="development",
            universe_size=0,
            config_fingerprint="config",
            dataset_fingerprint="dataset",
            model_fingerprint="model",
            cohort_ledger_fingerprint="ledger",
            selected_params={},
            parameter_grid=[],
            seed="not_applicable",
            dependency_versions={"python": "3.13"},
            hardware={"platform": "Darwin", "cpu_count": 8},
            training_seconds=0.0,
            resource_usage={
                "metric_name": "process_peak_rss_mib",
                "normalized_mib": 0.0,
            },
            model_size_bytes=0,
            environment={},
        )

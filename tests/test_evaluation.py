import pytest

from recagent_eval.data import Movie
from recagent_eval.evaluation import (
    EvaluationRecord,
    aggregate_metrics,
    ndcg_at_k,
    recall_at_k,
)
from recagent_eval.models import (
    PreferenceState,
    RecommendationResult,
    RecommendedMovie,
    ToolTrace,
)


def test_ranking_metrics_use_held_out_relevance() -> None:
    ranked = [4, 2, 9]
    relevant = {2, 8}

    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(
        (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)
    )


def test_aggregate_metrics_reports_agent_constraints_and_latency() -> None:
    movies = {
        1: Movie(1, "Allowed", ("Sci-Fi",), 2000),
        2: Movie(2, "Excluded", ("Drama",), 1990),
    }
    first = RecommendationResult(
        movies=[RecommendedMovie(movie_id=1, title="Allowed")],
        preference_state=PreferenceState(
            required_genres={"Sci-Fi"},
            excluded_movie_ids={2},
        ),
        traces=[ToolTrace(tool="rerank", success=True)],
        latency_ms=100,
        plan_valid=True,
        prompt_tokens=10,
        completion_tokens=5,
    )
    second = RecommendationResult(
        movies=[],
        preference_state=PreferenceState(),
        traces=[ToolTrace(tool="rerank", success=False, error="failed")],
        latency_ms=300,
        plan_valid=False,
        fallback_used=True,
    )
    records = [
        EvaluationRecord(result=first, relevant_movie_ids={1}),
        EvaluationRecord(result=second, relevant_movie_ids={9}),
    ]

    metrics = aggregate_metrics(records, movies, k=10)

    assert metrics["recall_at_10"] == 0.5
    assert metrics["hit_rate_at_10"] == 0.5
    assert metrics["plan_valid_rate"] == 0.5
    assert metrics["tool_success_rate"] == 0.5
    assert metrics["episode_failure_rate"] == 0.5
    assert metrics["fallback_rate"] == 0.5
    assert metrics["constraint_satisfaction_rate"] == 1.0
    assert metrics["excluded_item_violation_rate"] == 0.0
    assert metrics["latency_p50_ms"] == 200
    assert metrics["latency_p95_ms"] == pytest.approx(290)
    assert metrics["total_tokens"] == 15


def test_retention_accepts_hard_exclusion_for_soft_negative_preference() -> None:
    result = RecommendationResult(
        preference_state=PreferenceState(excluded_genres={"Action"}),
    )
    record = EvaluationRecord(
        result=result,
        relevant_movie_ids={1},
        expected_preferences=PreferenceState(disliked_genres={"Action"}),
    )

    metrics = aggregate_metrics([record], {}, k=10)

    assert metrics["preference_retention_rate"] == 1.0

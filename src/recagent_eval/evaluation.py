from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recagent_eval.data import Movie
from recagent_eval.models import PreferenceState, RecommendationResult


@dataclass(frozen=True)
class EvaluationRecord:
    result: RecommendationResult
    relevant_movie_ids: set[int]
    expected_preferences: PreferenceState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def hit_rate_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant))


def ndcg_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1 / math.log2(index + 2)
        for index, movie_id in enumerate(ranked_ids[:k])
        if movie_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def aggregate_metrics(
    records: list[EvaluationRecord],
    movies: dict[int, Movie],
    *,
    k: int = 10,
) -> dict[str, float | int]:
    if not records:
        return {}
    recalls: list[float] = []
    ndcgs: list[float] = []
    hits: list[float] = []
    plan_valid: list[float] = []
    constraint_satisfied: list[float] = []
    retention: list[float] = []
    trace_success: list[float] = []
    episode_failures: list[float] = []
    fallbacks: list[float] = []
    latencies: list[float] = []
    excluded_violations = 0
    recommendation_count = 0

    for record in records:
        result = record.result
        ranked_ids = [movie.movie_id for movie in result.movies]
        recalls.append(recall_at_k(ranked_ids, record.relevant_movie_ids, k))
        ndcgs.append(ndcg_at_k(ranked_ids, record.relevant_movie_ids, k))
        hits.append(hit_rate_at_k(ranked_ids, record.relevant_movie_ids, k))
        plan_valid.append(float(result.plan_valid))
        trace_success.extend(float(trace.success) for trace in result.traces)
        episode_failures.append(
            float(bool(result.errors) or any(not trace.success for trace in result.traces))
        )
        fallbacks.append(float(result.fallback_used))
        latencies.append(result.latency_ms)

        satisfied = True
        state = result.preference_state
        for movie_id in ranked_ids:
            movie = movies.get(movie_id)
            recommendation_count += 1
            if movie_id in state.excluded_movie_ids:
                excluded_violations += 1
                satisfied = False
            if movie is None or not _movie_satisfies(movie, state):
                satisfied = False
        constraint_satisfied.append(float(satisfied))
        if record.expected_preferences is not None:
            retention.append(float(_retains(state, record.expected_preferences)))

    return {
        f"recall_at_{k}": _mean(recalls),
        f"ndcg_at_{k}": _mean(ndcgs),
        f"hit_rate_at_{k}": _mean(hits),
        "plan_valid_rate": _mean(plan_valid),
        "tool_success_rate": _mean(trace_success),
        "episode_failure_rate": _mean(episode_failures),
        "fallback_rate": _mean(fallbacks),
        "constraint_satisfaction_rate": _mean(constraint_satisfied),
        "excluded_item_violation_rate": (
            excluded_violations / recommendation_count if recommendation_count else 0.0
        ),
        "preference_retention_rate": _mean(retention) if retention else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "llm_calls": sum(record.result.llm_calls for record in records),
        "total_tokens": sum(
            record.result.prompt_tokens + record.result.completion_tokens for record in records
        ),
        "episodes": len(records),
    }


def _movie_satisfies(movie: Movie, state: PreferenceState) -> bool:
    genres = set(movie.genres)
    if state.required_genres and not state.required_genres.issubset(genres):
        return False
    if state.excluded_genres & genres:
        return False
    if state.year_min is not None and (movie.year is None or movie.year < state.year_min):
        return False
    return not (state.year_max is not None and (movie.year is None or movie.year > state.year_max))


def _retains(actual: PreferenceState, expected: PreferenceState) -> bool:
    return (
        expected.liked_movie_ids.issubset(actual.liked_movie_ids)
        and expected.disliked_movie_ids.issubset(actual.disliked_movie_ids)
        and expected.liked_genres.issubset(actual.liked_genres)
        and expected.disliked_genres.issubset(actual.disliked_genres)
        and expected.excluded_movie_ids.issubset(actual.excluded_movie_ids)
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

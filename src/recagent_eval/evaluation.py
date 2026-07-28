from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recagent_eval.data import Movie
from recagent_eval.models import (
    PreferenceState,
    RecommendationResult,
    ToolName,
    ToolTrace,
)
from recagent_eval.retrieval import hard_filter


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


def _rank(movie_id: int, movie_ids: list[int]) -> int | None:
    try:
        return movie_ids.index(movie_id) + 1
    except ValueError:
        return None


def build_candidate_diagnostics(
    relevant_movie_ids: set[int],
    movies: dict[int, Movie],
    state: PreferenceState,
    traces: list[ToolTrace],
) -> list[dict[str, Any]]:
    by_tool = {trace.tool: trace for trace in traces}
    itemcf_ids = by_tool.get(
        "itemcf_retrieve",
        ToolTrace(tool="itemcf_retrieve"),
    ).candidate_movie_ids
    semantic_ids = by_tool.get(
        "semantic_retrieve",
        ToolTrace(tool="semantic_retrieve"),
    ).candidate_movie_ids
    ranked_ids = by_tool.get(
        "rerank",
        ToolTrace(tool="rerank"),
    ).candidate_movie_ids
    allowed_ids = {
        movie.movie_id for movie in hard_filter(movies.values(), state)
    }
    union_ids = set(itemcf_ids) | set(semantic_ids)
    return [
        {
            "movie_id": movie_id,
            "eligible": movie_id in allowed_ids,
            "itemcf_rank": _rank(movie_id, itemcf_ids),
            "semantic_rank": _rank(movie_id, semantic_ids),
            "union_member": movie_id in union_ids,
            "final_rank": _rank(movie_id, ranked_ids),
        }
        for movie_id in sorted(relevant_movie_ids)
    ]


def pipeline_compliant(
    traces: list[ToolTrace],
    required_tools: tuple[ToolName, ...],
) -> bool:
    names = [trace.tool for trace in traces if trace.success]
    if "hard_filter" not in names or "rerank" not in names:
        return False
    positions = [names.index(tool) for tool in required_tools if tool in names]
    return (
        len(positions) == len(required_tools)
        and positions == sorted(positions)
        and names.index("hard_filter") < min(positions)
        and max(positions) < names.index("rerank")
    )


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
    label_eligibility: list[float] = []
    final_state_eligibility: list[float] = []
    itemcf_candidate_hits: list[float] = []
    semantic_candidate_hits: list[float] = []
    union_candidate_hits: list[float] = []
    pipeline_compliance_values: list[float] = []
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
        label_eligibility.append(
            float(record.metadata.get("label_eligible", True))
        )
        pipeline_compliance_values.append(
            float(record.metadata.get("pipeline_compliant", False))
        )
        for diagnostic in record.metadata.get("candidate_diagnostics", []):
            final_state_eligibility.append(float(diagnostic["eligible"]))
            itemcf_candidate_hits.append(
                float(diagnostic["itemcf_rank"] is not None)
            )
            semantic_candidate_hits.append(
                float(diagnostic["semantic_rank"] is not None)
            )
            union_candidate_hits.append(float(diagnostic["union_member"]))
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
        "relevance_label_eligibility_rate": _mean(label_eligibility),
        "final_state_target_eligibility_rate": _mean(final_state_eligibility),
        "itemcf_candidate_recall": _mean(itemcf_candidate_hits),
        "semantic_candidate_recall": _mean(semantic_candidate_hits),
        "union_candidate_recall": _mean(union_candidate_hits),
        "pipeline_compliance_rate": _mean(pipeline_compliance_values),
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
    actual_negative_genres = actual.disliked_genres | actual.excluded_genres
    return (
        expected.liked_movie_ids.issubset(actual.liked_movie_ids)
        and expected.disliked_movie_ids.issubset(actual.disliked_movie_ids)
        and expected.liked_genres.issubset(actual.liked_genres)
        and expected.disliked_genres.issubset(actual_negative_genres)
        and expected.required_genres.issubset(actual.required_genres)
        and expected.excluded_genres.issubset(actual.excluded_genres)
        and expected.excluded_movie_ids.issubset(actual.excluded_movie_ids)
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

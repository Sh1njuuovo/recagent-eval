from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from recagent_eval.agent import build_semantic_profile
from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    build_candidate_queries,
)
from recagent_eval.models import PreferenceState
from recagent_eval.retrieval import SemanticRetriever


@dataclass(frozen=True)
class RecallVariant:
    name: str
    semantic_top_k: int
    history_cap: int
    query_style: str = "baseline"


@dataclass(frozen=True)
class RecallResult:
    variant: RecallVariant
    user_count: int
    dense_recall: float
    itemcf_recall: float
    union_recall: float
    fingerprint: str


@dataclass(frozen=True)
class RecallGateDecision:
    baseline: RecallResult
    winner: RecallResult | None
    reason: str

    @property
    def passed(self) -> bool:
        return self.winner is not None


DEFAULT_VARIANTS: tuple[RecallVariant, ...] = (
    RecallVariant("baseline-top500", 500, 50),
    RecallVariant("top250", 250, 50),
    RecallVariant("top750", 750, 50),
    RecallVariant("top1000", 1000, 50),
    RecallVariant("top1500", 1500, 50),
    RecallVariant("history20", 500, 20),
    RecallVariant("history100", 500, 100),
    RecallVariant("titles-only", 500, 50, query_style="titles-only"),
    RecallVariant(
        "titles-genres-year", 500, 50, query_style="titles-genres-year"
    ),
)


def _semantic_query_builder(
    query_style: str,
) -> Callable[[PreferenceState, dict[int, Movie], int], str]:
    def build(
        state: PreferenceState,
        movies: dict[int, Movie],
        history_cap: int,
    ) -> str:
        if query_style == "baseline":
            return build_semantic_profile("", state, movies, history_cap=history_cap)
        if query_style == "titles-only":
            titles = []
            for movie_id in sorted(state.liked_movie_ids)[:history_cap]:
                movie = movies.get(movie_id)
                if movie is not None:
                    titles.append(movie.title)
            return " ".join(titles)
        if query_style == "titles-genres-year":
            parts = list(sorted(state.liked_genres))
            for movie_id in sorted(state.liked_movie_ids)[:history_cap]:
                movie = movies.get(movie_id)
                if movie is None:
                    continue
                year = f" ({movie.year})" if movie.year is not None else ""
                parts.append(f"{movie.title}{year} {' '.join(movie.genres)}")
            return " ".join(parts)
        raise ValueError(f"unsupported query style: {query_style}")

    return build


def _variant_fingerprint(
    variant: RecallVariant,
    *,
    dataset_fingerprint: str,
    retrieval_top_k: int,
) -> str:
    payload = {
        "variant": asdict(variant),
        "dataset_fingerprint": dataset_fingerprint,
        "retrieval_top_k": retrieval_top_k,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_recall_sweep(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    *,
    retrieval_top_k: int,
    max_users: int,
    dataset_fingerprint: str = "unspecified",
    variants: Sequence[RecallVariant] = DEFAULT_VARIANTS,
) -> tuple[list[RecallResult], RecallGateDecision]:
    """Measure per-user candidate recall for each variant on validation users."""
    histories = _positive_histories(split.legal_retrieval_train, movies)
    results: list[RecallResult] = []
    for variant in variants:
        queries = build_candidate_queries(
            movies,
            split.legal_retrieval_train,
            histories,
            split.validation_targets,
            semantic,
            retrieval_top_k=retrieval_top_k,
            history_cap=variant.history_cap,
            max_users=max_users,
            semantic_top_k=variant.semantic_top_k,
            semantic_query_builder=_semantic_query_builder(variant.query_style),
        )
        dense = itemcf = union = 0
        for query in queries:
            if query.target_movie_id not in query.features_by_movie:
                continue
            union += 1
            if query.features_by_movie[query.target_movie_id][9] == 1.0:
                dense += 1
            if query.features_by_movie[query.target_movie_id][8] == 1.0:
                itemcf += 1
        count = len(queries)
        results.append(
            RecallResult(
                variant=variant,
                user_count=count,
                dense_recall=dense / count if count else 0.0,
                itemcf_recall=itemcf / count if count else 0.0,
                union_recall=union / count if count else 0.0,
                fingerprint=_variant_fingerprint(
                    variant,
                    dataset_fingerprint=dataset_fingerprint,
                    retrieval_top_k=retrieval_top_k,
                ),
            )
        )
    return results, select_recall_winner(results)


def select_recall_winner(
    results: Sequence[RecallResult],
    *,
    baseline_name: str = "baseline-top500",
    dense_lift: float = 0.05,
    union_floor: float | None = None,
) -> RecallGateDecision:
    """Pick the variant that lifts dense recall without losing union coverage."""
    if not results:
        raise ValueError("recall sweep produced no results")
    baselines = [result for result in results if result.variant.name == baseline_name]
    if not baselines:
        raise ValueError(f"baseline variant {baseline_name!r} missing from sweep")
    baseline = baselines[0]
    floor = baseline.union_recall if union_floor is None else union_floor
    candidates = [
        result
        for result in results
        if result is not baseline
        and result.dense_recall >= baseline.dense_recall + dense_lift
        and result.union_recall > floor
    ]
    winner = max(
        candidates,
        key=lambda result: (result.dense_recall, result.union_recall),
        default=None,
    )
    if winner is None:
        return RecallGateDecision(
            baseline=baseline,
            winner=None,
            reason=(
                f"no variant lifted dense recall by >= {dense_lift:.3f} "
                f"above baseline {baseline.dense_recall:.3f} while keeping "
                f"union recall above {floor:.3f}"
            ),
        )
    return RecallGateDecision(
        baseline=baseline,
        winner=winner,
        reason=(
            f"selected {winner.variant.name}: dense recall "
            f"{winner.dense_recall:.3f} vs baseline {baseline.dense_recall:.3f}, "
            f"union recall {winner.union_recall:.3f} vs {baseline.union_recall:.3f}"
        ),
    )

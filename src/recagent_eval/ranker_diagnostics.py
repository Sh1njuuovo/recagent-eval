from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from recagent_eval.candidate_features import FEATURE_NAMES
from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    CandidateQuery,
    _positive_histories,
    build_candidate_queries,
)
from recagent_eval.learned_ranking import LearnedRanker
from recagent_eval.retrieval import SemanticRetriever


@dataclass(frozen=True)
class DiagnosticUserRow:
    user_id: int
    in_union: bool
    in_itemcf: bool
    in_dense: bool
    itemcf_top10: bool
    lambdamart_top10: bool
    itemcf_ndcg_at_10: float
    lambdamart_ndcg_at_10: float
    target_itemcf_rank: int | None
    target_lambdamart_rank: int | None
    target_features: tuple[float, ...] | None
    negative_mean_features: tuple[float, ...] | None


@dataclass(frozen=True)
class DiagnosticSummary:
    user_count: int
    present_user_count: int
    union_recall: float
    itemcf_recall: float
    dense_recall: float
    itemcf_top10_hit: float
    lambdamart_top10_hit: float
    itemcf_top10_hit_present: float
    lambdamart_top10_hit_present: float
    itemcf_ndcg_at_10: float
    lambdamart_ndcg_at_10: float
    itemcf_ndcg_at_10_present: float
    lambdamart_ndcg_at_10_present: float
    target_itemcf_rank_quantiles: dict[str, float]
    target_lambdamart_rank_quantiles: dict[str, float]
    feature_separation: dict[str, float]
    fingerprints: dict[str, str]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantiles(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    ordered = sorted(ranks)
    count = len(ordered)
    result: dict[str, float] = {}
    for label, position in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
        index = min(count - 1, int(position * count))
        result[label] = float(ordered[index])
    return result


def aggregate_diagnostics(
    rows: Sequence[DiagnosticUserRow],
    *,
    fingerprints: Mapping[str, str] | None = None,
) -> DiagnosticSummary:
    """Aggregate per-user diagnostics with target-present conditioning."""
    if not rows:
        raise ValueError("diagnostics produced no user rows")
    present = [row for row in rows if row.in_union]
    separation: dict[str, float] = {}
    for index, name in enumerate(FEATURE_NAMES):
        deltas = [
            row.target_features[index] - row.negative_mean_features[index]
            for row in present
            if row.target_features is not None
            and row.negative_mean_features is not None
        ]
        separation[name] = _mean(deltas)
    itemcf_ranks = [
        row.target_itemcf_rank
        for row in present
        if row.target_itemcf_rank is not None
    ]
    lambdamart_ranks = [
        row.target_lambdamart_rank
        for row in present
        if row.target_lambdamart_rank is not None
    ]
    return DiagnosticSummary(
        user_count=len(rows),
        present_user_count=len(present),
        union_recall=len(present) / len(rows),
        itemcf_recall=sum(row.in_itemcf for row in rows) / len(rows),
        dense_recall=sum(row.in_dense for row in rows) / len(rows),
        itemcf_top10_hit=_mean([row.itemcf_top10 for row in rows]),
        lambdamart_top10_hit=_mean([row.lambdamart_top10 for row in rows]),
        itemcf_top10_hit_present=_mean([row.itemcf_top10 for row in present]),
        lambdamart_top10_hit_present=_mean(
            [row.lambdamart_top10 for row in present]
        ),
        itemcf_ndcg_at_10=_mean([row.itemcf_ndcg_at_10 for row in rows]),
        lambdamart_ndcg_at_10=_mean(
            [row.lambdamart_ndcg_at_10 for row in rows]
        ),
        itemcf_ndcg_at_10_present=_mean(
            [row.itemcf_ndcg_at_10 for row in present]
        ),
        lambdamart_ndcg_at_10_present=_mean(
            [row.lambdamart_ndcg_at_10 for row in present]
        ),
        target_itemcf_rank_quantiles=_quantiles(itemcf_ranks),
        target_lambdamart_rank_quantiles=_quantiles(lambdamart_ranks),
        feature_separation=separation,
        fingerprints=dict(fingerprints or {}),
    )


def build_user_diagnostics(
    queries: Sequence[CandidateQuery],
    movies: dict[int, Movie],
    learned: LearnedRanker,
) -> list[DiagnosticUserRow]:
    """Build per-user diagnostic rows from candidate queries and the ranker."""
    rows: list[DiagnosticUserRow] = []
    for query in queries:
        features_by_movie = query.features_by_movie
        target = query.target_movie_id
        in_union = target in features_by_movie
        target_row = features_by_movie.get(target)
        negatives = [
            values
            for movie_id, values in features_by_movie.items()
            if movie_id != target
        ]
        if target_row is not None:
            target_features = tuple(target_row)
            negative_mean = tuple(
                sum(values[index] for values in negatives) / len(negatives)
                if negatives
                else 0.0
                for index in range(len(FEATURE_NAMES))
            )
        else:
            target_features = None
            negative_mean = None
        itemcf_order = sorted(
            features_by_movie,
            key=lambda movie_id: (
                -features_by_movie[movie_id][0],
                movie_id,
            ),
        )
        itemcf_rank = itemcf_order.index(target) + 1 if in_union else None
        itemcf_top10 = in_union and target in itemcf_order[:10]
        itemcf_ndcg = _single_ndcg(itemcf_order[:10], target) if in_union else 0.0
        ranked = learned.rank_feature_rows(movies, features_by_movie, top_k=10)
        lambdamart_ids = [item.movie_id for item in ranked]
        lambdamart_top10 = in_union and target in lambdamart_ids
        lambdamart_ndcg = _single_ndcg(lambdamart_ids, target) if in_union else 0.0
        lambdamart_rank = None
        if in_union:
            predictions = [
                float(value)
                for value in learned.estimator.predict(
                    [features_by_movie[movie_id] for movie_id in features_by_movie]
                )
            ]
            score_by_movie = dict(zip(features_by_movie, predictions, strict=True))
            lambdamart_order = sorted(
                score_by_movie,
                key=lambda movie_id: (-score_by_movie[movie_id], movie_id),
            )
            lambdamart_rank = lambdamart_order.index(target) + 1
        rows.append(
            DiagnosticUserRow(
                user_id=query.user_id,
                in_union=in_union,
                in_itemcf=bool(target_row is not None and target_row[8] == 1.0),
                in_dense=bool(target_row is not None and target_row[9] == 1.0),
                itemcf_top10=itemcf_top10,
                lambdamart_top10=lambdamart_top10,
                itemcf_ndcg_at_10=itemcf_ndcg,
                lambdamart_ndcg_at_10=lambdamart_ndcg,
                target_itemcf_rank=itemcf_rank,
                target_lambdamart_rank=lambdamart_rank,
                target_features=target_features,
                negative_mean_features=negative_mean,
            )
        )
    return rows


def _single_ndcg(ranked_ids: Sequence[int], target: int) -> float:
    if target not in ranked_ids[:10]:
        return 0.0
    return 1.0 / math.log2(ranked_ids.index(target) + 2)


def build_diagnostic_queries(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    *,
    retrieval_top_k: int,
    history_cap: int,
    semantic_top_k: int | None,
    score_calibration: str,
    max_users: int,
):
    histories = _positive_histories(split.legal_retrieval_train, movies)
    return build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        histories,
        split.validation_targets,
        semantic,
        retrieval_top_k=retrieval_top_k,
        history_cap=history_cap,
        semantic_top_k=semantic_top_k,
        score_calibration=score_calibration,
        max_users=max_users,
    )

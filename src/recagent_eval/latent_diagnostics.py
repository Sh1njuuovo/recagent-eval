from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    build_candidate_queries,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.retrieval import SemanticRetriever


@dataclass(frozen=True)
class LatentDiagnosticUserRow:
    user_id: int
    in_union: bool
    in_itemcf: bool
    in_dense: bool
    in_latent: bool
    latent_rank: int | None
    latent_recall_10: float
    latent_recall_50: float
    latent_recall_100: float
    latent_recall_500: float
    latent_only: bool
    itemcf_ids: frozenset[int]
    dense_ids: frozenset[int]
    latent_ids: frozenset[int]


@dataclass(frozen=True)
class LatentDiagnosticSummary:
    user_count: int
    latent_present_user_count: int
    latent_recall_500_all: float
    latent_recall_100_all: float
    latent_recall_50_all: float
    latent_recall_10_all: float
    latent_recall_500_present: float
    latent_recall_10_present: float
    union_recall_three_route: float
    latent_only_coverage: float
    target_latent_rank_quantiles: dict[str, float]
    overlap_itemcf_latent: float
    overlap_dense_latent: float
    fit_seconds: float
    fingerprints: dict[str, str]


def build_latent_diagnostic_queries(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    *,
    latent: LatentFactorRetriever,
    retrieval_top_k: int,
    history_cap: int,
    semantic_top_k: int | None,
    latent_top_k: int,
    feature_version: str,
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
        latent=latent,
        latent_top_k=latent_top_k,
        feature_version=feature_version,
        max_users=max_users,
    )


def build_latent_user_rows(queries: Sequence) -> list[LatentDiagnosticUserRow]:
    rows: list[LatentDiagnosticUserRow] = []
    for query in queries:
        features = query.features_by_movie
        if features:
            sample = next(iter(features.values()))
            if len(sample) != 13:
                raise ValueError("latent diagnostics require candidate-features/v2 rows")
        target = query.target_movie_id
        in_union = target in features
        target_row = features.get(target)
        in_itemcf = bool(target_row is not None and target_row[8] == 1.0)
        in_dense = bool(target_row is not None and target_row[9] == 1.0)
        in_latent = bool(target_row is not None and target_row[10] == 1.0)
        itemcf_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[8] == 1.0
        )
        dense_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[9] == 1.0
        )
        latent_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[12] == 1.0
        )
        latent_order = [
            movie_id
            for movie_id in sorted(
                features,
                key=lambda movie_id: (-features[movie_id][10], movie_id),
            )
            if features[movie_id][12] == 1.0
        ]
        latent_rank = latent_order.index(target) + 1 if in_latent else None
        rows.append(
            LatentDiagnosticUserRow(
                user_id=query.user_id,
                in_union=in_union,
                in_itemcf=in_itemcf,
                in_dense=in_dense,
                in_latent=in_latent,
                latent_rank=latent_rank,
                latent_recall_10=float(
                    in_latent and latent_rank is not None and latent_rank <= 10
                ),
                latent_recall_50=float(
                    in_latent and latent_rank is not None and latent_rank <= 50
                ),
                latent_recall_100=float(
                    in_latent and latent_rank is not None and latent_rank <= 100
                ),
                latent_recall_500=float(in_latent),
                latent_only=bool(in_latent and not in_itemcf and not in_dense),
                itemcf_ids=itemcf_ids,
                dense_ids=dense_ids,
                latent_ids=latent_ids,
            )
        )
    return rows


def aggregate_latent_diagnostics(
    rows: Sequence[LatentDiagnosticUserRow],
    *,
    fingerprints: Mapping[str, str] | None = None,
    fit_seconds: float = 0.0,
) -> LatentDiagnosticSummary:
    if not rows:
        raise ValueError("latent diagnostics produced no user rows")
    present = [row for row in rows if row.in_latent]
    ranks = [row.latent_rank for row in present if row.latent_rank is not None]
    union = [row for row in rows if row.in_union]
    return LatentDiagnosticSummary(
        user_count=len(rows),
        latent_present_user_count=len(present),
        latent_recall_500_all=_mean([row.latent_recall_500 for row in rows]),
        latent_recall_100_all=_mean([row.latent_recall_100 for row in rows]),
        latent_recall_50_all=_mean([row.latent_recall_50 for row in rows]),
        latent_recall_10_all=_mean([row.latent_recall_10 for row in rows]),
        latent_recall_500_present=_mean([row.latent_recall_500 for row in present]),
        latent_recall_10_present=_mean([row.latent_recall_10 for row in present]),
        union_recall_three_route=len(union) / len(rows),
        latent_only_coverage=sum(row.latent_only for row in rows) / len(rows),
        target_latent_rank_quantiles=_quantiles(ranks),
        overlap_itemcf_latent=_route_overlap(rows, left="itemcf", right="latent"),
        overlap_dense_latent=_route_overlap(rows, left="dense", right="latent"),
        fit_seconds=fit_seconds,
        fingerprints=dict(fingerprints or {}),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantiles(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    ordered = sorted(ranks)
    result: dict[str, float] = {}
    for label, position in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
        index = min(len(ordered) - 1, int(position * len(ordered)))
        result[label] = float(ordered[index])
    return result


def _route_overlap(
    rows: Sequence[LatentDiagnosticUserRow], *, left: str, right: str
) -> float:
    values = []
    for row in rows:
        left_ids = getattr(row, f"{left}_ids")
        right_ids = getattr(row, f"{right}_ids")
        union = left_ids | right_ids
        if not union:
            values.append(0.0)
        else:
            values.append(len(left_ids & right_ids) / len(union))
    return _mean(values)

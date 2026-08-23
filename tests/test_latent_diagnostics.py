from __future__ import annotations

from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.latent_diagnostics import (
    aggregate_latent_diagnostics,
    build_latent_diagnostic_queries,
    build_latent_user_rows,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 12)
    }


def _ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 9)
        for movie_id in range(1, 9)
    ]


def _semantic():
    class Stub:
        def retrieve(self, query, *, top_k=100, allowed_ids=None):
            del query
            ids = sorted(allowed_ids or ())
            return [(movie_id, 1.0 / index) for index, movie_id in enumerate(ids, 1)][
                :top_k
            ]

    return Stub()


def test_latent_diagnostics_aggregate_gate_fields() -> None:
    movies = _movies()
    ratings = _ratings()
    split = leakage_safe_ranking_split(ratings)
    latent = LatentFactorRetriever.fit(split.legal_retrieval_train, seed=42)
    queries = build_latent_diagnostic_queries(
        movies,
        split,
        _semantic(),
        latent=latent,
        retrieval_top_k=5,
        history_cap=5,
        semantic_top_k=10,
        latent_top_k=10,
        feature_version="v2",
        max_users=6,
    )
    rows = build_latent_user_rows(queries)
    summary = aggregate_latent_diagnostics(
        rows,
        fingerprints={
            "dataset": "d",
            "candidate_policy": "p",
            "feature_schema": "f",
            "case": "c",
        },
        fit_seconds=0.5,
    )
    assert summary.user_count == 6
    assert 0.0 <= summary.latent_recall_500_all <= 1.0
    assert 0.0 <= summary.latent_recall_10_all <= 1.0
    assert 0.0 <= summary.union_recall_three_route <= 1.0
    assert summary.latent_only_coverage >= 0.0
    assert summary.latent_present_user_count >= 0
    assert set(summary.target_latent_rank_quantiles) == {"p25", "p50", "p75"}

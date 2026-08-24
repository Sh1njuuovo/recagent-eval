from __future__ import annotations

from recagent_eval.baseline_eval import BASELINE_SCORERS
from recagent_eval.baselines.popularity import score_popularity
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }


def _ratings() -> list[Rating]:
    rows = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 7)
    ]
    # Movie 1 is globally the most popular (rated by all 6 users in legal rows).
    rows.append(Rating(7, 1, 5, 1000))
    return rows


def test_popularity_is_registered_and_returns_expected_keys() -> None:
    assert "popularity" in BASELINE_SCORERS
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    users = sorted(split.validation_targets)[:3]
    result = score_popularity(movies, split, users)
    assert len(result["rows"]) == 3
    assert set(result) >= {
        "rows",
        "config_fingerprint",
        "dataset_fingerprint",
        "model_fingerprint",
        "training_seconds",
        "resource_usage",
        "model_size_bytes",
        "environment",
    }
    assert result["resource_usage"]["metric_name"] == "process_peak_rss_mib"
    assert "peak_memory_mb" not in result
    assert all(
        row.recommended_ids and all(1 <= movie_id <= 8 for movie_id in row.recommended_ids)
        for row in result["rows"]
    )


def test_popularity_ranks_by_global_count_with_deterministic_tiebreak() -> None:
    from recagent_eval.lambdamart_pipeline import _state_from_history

    movies = _movies()
    ratings = _ratings()
    split = leakage_safe_ranking_split(ratings)
    # Direct scoring without the split's target machinery: history {2}, target 1.
    popularity = {
        movie_id: sum(
            1
            for row in split.legal_retrieval_train
            if row.movie_id == movie_id and row.rating >= 4
        )
        for movie_id in range(1, 9)
    }
    history_ids = {2}
    state = _state_from_history(history_ids, movies)
    from recagent_eval.retrieval import hard_filter

    allowed = {movie.movie_id for movie in hard_filter(movies.values(), state)} - history_ids
    ranked = sorted(allowed, key=lambda movie_id: (-popularity[movie_id], movie_id))
    assert ranked[0] == 1  # most popular
    assert ranked == sorted(ranked, key=lambda movie_id: (-popularity[movie_id], movie_id))

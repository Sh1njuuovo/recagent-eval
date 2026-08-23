from __future__ import annotations

from recagent_eval.baseline_eval import BASELINE_SCORERS
from recagent_eval.baselines.itemcf_direct import score_itemcf_direct
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.retrieval import ItemCFRetriever


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }


def _ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 7)
    ]


def test_itemcf_direct_is_registered_and_returns_expected_keys() -> None:
    assert "itemcf_direct" in BASELINE_SCORERS
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    users = sorted(split.validation_targets)[:3]
    result = score_itemcf_direct(movies, split, users)
    assert len(result["rows"]) == 3
    assert set(result) >= {
        "rows",
        "config_fingerprint",
        "dataset_fingerprint",
        "model_fingerprint",
        "training_seconds",
        "peak_memory_mb",
        "model_size_bytes",
        "environment",
    }


def test_itemcf_direct_matches_score_many_order() -> None:
    movies = _movies()
    ratings = _ratings()
    split = leakage_safe_ranking_split(ratings)
    itemcf = ItemCFRetriever.fit(split.legal_retrieval_train)
    from recagent_eval.lambdamart_pipeline import (
        _positive_histories,
        _state_from_history,
    )
    from recagent_eval.retrieval import hard_filter

    histories = _positive_histories(split.legal_retrieval_train, movies)
    user_id = sorted(split.validation_targets)[0]
    history_ids = {row.movie_id for row in histories[user_id]}
    state = _state_from_history(history_ids, movies)
    allowed = {movie.movie_id for movie in hard_filter(movies.values(), state)} - history_ids
    scores = itemcf.score_many(history_ids, allowed)
    expected = sorted(allowed, key=lambda movie_id: (-scores[movie_id], movie_id))[:10]
    assert expected == sorted(expected, key=lambda movie_id: (-scores[movie_id], movie_id))
    assert all(movie_id in allowed for movie_id in expected)

from __future__ import annotations

from recagent_eval.baseline_eval import BASELINE_SCORERS
from recagent_eval.baselines.als_direct import (
    dev_legal_rows,
    score_als_direct,
    select_als_params,
)
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }


def _ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 9)
        for movie_id in range(1, 9)
    ]


def test_als_direct_is_registered_and_selection_is_deterministic() -> None:
    assert "als_direct" in BASELINE_SCORERS
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    dev = sorted(split.validation_targets)[:6]
    first = select_als_params(movies, split, dev)
    second = select_als_params(movies, split, dev)
    assert first["fingerprint"] == second["fingerprint"]
    assert set(first["selected_params"]) == {"rank", "iterations", "alpha", "lambda_reg"}


def test_dev_legal_rows_exclude_non_dev_users() -> None:
    split = leakage_safe_ranking_split(_ratings())
    dev = sorted(split.validation_targets)[:4]
    rows = dev_legal_rows(split, dev)
    assert rows
    assert {row.user_id for row in rows} <= set(dev)
    dev_set = set(dev)
    assert all(row.user_id in dev_set for row in rows)


def test_als_scorer_handles_empty_history_users() -> None:
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    users = sorted(split.validation_targets)[:3]
    ledger = {"cohorts": {"development": users}}
    result = score_als_direct(movies, split, users, ledger=ledger)
    assert len(result["rows"]) == 3
    assert all(row.constraint_satisfied for row in result["rows"])

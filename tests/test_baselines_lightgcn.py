from __future__ import annotations

import json

import numpy as np
import pytest

import recagent_eval.baselines.lightgcn as module
from recagent_eval.baseline_eval import BASELINE_SCORERS
from recagent_eval.baselines.lightgcn import (
    LightGCN,
    score_lightgcn,
    select_lightgcn_params,
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
        for user_id in range(1, 13)
        for movie_id in range(1, 9)
    ]


def test_lightgcn_fit_is_bit_identical_across_two_runs() -> None:
    rows = tuple(r for r in _ratings())
    first = LightGCN.fit(rows, rank=8, layers=2, learning_rate=1e-3, reg=1e-3, epochs=3, seed=42)
    second = LightGCN.fit(rows, rank=8, layers=2, learning_rate=1e-3, reg=1e-3, epochs=3, seed=42)
    assert np.array_equal(first.user_embeddings, second.user_embeddings)
    assert np.array_equal(first.item_embeddings, second.item_embeddings)
    assert first.training_fingerprint == second.training_fingerprint


def test_lightgcn_uses_only_supplied_rows() -> None:
    rows = tuple(r for r in _ratings())
    legal = rows[: len(rows) // 2]
    model = LightGCN.fit(legal, rank=8, layers=2, learning_rate=1e-3, reg=1e-3, epochs=2, seed=1)
    assert set(model.user_ids.tolist()) <= {row.user_id for row in legal}
    assert set(model.item_ids.tolist()) <= {row.movie_id for row in legal}


def test_lightgcn_save_load_roundtrip_and_checksum(tmp_path) -> None:
    rows = tuple(r for r in _ratings())
    model = LightGCN.fit(rows, rank=8, layers=2, learning_rate=1e-3, reg=1e-3, epochs=2, seed=7)
    path = tmp_path / "lightgcn.npz"
    model.save(path)
    loaded = LightGCN.load(path)
    assert np.array_equal(loaded.user_embeddings, model.user_embeddings)
    assert np.array_equal(loaded.item_embeddings, model.item_embeddings)
    manifest = json.loads((tmp_path / "lightgcn.npz.json").read_text())
    assert manifest["training_fingerprint"] == model.training_fingerprint
    with pytest.raises(ValueError, match="overwrite"):
        model.save(path)


def test_lightgcn_scores_are_finite_and_empty_for_unknown_user() -> None:
    rows = tuple(r for r in _ratings())
    model = LightGCN.fit(rows, rank=8, layers=2, learning_rate=1e-3, reg=1e-3, epochs=2, seed=3)
    scores = model.score_user(rows[0].user_id, {1, 2, 3, 4, 5, 6, 7, 8})
    assert scores
    assert all(np.isfinite(value) for value in scores.values())
    assert model.score_user(9999, {1, 2}) == {}


def test_lightgcn_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="rank"):
        LightGCN.fit((), rank=0)
    with pytest.raises(ValueError, match="layers"):
        LightGCN.fit((), layers=0)


def test_lightgcn_scorer_registered_and_selection_deterministic() -> None:
    assert "lightgcn" in BASELINE_SCORERS
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    dev = sorted(split.validation_targets)[:6]
    first = select_lightgcn_params(movies, split, dev)
    second = select_lightgcn_params(movies, split, dev)
    assert first["fingerprint"] == second["fingerprint"]
    assert set(first["selected_params"]) == {"rank", "layers", "learning_rate", "reg"}


def test_lightgcn_scorer_handles_empty_history_users() -> None:
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    users = sorted(split.validation_targets)[:3]
    ledger = {"cohorts": {"development": users}}
    result = score_lightgcn(movies, split, users, ledger=ledger)
    assert len(result["rows"]) == 3


def test_lightgcn_scorer_uses_locked_params_without_selection(monkeypatch) -> None:
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    users = sorted(split.validation_targets)[:3]
    monkeypatch.setattr(
        module,
        "select_lightgcn_params",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("grid rerun")),
    )
    result = score_lightgcn(
        movies,
        split,
        users,
        selected_params={
            "rank": 8,
            "layers": 2,
            "learning_rate": 1e-3,
            "reg": 1e-3,
            "epochs": 1,
        },
        selection_fingerprint="s" * 64,
        seed=2026,
    )
    assert result["seed"] == 2026
    assert result["config_fingerprint"] == "s" * 64

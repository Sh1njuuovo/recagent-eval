from __future__ import annotations

import pytest

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V2B,
    FEATURE_SCHEMA_FINGERPRINT,
    FEATURE_SCHEMA_FINGERPRINT_V2,
    FEATURE_SCHEMA_FINGERPRINT_V2B,
    build_candidate_feature_rows,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState


def _movies() -> dict[int, Movie]:
    return {
        1: Movie(1, "A", ("Drama",), 1995),
        2: Movie(2, "B", ("Comedy",), 1998),
        3: Movie(3, "C", ("Drama", "Comedy"), 2001),
    }


def _history() -> tuple[Rating, ...]:
    return (Rating(7, 1, 5, 10), Rating(7, 2, 4, 20))


def _state() -> PreferenceState:
    return PreferenceState(liked_movie_ids={1, 2}, liked_genres={"Drama"})


def _scores() -> dict[int, float]:
    return {1: 5.0, 2: 3.0}


def _latent() -> dict[int, float]:
    return {1: 0.8, 3: 0.4}


def test_v1_default_behavior_and_fingerprint_unchanged() -> None:
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        history=_history(),
        train_rows=_history(),
        state=_state(),
    )
    assert len(rows) == 2  # only itemcf/dense candidates under v1
    assert all(len(row.values) == len(FEATURE_NAMES) for row in rows)
    assert FEATURE_SCHEMA_FINGERPRINT == FEATURE_SCHEMA_FINGERPRINT


def test_v2_adds_latent_features_and_fingerprint() -> None:
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        latent_scores=_latent(),
        history=_history(),
        train_rows=_history(),
        state=_state(),
        feature_version="v2",
    )
    assert len(FEATURE_NAMES_V2) == len(FEATURE_NAMES) + 3
    assert all(len(row.values) == len(FEATURE_NAMES_V2) for row in rows)
    by_id = {row.movie_id: row.as_dict() for row in rows}
    assert by_id[1]["latent_score"] == 0.8
    assert by_id[2]["latent_score"] == 0.0
    assert by_id[2]["in_latent"] == 0.0
    assert by_id[3]["in_latent"] == 1.0
    assert FEATURE_SCHEMA_FINGERPRINT_V2 != FEATURE_SCHEMA_FINGERPRINT


def test_v2b_adds_cross_recent_year_features() -> None:
    recent = {1: 6.0, 2: 1.0}
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        latent_scores=_latent(),
        recent_itemcf_scores=recent,
        history=_history(),
        train_rows=_history(),
        state=_state(),
        feature_version="v2b",
    )
    assert len(FEATURE_NAMES_V2B) == len(FEATURE_NAMES_V2) + 3
    assert all(len(row.values) == len(FEATURE_NAMES_V2B) for row in rows)
    assert FEATURE_SCHEMA_FINGERPRINT_V2B != FEATURE_SCHEMA_FINGERPRINT_V2


def test_unknown_feature_version_fails() -> None:
    with pytest.raises(ValueError, match="feature_version"):
        build_candidate_feature_rows(
            user_id=7,
            movies=_movies(),
            itemcf_scores=_scores(),
            dense_scores=_scores(),
            history=_history(),
            train_rows=_history(),
            state=_state(),
            feature_version="v9",
        )

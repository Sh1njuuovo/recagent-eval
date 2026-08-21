from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    build_candidate_feature_rows,
)
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.learned_ranking import (
    DEFAULT_PARAMETER_GRID,
    CandidateQuery,
    LearnedRanker,
    RankerArtifact,
    build_training_matrix,
    load_ranker_artifact,
    make_lgbm_ranker,
    save_ranker_artifact,
)
from recagent_eval.models import PreferenceState, ScoreBreakdown


def _valid_artifact(**overrides) -> RankerArtifact:
    fold_map = {1: 0, 2: 1, 3: 2}
    cv_results = [
        {"params": params, "mean_ndcg_at_10": 0.0, "mean_recall_at_10": 0.0}
        for params in DEFAULT_PARAMETER_GRID
    ] + [
        {
            "params": params,
            "fold": fold,
            "train_users": [user for user in fold_map if fold_map[user] != fold],
            "validation_users": [user for user in fold_map if fold_map[user] == fold],
            "ndcg_at_10": 0.0,
            "recall_at_10": 0.0,
        }
        for params in DEFAULT_PARAMETER_GRID
        for fold in range(3)
    ]
    values = {
        "selected_params": {
            "num_leaves": 15,
            "learning_rate": 0.03,
            "n_estimators": 100,
            "min_child_samples": 50,
        },
        "dataset_fingerprint": "dataset",
        "training_user_count": 3,
        "training_group_count": 3,
        "dependency_versions": {"lightgbm": "test"},
        "model_string": "model contents",
        "training_rows_fingerprint": "train",
        "history_fingerprint": "history",
        "fold_map_fingerprint": hashlib.sha256(
            json.dumps(fold_map, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "group_fingerprint": "groups",
        "candidate_policy_fingerprint": "policy",
        "config_fingerprint": "config",
        "metric_fingerprint": "metric",
        "case_fingerprint": "cases",
        "report_fingerprint": hashlib.sha256(
            json.dumps(cv_results, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cv_results": cv_results,
        "fold_map": fold_map,
        "validation_rows_fingerprint": "validation",
        "validation_user_count": 3,
    }
    values.update(overrides)
    return RankerArtifact(**values)


def _ratings() -> list[Rating]:
    return [
        Rating(1, 99, 2, 5),
        Rating(1, 1, 5, 10),
        Rating(1, 2, 4, 20),
        Rating(1, 3, 5, 30),
        Rating(1, 4, 5, 40),
        Rating(1, 100, 1, 50),
        Rating(2, 5, 5, 10),
        Rating(2, 6, 5, 20),
        Rating(2, 7, 5, 30),
    ]


def test_leakage_safe_split_has_three_disjoint_targets_and_ordered_history() -> None:
    split = leakage_safe_ranking_split(list(reversed(_ratings())))

    assert split.ranker_targets == {1: 2, 2: 5}
    assert split.validation_targets == {1: 3, 2: 6}
    assert split.test_targets == {1: 4, 2: 7}
    assert [row.movie_id for row in split.histories[1]] == [99, 1]
    assert [row.movie_id for row in split.legal_retrieval_train if row.user_id == 1] == [99, 1]
    assert set(split.ranker_targets) == {1, 2}
    assert len({split.ranker_targets[1], split.validation_targets[1], split.test_targets[1]}) == 3


def test_split_uses_latest_three_distinct_movies_and_removes_all_target_rows() -> None:
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(1, 2, 4, 3),
        Rating(1, 3, 5, 4),
        Rating(1, 4, 5, 5),
    ]

    split = leakage_safe_ranking_split(ratings)

    assert split.ranker_targets == {1: 2}
    assert split.validation_targets == {1: 3}
    assert split.test_targets == {1: 4}
    assert [row.movie_id for row in split.histories[1]] == [1]
    assert [row.movie_id for row in split.legal_retrieval_train] == [1]


def test_feature_schema_values_are_finite_and_routes_have_presence_flags() -> None:
    movies = {
        1: Movie(1, "Old", ("Drama",), 2000),
        2: Movie(2, "Candidate", ("Drama", "Comedy"), 2000),
        3: Movie(3, "Dense", ("Action",), None),
    }
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=movies,
        candidate_ids={2, 3},
        itemcf_scores={2: 0.8},
        dense_scores={3: 0.8},
        history=(Rating(7, 1, 5, 10),),
        train_rows=(Rating(8, 2, 5, 1), Rating(9, 2, 5, 2)),
        state=PreferenceState(liked_genres={"Drama"}),
    )

    assert FEATURE_NAMES == (
        "itemcf_score",
        "itemcf_reciprocal_rank",
        "dense_score",
        "dense_reciprocal_rank",
        "log1p_popularity",
        "history_genre_jaccard",
        "history_year_match",
        "preference_affinity",
        "in_itemcf",
        "in_dense",
    )
    assert [row.movie_id for row in rows] == [2, 3]
    assert rows[0].values[8:] == (1.0, 0.0)
    assert rows[1].values[0] == rows[1].values[1] == 0.0
    assert all(math.isfinite(value) for row in rows for value in row.values)


def test_feature_builder_rejects_nonfinite_with_context() -> None:
    with pytest.raises(ValueError, match=r"user=7.*movie=2.*itemcf_score"):
        build_candidate_feature_rows(
            user_id=7,
            movies={2: Movie(2, "Candidate", (), 2000)},
            candidate_ids={2},
            itemcf_scores={2: float("nan")},
            dense_scores={},
            history=(),
            train_rows=(),
            state=PreferenceState(),
        )


class _FakeEstimator:
    def fit(self, features, labels, *, group):
        self.group = list(group)
        return self

    def predict(self, features, *, pred_contrib=False):
        if pred_contrib:
            return [[*row, 0.25] for row in features]
        return [sum(row) + 0.25 for row in features]


def test_training_matrix_skips_target_misses_but_keeps_denominator() -> None:
    matrix = build_training_matrix(
        [
            CandidateQuery(1, 2, {2: (1.0,) * 10, 3: (0.0,) * 10}),
            CandidateQuery(2, 9, {4: (0.0,) * 10}),
        ]
    )

    assert matrix.groups == (2,)
    assert matrix.labels == (1, 0)
    assert matrix.evaluation_users == 2
    assert matrix.training_users == 1


def test_learned_ranker_ties_by_movie_id_and_populates_contributions() -> None:
    ranker = LearnedRanker(_FakeEstimator())
    movies = {1: Movie(1, "A", ()), 2: Movie(2, "B", ())}
    features = {2: (0.0,) * 10, 1: (0.0,) * 10}

    ranked = ranker.rank_feature_rows(movies, features, top_k=10)

    assert [movie.movie_id for movie in ranked] == [1, 2]
    assert ranked[0].score.feature_contributions["bias"] == 0.25
    assert sum(ranked[0].score.feature_contributions.values()) == pytest.approx(
        ranked[0].score.final
    )
    assert ScoreBreakdown.model_validate({"final": 1}).feature_contributions == {}


def test_learned_ranker_supports_public_hybrid_rank_signature() -> None:
    ranker = LearnedRanker(
        _FakeEstimator(),
        legal_train_rows=(Rating(2, 2, 5, 1),),
    )
    movies = {1: Movie(1, "A", ()), 2: Movie(2, "B", ())}

    ranked = ranker.rank(
        movies,
        itemcf_scores={1: 0.0},
        semantic_scores={2: 0.0},
        state=PreferenceState(),
    )

    assert [movie.movie_id for movie in ranked] == [2, 1]


def test_ranker_artifact_rejects_tampering(tmp_path: Path) -> None:
    artifact = _valid_artifact()
    path = tmp_path / "ranker.json"
    save_ranker_artifact(artifact, path)
    assert load_ranker_artifact(path, expected_dataset_fingerprint="dataset") == artifact

    payload = json.loads(path.read_text())
    payload["model_string"] += "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_ranker_artifact(path, expected_dataset_fingerprint="dataset")


def test_ranker_artifact_rejects_missing_or_extra_provenance(tmp_path: Path) -> None:
    artifact = _valid_artifact(model_string="model")
    path = tmp_path / "ranker.json"
    save_ranker_artifact(artifact, path)
    payload = json.loads(path.read_text())
    payload.pop("fold_map_fingerprint")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required fields"):
        load_ranker_artifact(path)

    payload["fold_map_fingerprint"] = "unspecified"
    payload["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extra"):
        load_ranker_artifact(path)


def test_real_lightgbm_ranker_smoke_uses_query_groups() -> None:
    matrix = build_training_matrix(
        [
            CandidateQuery(
                user,
                user * 10,
                {
                    user * 10: (1.0,) + (0.0,) * 9,
                    user * 10 + 1: (0.0,) * 10,
                },
            )
            for user in range(1, 4)
        ]
    )
    estimator = make_lgbm_ranker(
        {
            "num_leaves": 3,
            "learning_rate": 0.05,
            "n_estimators": 5,
            "min_child_samples": 1,
        },
        seed=7,
    )

    LearnedRanker(estimator).fit(matrix)

    assert len(estimator.predict(matrix.features)) == len(matrix.labels)

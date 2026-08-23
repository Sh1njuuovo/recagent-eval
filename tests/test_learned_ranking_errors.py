from __future__ import annotations

import hashlib
import json
import math

import pytest

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_SCHEMA_FINGERPRINT_V2,
    FEATURE_SCHEMA_VERSION_V2,
)
from recagent_eval.data import Movie
from recagent_eval.learned_ranking import (
    DEFAULT_PARAMETER_GRID,
    CandidateQuery,
    LearnedRanker,
    RankerArtifact,
    TrainingMatrix,
    artifact_from_estimator,
    build_training_matrix,
    parse_ranker_artifact,
)


def _feature(value: float = 0.0) -> tuple[float, ...]:
    return (value,) + (0.0,) * (len(FEATURE_NAMES) - 1)


def _cv_results(fold_map: dict[int, int]) -> list[dict]:
    cv_results = [
        {"params": params, "mean_ndcg_at_10": 0.0, "mean_recall_at_10": 0.0}
        for params in DEFAULT_PARAMETER_GRID
    ] + [
        {
            "params": params,
            "fold": fold,
            "train_users": sorted(user for user in fold_map if fold_map[user] != fold),
            "validation_users": sorted(user for user in fold_map if fold_map[user] == fold),
            "ndcg_at_10": 0.0,
            "recall_at_10": 0.0,
            "validation_count": 1,
            "ndcg_sum": 0.0,
            "recall_sum": 0.0,
        }
        for params in DEFAULT_PARAMETER_GRID
        for fold in range(3)
    ]
    return cv_results


def _valid_artifact(**overrides) -> RankerArtifact:
    fold_map = {3: 0, 2: 1, 1: 2}
    cv_results = _cv_results(fold_map)
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
        "dependency_versions": {
            "lightgbm": "test",
            "numpy": "test",
            "scikit-learn": "test",
        },
        "model_string": "model contents",
        "model_checksum": hashlib.sha256(b"model contents").hexdigest(),
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
    if "model_string" in overrides and "model_checksum" not in overrides:
        values["model_checksum"] = hashlib.sha256(
            str(values["model_string"]).encode()
        ).hexdigest()
    return RankerArtifact(**values)


class _ContractEstimator:
    def __init__(self, scores, contributions):
        self.scores = scores
        self.contributions = contributions

    def fit(self, features, labels, *, group):
        self.fit_call = (features, labels, group)
        return self

    def predict(self, features, *, pred_contrib=False):
        return self.contributions if pred_contrib else self.scores


def test_training_matrix_limits_negatives_and_rejects_invalid_limit() -> None:
    query = CandidateQuery(1, 10, {10: _feature(1), 20: _feature(), 30: _feature()})

    matrix = build_training_matrix([query], max_negatives=0)

    assert matrix.movie_ids == (10,)
    assert matrix.labels == (1,)
    assert matrix.groups == (1,)
    with pytest.raises(ValueError, match="non-negative"):
        build_training_matrix([query], max_negatives=-1)


def test_learned_ranker_refuses_empty_training_groups_and_invalid_top_k() -> None:
    empty = TrainingMatrix((), (), (), (), (), evaluation_users=0, training_users=0)
    ranker = LearnedRanker(_ContractEstimator([], []))

    with pytest.raises(ValueError, match="no trainable query groups"):
        ranker.fit(empty)
    with pytest.raises(ValueError, match="top_k"):
        ranker.rank_feature_rows({1: Movie(1, "M", ("Drama",))}, {1: _feature()}, top_k=-1)
    assert ranker.rank_feature_rows({}, {}, top_k=10) == []


@pytest.mark.parametrize(
    ("scores", "contributions", "message"),
    [
        ([], [], "prediction shape"),
        ([math.nan], [[0.0] * (len(FEATURE_NAMES) + 1)], "must be finite"),
        ([1.0], [[1.0]], "feature schema"),
        ([1.0], [[0.0] * (len(FEATURE_NAMES) + 1)], "do not reconcile"),
    ],
)
def test_learned_ranker_validates_prediction_and_explanation_contract(
    scores, contributions, message
) -> None:
    ranker = LearnedRanker(_ContractEstimator(scores, contributions))

    with pytest.raises(ValueError, match=message):
        ranker.rank_feature_rows(
            {1: Movie(1, "Movie", ("Drama",))},
            {1: _feature()},
        )


def test_artifact_creation_and_parser_fail_closed_on_missing_or_oversized_data(
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="does not expose a booster"):
        artifact_from_estimator(
            object(),
            selected_params={},
            dataset_fingerprint="data",
            training_user_count=1,
            training_group_count=1,
        )
    with pytest.raises(ValueError, match="missing required fields"):
        parse_ranker_artifact(b"{}")

    monkeypatch.setattr("recagent_eval.learned_ranking.MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        parse_ranker_artifact(b"{}")


def test_v2_artifact_requires_latent_provenance() -> None:
    with pytest.raises(ValueError, match="latent"):
        _valid_artifact(
            schema_version="lambdamart-artifact/v2",
            feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
            feature_names=FEATURE_NAMES_V2,
            feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
        )
    artifact = _valid_artifact(
        schema_version="lambdamart-artifact/v2",
        feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
        feature_names=FEATURE_NAMES_V2,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
        latent_artifact_checksum="a" * 64,
        latent_provenance={
            "training_fingerprint": "b" * 64,
            "rank": 20,
            "iterations": 12,
            "alpha": 40.0,
            "lambda_reg": 0.1,
            "seed": 42,
            "top_k": 500,
            "artifact_path": "artifacts/experiments/run/latent.npz",
        },
    )
    assert artifact.latent_artifact_checksum == "a" * 64


def test_v1_artifact_rejects_latent_fields() -> None:
    with pytest.raises(ValueError, match="latent"):
        _valid_artifact(latent_artifact_checksum="a" * 64)


def _valid_v2_artifact() -> RankerArtifact:
    return _valid_artifact(
        schema_version="lambdamart-artifact/v2",
        feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
        feature_names=FEATURE_NAMES_V2,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
        latent_artifact_checksum="a" * 64,
        latent_provenance={
            "training_fingerprint": "b" * 64,
            "rank": 20,
            "iterations": 12,
            "alpha": 40.0,
            "lambda_reg": 0.1,
            "seed": 42,
            "top_k": 500,
            "artifact_path": "artifacts/experiments/run/latent.npz",
        },
    )


def test_parse_v2_artifact_requires_latent_fields() -> None:
    payload = json.loads(_valid_v2_artifact().model_dump_json())
    payload.pop("latent_provenance")
    with pytest.raises(ValueError, match="missing required fields"):
        parse_ranker_artifact(json.dumps(payload).encode())


def test_parse_v2_latent_checksum_mismatch() -> None:
    raw = _valid_v2_artifact().model_dump_json().encode()
    with pytest.raises(ValueError, match="latent checksum"):
        parse_ranker_artifact(raw, expected_latent_artifact_checksum="b" * 64)


def test_artifact_from_estimator_v2_sets_latent_provenance() -> None:
    class _BoosterStub:
        def model_to_string(self) -> str:
            return "model string"

    class _EstimatorStub:
        booster_ = _BoosterStub()

    fold_map = {3: 0, 2: 1, 1: 2}
    cv_results = _cv_results(fold_map)
    artifact = artifact_from_estimator(
        _EstimatorStub(),
        selected_params={
            "num_leaves": 15,
            "learning_rate": 0.03,
            "n_estimators": 100,
            "min_child_samples": 50,
        },
        dataset_fingerprint="dataset",
        training_user_count=1,
        training_group_count=1,
        provenance={
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
            "fold_map": fold_map,
            "validation_rows_fingerprint": "validation",
            "validation_user_count": 1,
        },
        feature_version="v2",
        cv_results=cv_results,
        latent_artifact_checksum="a" * 64,
        latent_provenance={
            "training_fingerprint": "b" * 64,
            "rank": 20,
            "iterations": 12,
            "alpha": 40.0,
            "lambda_reg": 0.1,
            "seed": 42,
            "top_k": 500,
            "artifact_path": "artifacts/experiments/run/latent.npz",
        },
    )
    assert artifact.schema_version == "lambdamart-artifact/v2"


def test_feature_schema_rejects_unknown_version() -> None:
    from recagent_eval.learned_ranking import _feature_schema

    with pytest.raises(ValueError, match="feature_version"):
        _feature_schema("v9")

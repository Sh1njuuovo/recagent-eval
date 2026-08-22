from __future__ import annotations

import math

import pytest

from recagent_eval.candidate_features import FEATURE_NAMES
from recagent_eval.data import Movie
from recagent_eval.learned_ranking import (
    CandidateQuery,
    LearnedRanker,
    TrainingMatrix,
    artifact_from_estimator,
    build_training_matrix,
    parse_ranker_artifact,
)


def _feature(value: float = 0.0) -> tuple[float, ...]:
    return (value,) + (0.0,) * (len(FEATURE_NAMES) - 1)


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

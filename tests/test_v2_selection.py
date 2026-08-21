from __future__ import annotations

import copy

import pytest

from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.lambdamart_pipeline import build_fold_queries
from recagent_eval.learned_ranking import CandidateQuery
from recagent_eval.retrieval import ItemCFRetriever
from recagent_eval.runner import ExperimentConfig
from recagent_eval.v2_selection import (
    LearnedValidationEvidence,
    build_validation_evidence,
    consume_frozen_authorization,
    cross_validate_lambdamart,
    paired_bootstrap_ndcg,
    validate_learned_gate,
)


class _Estimator:
    def __init__(self, params):
        self.params = params
        self.fit_users = set()

    def fit(self, features, labels, *, group):
        return self

    def predict(self, features, *, pred_contrib=False):
        return [row[0] for row in features]


def _queries() -> list[CandidateQuery]:
    return [CandidateQuery(user, 10, {10: (1.0,) * 10, 20: (0.0,) * 10}) for user in range(1, 7)]


def test_group_cv_is_deterministic_and_ties_choose_simpler_parameters() -> None:
    grid = (
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 50},
        {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 100, "min_child_samples": 20},
    )
    first = cross_validate_lambdamart(
        _queries(), estimator_factory=_Estimator, parameter_grid=grid, seed=9
    )
    second = cross_validate_lambdamart(
        _queries(), estimator_factory=_Estimator, parameter_grid=grid, seed=9
    )

    assert first.fold_by_user == second.fold_by_user
    assert set(first.fold_by_user) == set(range(1, 7))
    assert first.selected_params == grid[1]
    assert all(set(row.train_users).isdisjoint(row.validation_users) for row in first.fold_rows)


def test_cv_aggregate_is_weighted_over_all_held_out_users() -> None:
    queries = [
        CandidateQuery(
            user,
            user * 10,
            {
                user * 10: ((2.0 if user == 1 else 0.0),) + (0.0,) * 9,
                **{
                    user * 100 + offset: (1.0,) + (0.0,) * 9
                    for offset in range(1, 11)
                },
            },
        )
        for user in range(1, 5)
    ]
    result = cross_validate_lambdamart(
        queries,
        estimator_factory=_Estimator,
        parameter_grid=({"num_leaves": 15},),
        seed=1,
    )

    assert result.parameter_rows[0]["mean_ndcg_at_10"] == pytest.approx(0.25)


def test_paired_bootstrap_is_exact_and_deterministic() -> None:
    baseline = [0.0] * 20
    learned = [0.2] * 20
    first = paired_bootstrap_ndcg(baseline, learned, seed=42)
    second = paired_bootstrap_ndcg(baseline, learned, seed=42)

    assert first == second
    assert first.resamples == 2000
    assert first.lower == pytest.approx(0.2)
    assert first.upper == pytest.approx(0.2)


def _positive_evidence() -> LearnedValidationEvidence:
    rows = [
        {
            "user_id": user,
            "itemcf_ndcg_at_10": 0.0,
            "lambdamart_ndcg_at_10": 0.2,
            "itemcf_recall_at_10": 0.0,
            "lambdamart_recall_at_10": 1.0,
            "itemcf_hit_at_10": 0.0,
            "lambdamart_hit_at_10": 1.0,
            "itemcf_candidate_recall": 1.0,
            "dense_candidate_recall": 1.0,
            "union_candidate_recall": 1.0,
            "constraint_satisfied": True,
            "legal_history_movie_ids": [100 + user],
            "allowed_movie_ids": [1, 2],
            "lambdamart_ranked_movie_ids": [1],
            "latency_ms": 1.0,
        }
        for user in range(20)
    ]
    return build_validation_evidence(
        rows,
        dataset_fingerprint="data",
        feature_fingerprint="features",
        model_fingerprint="model",
        candidate_policy_fingerprint="policy",
        seed=42,
    )


def test_gate_recomputes_rows_and_rejects_edited_aggregate_or_constraints() -> None:
    evidence = _positive_evidence()
    validate_learned_gate(
        evidence,
        dataset_fingerprint="data",
        feature_fingerprint="features",
        model_fingerprint="model",
        candidate_policy_fingerprint="policy",
    )

    edited = copy.deepcopy(evidence)
    edited.mean_lambdamart_ndcg_at_10 = 99
    with pytest.raises(ValueError, match="inconsistent"):
        validate_learned_gate(
            edited,
            dataset_fingerprint="data",
            feature_fingerprint="features",
            model_fingerprint="model",
            candidate_policy_fingerprint="policy",
        )

    bad_rows = [dict(row) for row in evidence.per_user_rows]
    bad_rows[0]["lambdamart_ranked_movie_ids"] = [999]
    constrained = build_validation_evidence(
        bad_rows,
        dataset_fingerprint="data",
        feature_fingerprint="features",
        model_fingerprint="model",
        candidate_policy_fingerprint="policy",
        seed=42,
    )
    with pytest.raises(ValueError, match="constraints"):
        validate_learned_gate(
            constrained,
            dataset_fingerprint="data",
            feature_fingerprint="features",
            model_fingerprint="model",
            candidate_policy_fingerprint="policy",
        )


def test_validation_evidence_rejects_duplicate_user_rows() -> None:
    rows = _positive_evidence().per_user_rows
    with pytest.raises(ValueError, match="duplicate user"):
        build_validation_evidence(
            [rows[0], rows[0]],
            dataset_fingerprint="data",
            feature_fingerprint="features",
            model_fingerprint="model",
            candidate_policy_fingerprint="policy",
            seed=42,
        )


def test_validation_evidence_parser_rejects_missing_and_extra_provenance() -> None:
    payload = _positive_evidence().model_dump(mode="json")
    payload.pop("metric_fingerprint")
    with pytest.raises(ValueError, match="metric_fingerprint"):
        LearnedValidationEvidence.model_validate(payload)
    payload["metric_fingerprint"] = "metric"
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="extra"):
        LearnedValidationEvidence.model_validate(payload)


@pytest.mark.parametrize("invalid", ["false", 1, None])
def test_validation_evidence_requires_json_boolean_constraints(invalid) -> None:
    rows = _positive_evidence().per_user_rows
    rows[0]["constraint_satisfied"] = invalid
    with pytest.raises(ValueError, match="JSON boolean"):
        build_validation_evidence(
            rows,
            dataset_fingerprint="data",
            feature_fingerprint="features",
            model_fingerprint="model",
            candidate_policy_fingerprint="policy",
            seed=42,
        )


def test_gate_recomputes_actual_ranked_constraint_violation() -> None:
    rows = _positive_evidence().per_user_rows
    rows[0]["lambdamart_ranked_movie_ids"] = [999]
    rows[0]["constraint_satisfied"] = True
    evidence = build_validation_evidence(
        rows,
        dataset_fingerprint="data",
        feature_fingerprint="features",
        model_fingerprint="model",
        candidate_policy_fingerprint="policy",
        seed=42,
    )
    with pytest.raises(ValueError, match="constraints"):
        validate_learned_gate(
            evidence,
            dataset_fingerprint="data",
            feature_fingerprint="features",
            model_fingerprint="model",
            candidate_policy_fingerprint="policy",
        )


def test_frozen_authorization_is_atomically_one_time_and_bound(tmp_path) -> None:
    marker = tmp_path / "consumed.json"
    consume_frozen_authorization(
        marker, evidence_hash="evidence", case_fingerprint="cases"
    )
    with pytest.raises(ValueError, match="already consumed"):
        consume_frozen_authorization(
            marker, evidence_hash="evidence", case_fingerprint="cases"
        )
    with pytest.raises(ValueError, match="mismatch"):
        consume_frozen_authorization(
            marker, evidence_hash="different", case_fingerprint="cases"
        )


def test_frozen_authorization_fails_closed_on_partial_marker(tmp_path) -> None:
    marker = tmp_path / "consumed.json"
    marker.write_text('{"evidence_hash":"only"}')
    with pytest.raises(ValueError, match="invalid"):
        consume_frozen_authorization(
            marker, evidence_hash="only", case_fingerprint="cases"
        )


def test_fold_candidate_statistics_fit_only_training_users(monkeypatch) -> None:
    movies = {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 2000)
        for movie_id in range(1, 50)
    }
    split = leakage_safe_ranking_split(
        [
            Rating(user, user * 10 + offset, 5, offset)
            for user in range(1, 5)
            for offset in range(4)
        ]
    )
    seen_fit_users = []
    seen_fit_movies = []
    original_fit = ItemCFRetriever.fit

    def capture_fit(rows):
        rows = tuple(rows)
        seen_fit_users.append({row.user_id for row in rows})
        seen_fit_movies.append({row.movie_id for row in rows})
        return original_fit(rows)

    class EmptySemantic:
        def retrieve(self, query, top_k, allowed_ids):
            return []

    monkeypatch.setattr(ItemCFRetriever, "fit", capture_fit)
    build_fold_queries(
        movies,
        split,
        EmptySemantic(),
        ExperimentConfig(name="fold"),
        train_users=(1, 2, 3),
        validation_users=(4,),
    )

    assert seen_fit_users == [{1, 2, 3}, {1, 2, 3}]
    assert all(40 not in movie_ids for movie_ids in seen_fit_movies)

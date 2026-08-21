from __future__ import annotations

import copy

import pytest

from recagent_eval.learned_ranking import CandidateQuery
from recagent_eval.v2_selection import (
    LearnedValidationEvidence,
    build_validation_evidence,
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
    bad_rows[0]["constraint_satisfied"] = False
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

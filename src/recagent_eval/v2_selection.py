from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

from recagent_eval.learned_ranking import (
    DEFAULT_PARAMETER_GRID,
    CandidateQuery,
    RankerEstimator,
    build_training_matrix,
)


@dataclass(frozen=True)
class CVFoldRow:
    params: dict[str, int | float]
    fold: int
    train_users: tuple[int, ...]
    validation_users: tuple[int, ...]
    ndcg_at_10: float
    recall_at_10: float


@dataclass(frozen=True)
class CVSelection:
    selected_params: dict[str, int | float]
    parameter_rows: tuple[dict[str, Any], ...]
    fold_rows: tuple[CVFoldRow, ...]
    fold_by_user: dict[int, int]


def cross_validate_lambdamart(
    queries: Sequence[CandidateQuery],
    *,
    estimator_factory: Callable[[Mapping[str, int | float]], RankerEstimator],
    parameter_grid: Sequence[Mapping[str, int | float]] = DEFAULT_PARAMETER_GRID,
    seed: int = 42,
    n_splits: int = 3,
) -> CVSelection:
    """Select LambdaMART parameters with whole-user GroupKFold splits."""
    del seed  # GroupKFold itself is deterministic and intentionally unshuffled.
    users = sorted({query.user_id for query in queries})
    if len(users) < n_splits:
        raise ValueError(f"GroupKFold requires at least {n_splits} users")
    if not parameter_grid:
        raise ValueError("LambdaMART parameter grid must not be empty")
    try:
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("grouped CV requires the optional 'ml' dependencies") from exc

    user_array = np.asarray(users)
    fold_by_user: dict[int, int] = {}
    folds: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_indexes, validation_indexes) in enumerate(
        splitter.split(user_array, groups=user_array)
    ):
        train_users = tuple(int(user_array[index]) for index in train_indexes)
        validation_users = tuple(int(user_array[index]) for index in validation_indexes)
        folds.append((train_users, validation_users))
        fold_by_user.update({user_id: fold for user_id in validation_users})

    query_by_user = {query.user_id: query for query in queries}
    fold_rows: list[CVFoldRow] = []
    parameter_rows: list[dict[str, Any]] = []
    for raw_params in parameter_grid:
        params = dict(raw_params)
        ndcgs: list[float] = []
        recalls: list[float] = []
        for fold, (train_users, validation_users) in enumerate(folds):
            train_matrix = build_training_matrix(
                [query_by_user[user_id] for user_id in train_users]
            )
            if not train_matrix.groups:
                raise ValueError(f"fold {fold} has no trainable query groups")
            estimator = estimator_factory(params)
            estimator.fit(
                list(train_matrix.features),
                list(train_matrix.labels),
                group=list(train_matrix.groups),
            )
            fold_ndcg: list[float] = []
            fold_recall: list[float] = []
            for user_id in validation_users:
                query = query_by_user[user_id]
                ranked = _rank_query(estimator, query)
                target_rank = (
                    ranked.index(query.target_movie_id) + 1
                    if query.target_movie_id in ranked[:10]
                    else None
                )
                fold_recall.append(float(target_rank is not None))
                fold_ndcg.append(
                    1.0 / math.log2(target_rank + 1) if target_rank is not None else 0.0
                )
            ndcg = _mean(fold_ndcg)
            recall = _mean(fold_recall)
            ndcgs.append(ndcg)
            recalls.append(recall)
            fold_rows.append(
                CVFoldRow(
                    params=params,
                    fold=fold,
                    train_users=train_users,
                    validation_users=validation_users,
                    ndcg_at_10=ndcg,
                    recall_at_10=recall,
                )
            )
        parameter_rows.append(
            {
                "params": params,
                "mean_ndcg_at_10": _mean(ndcgs),
                "mean_recall_at_10": _mean(recalls),
            }
        )
    selected = max(
        parameter_rows,
        key=lambda row: (
            row["mean_ndcg_at_10"],
            row["mean_recall_at_10"],
            tuple(-value for value in _complexity_key(row["params"])),
        ),
    )
    return CVSelection(
        selected_params=dict(selected["params"]),
        parameter_rows=tuple(parameter_rows),
        fold_rows=tuple(fold_rows),
        fold_by_user=fold_by_user,
    )


def _complexity_key(params: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(params.get("num_leaves", 0)),
        float(params.get("n_estimators", 0)),
        float(params.get("learning_rate", 0)),
        -float(params.get("min_child_samples", 0)),
    )


def _rank_query(estimator: RankerEstimator, query: CandidateQuery) -> list[int]:
    movie_ids = sorted(query.features_by_movie)
    if not movie_ids:
        return []
    scores = estimator.predict([query.features_by_movie[item] for item in movie_ids])
    if len(scores) != len(movie_ids):
        raise ValueError("LambdaMART returned an unexpected prediction shape")
    return sorted(
        movie_ids,
        key=lambda movie_id: (-float(scores[movie_ids.index(movie_id)]), movie_id),
    )


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int


def paired_bootstrap_ndcg(
    itemcf_values: Sequence[float],
    lambdamart_values: Sequence[float],
    *,
    seed: int,
) -> BootstrapInterval:
    if len(itemcf_values) != len(lambdamart_values) or not itemcf_values:
        raise ValueError("paired bootstrap requires equal, non-empty per-user arrays")
    baseline = np.asarray(itemcf_values, dtype=float)
    learned = np.asarray(lambdamart_values, dtype=float)
    if not np.isfinite(baseline).all() or not np.isfinite(learned).all():
        raise ValueError("paired bootstrap values must be finite")
    deltas = learned - baseline
    random = np.random.default_rng(seed)
    samples = random.integers(0, len(deltas), size=(2000, len(deltas)))
    bootstrap_means = deltas[samples].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return BootstrapInterval(
        mean_delta=float(deltas.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=0.95,
        resamples=2000,
        seed=seed,
    )


_REQUIRED_ROW_FIELDS = {
    "user_id",
    "itemcf_ndcg_at_10",
    "lambdamart_ndcg_at_10",
    "itemcf_recall_at_10",
    "lambdamart_recall_at_10",
    "itemcf_hit_at_10",
    "lambdamart_hit_at_10",
    "itemcf_candidate_recall",
    "dense_candidate_recall",
    "union_candidate_recall",
    "constraint_satisfied",
    "latency_ms",
}


class LearnedValidationEvidence(BaseModel):
    schema_version: str = "lambdamart-validation/v1"
    per_user_rows: list[dict[str, Any]]
    dataset_fingerprint: str
    feature_fingerprint: str
    model_fingerprint: str
    candidate_policy_fingerprint: str
    seed: int
    bootstrap_resamples: int
    mean_itemcf_ndcg_at_10: float
    mean_lambdamart_ndcg_at_10: float
    mean_ndcg_delta: float
    ndcg_delta_ci_lower: float
    ndcg_delta_ci_upper: float
    constraint_satisfaction_rate: float
    aggregates: dict[str, float]
    evidence_fingerprint: str


def build_validation_evidence(
    per_user_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    model_fingerprint: str,
    candidate_policy_fingerprint: str,
    seed: int,
) -> LearnedValidationEvidence:
    rows = [dict(row) for row in sorted(per_user_rows, key=lambda row: int(row["user_id"]))]
    if not rows:
        raise ValueError("validation evidence requires per-user rows")
    user_ids = [int(row["user_id"]) for row in rows]
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("validation evidence contains a duplicate user row")
    for row in rows:
        missing = sorted(_REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"validation evidence row user={row.get('user_id')} missing cells: {missing}"
            )
        for key in _REQUIRED_ROW_FIELDS - {"user_id", "constraint_satisfied"}:
            if not math.isfinite(float(row[key])):
                raise ValueError(
                    f"validation evidence row user={row['user_id']} has non-finite {key}"
                )
    itemcf = [float(row["itemcf_ndcg_at_10"]) for row in rows]
    learned = [float(row["lambdamart_ndcg_at_10"]) for row in rows]
    interval = paired_bootstrap_ndcg(itemcf, learned, seed=seed)
    aggregates = {
        field: _mean([float(row[field]) for row in rows])
        for field in sorted(_REQUIRED_ROW_FIELDS - {"user_id", "constraint_satisfied"})
    }
    constraints = _mean([float(bool(row["constraint_satisfied"])) for row in rows])
    fingerprint_payload = {
        "rows": rows,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "model_fingerprint": model_fingerprint,
        "candidate_policy_fingerprint": candidate_policy_fingerprint,
        "seed": seed,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return LearnedValidationEvidence(
        per_user_rows=rows,
        dataset_fingerprint=dataset_fingerprint,
        feature_fingerprint=feature_fingerprint,
        model_fingerprint=model_fingerprint,
        candidate_policy_fingerprint=candidate_policy_fingerprint,
        seed=seed,
        bootstrap_resamples=interval.resamples,
        mean_itemcf_ndcg_at_10=_mean(itemcf),
        mean_lambdamart_ndcg_at_10=_mean(learned),
        mean_ndcg_delta=interval.mean_delta,
        ndcg_delta_ci_lower=interval.lower,
        ndcg_delta_ci_upper=interval.upper,
        constraint_satisfaction_rate=constraints,
        aggregates=aggregates,
        evidence_fingerprint=fingerprint,
    )


def validate_learned_gate(
    evidence: LearnedValidationEvidence,
    *,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    model_fingerprint: str,
    candidate_policy_fingerprint: str,
) -> None:
    if evidence.schema_version != "lambdamart-validation/v1":
        raise ValueError("unsupported validation evidence schema")
    try:
        derived = build_validation_evidence(
            evidence.per_user_rows,
            dataset_fingerprint=evidence.dataset_fingerprint,
            feature_fingerprint=evidence.feature_fingerprint,
            model_fingerprint=evidence.model_fingerprint,
            candidate_policy_fingerprint=evidence.candidate_policy_fingerprint,
            seed=evidence.seed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"validation evidence is incomplete: {exc}") from exc
    derived_fields = (
        "bootstrap_resamples",
        "mean_itemcf_ndcg_at_10",
        "mean_lambdamart_ndcg_at_10",
        "mean_ndcg_delta",
        "ndcg_delta_ci_lower",
        "ndcg_delta_ci_upper",
        "constraint_satisfaction_rate",
        "aggregates",
        "evidence_fingerprint",
    )
    if any(not _same(getattr(evidence, name), getattr(derived, name)) for name in derived_fields):
        raise ValueError("validation evidence aggregate fields are inconsistent with per-user rows")
    comparisons = {
        "dataset_fingerprint": (evidence.dataset_fingerprint, dataset_fingerprint),
        "feature_fingerprint": (evidence.feature_fingerprint, feature_fingerprint),
        "model_fingerprint": (evidence.model_fingerprint, model_fingerprint),
        "candidate_policy_fingerprint": (
            evidence.candidate_policy_fingerprint,
            candidate_policy_fingerprint,
        ),
    }
    mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError("validation evidence fingerprint mismatch: " + ", ".join(mismatches))
    if evidence.mean_lambdamart_ndcg_at_10 <= evidence.mean_itemcf_ndcg_at_10:
        raise ValueError("frozen test is locked: LambdaMART did not improve mean NDCG@10")
    if evidence.ndcg_delta_ci_lower <= 0:
        raise ValueError("frozen test is locked: paired bootstrap confidence interval crosses zero")
    if evidence.constraint_satisfaction_rate != 1.0:
        raise ValueError("frozen test is locked: constraints were not satisfied for every user")


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

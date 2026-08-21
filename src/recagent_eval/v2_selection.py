from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

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
    fold_query_builder: Callable[
        [tuple[int, ...], tuple[int, ...]],
        tuple[Sequence[CandidateQuery], Sequence[CandidateQuery]],
    ]
    | None = None,
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
    if len(query_by_user) != len(queries):
        raise ValueError("grouped CV requires exactly one query per user")
    fold_queries = [
        fold_query_builder(train_users, validation_users)
        if fold_query_builder is not None
        else (
            [query_by_user[user_id] for user_id in train_users],
            [query_by_user[user_id] for user_id in validation_users],
        )
        for train_users, validation_users in folds
    ]
    fold_rows: list[CVFoldRow] = []
    parameter_rows: list[dict[str, Any]] = []
    for raw_params in parameter_grid:
        params = dict(raw_params)
        held_out_ndcgs: list[float] = []
        held_out_recalls: list[float] = []
        for fold, (train_users, validation_users) in enumerate(folds):
            train_queries, validation_queries = fold_queries[fold]
            if {query.user_id for query in train_queries} & {
                query.user_id for query in validation_queries
            }:
                raise ValueError("fold query builder mixed training and validation users")
            train_matrix = build_training_matrix(train_queries)
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
            for query in validation_queries:
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
            held_out_ndcgs.extend(fold_ndcg)
            held_out_recalls.extend(fold_recall)
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
                "mean_ndcg_at_10": _mean(held_out_ndcgs),
                "mean_recall_at_10": _mean(held_out_recalls),
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
    "legal_history_movie_ids",
    "allowed_movie_ids",
    "lambdamart_ranked_movie_ids",
}
_LIST_ROW_FIELDS = {
    "legal_history_movie_ids",
    "allowed_movie_ids",
    "lambdamart_ranked_movie_ids",
}


class LearnedValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    training_rows_fingerprint: str
    history_fingerprint: str
    fold_map_fingerprint: str
    group_fingerprint: str
    config_fingerprint: str
    metric_fingerprint: str
    case_fingerprint: str
    report_fingerprint: str
    selected_params: dict[str, int | float]
    cv_results: list[dict[str, Any]]
    training_user_count: int
    training_group_count: int
    dependency_versions: dict[str, str]
    fold_map: dict[int, int]


def build_validation_evidence(
    per_user_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    model_fingerprint: str,
    candidate_policy_fingerprint: str,
    seed: int,
    provenance: Mapping[str, Any] | None = None,
) -> LearnedValidationEvidence:
    provenance_defaults: dict[str, Any] = {
        "training_rows_fingerprint": "unspecified",
        "history_fingerprint": "unspecified",
        "fold_map_fingerprint": "unspecified",
        "group_fingerprint": "unspecified",
        "config_fingerprint": "unspecified",
        "metric_fingerprint": "unspecified",
        "case_fingerprint": "unspecified",
        "selected_params": {},
        "cv_results": [],
        "training_user_count": 0,
        "training_group_count": 0,
        "dependency_versions": {},
        "fold_map": {},
    }
    supplied_provenance = dict(provenance or {})
    provenance = {
        key: supplied_provenance.get(key, default)
        for key, default in provenance_defaults.items()
    }
    if "report_fingerprint" in supplied_provenance:
        provenance["report_fingerprint"] = supplied_provenance[
            "report_fingerprint"
        ]
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
        if type(row["constraint_satisfied"]) is not bool:
            raise ValueError(
                f"validation evidence row user={row['user_id']} "
                "constraint_satisfied must be a JSON boolean"
            )
        for key in _LIST_ROW_FIELDS:
            value = row[key]
            if not isinstance(value, list) or any(type(item) is not int for item in value):
                raise ValueError(
                    f"validation evidence row user={row['user_id']} {key} "
                    "must be a JSON integer array"
                )
        allowed = set(row["allowed_movie_ids"])
        history = set(row["legal_history_movie_ids"])
        ranked = set(row["lambdamart_ranked_movie_ids"])
        row["constraint_satisfied"] = ranked <= allowed and ranked.isdisjoint(history)
        for key in _REQUIRED_ROW_FIELDS - {
            "user_id",
            "constraint_satisfied",
            *_LIST_ROW_FIELDS,
        }:
            if not math.isfinite(float(row[key])):
                raise ValueError(
                    f"validation evidence row user={row['user_id']} has non-finite {key}"
                )
    itemcf = [float(row["itemcf_ndcg_at_10"]) for row in rows]
    learned = [float(row["lambdamart_ndcg_at_10"]) for row in rows]
    interval = paired_bootstrap_ndcg(itemcf, learned, seed=seed)
    aggregates = {
        field: _mean([float(row[field]) for row in rows])
        for field in sorted(
            _REQUIRED_ROW_FIELDS
            - {"user_id", "constraint_satisfied"}
            - _LIST_ROW_FIELDS
        )
    }
    constraints = _mean([float(bool(row["constraint_satisfied"])) for row in rows])
    fingerprint_payload = {
        "rows": rows,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "model_fingerprint": model_fingerprint,
        "candidate_policy_fingerprint": candidate_policy_fingerprint,
        "seed": seed,
        "provenance": {
            key: value for key, value in provenance.items() if key != "report_fingerprint"
        },
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
        training_rows_fingerprint=str(
            provenance.get("training_rows_fingerprint", "unspecified")
        ),
        history_fingerprint=str(provenance.get("history_fingerprint", "unspecified")),
        fold_map_fingerprint=str(
            provenance.get("fold_map_fingerprint", "unspecified")
        ),
        group_fingerprint=str(provenance.get("group_fingerprint", "unspecified")),
        config_fingerprint=str(provenance.get("config_fingerprint", "unspecified")),
        metric_fingerprint=str(provenance.get("metric_fingerprint", "unspecified")),
        case_fingerprint=str(provenance.get("case_fingerprint", "unspecified")),
        report_fingerprint=str(provenance.get("report_fingerprint", fingerprint)),
        selected_params=dict(provenance.get("selected_params", {})),
        cv_results=[dict(row) for row in provenance.get("cv_results", [])],
        training_user_count=int(provenance.get("training_user_count", 0)),
        training_group_count=int(provenance.get("training_group_count", 0)),
        dependency_versions=dict(provenance.get("dependency_versions", {})),
        fold_map={
            int(user): int(fold)
            for user, fold in dict(provenance.get("fold_map", {})).items()
        },
    )


def validate_learned_gate(
    evidence: LearnedValidationEvidence,
    *,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    model_fingerprint: str,
    candidate_policy_fingerprint: str,
    case_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
    artifact_provenance: Mapping[str, Any] | None = None,
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
            provenance={
                "training_rows_fingerprint": evidence.training_rows_fingerprint,
                "history_fingerprint": evidence.history_fingerprint,
                "fold_map_fingerprint": evidence.fold_map_fingerprint,
                "group_fingerprint": evidence.group_fingerprint,
                "config_fingerprint": evidence.config_fingerprint,
                "metric_fingerprint": evidence.metric_fingerprint,
                "case_fingerprint": evidence.case_fingerprint,
                "report_fingerprint": evidence.report_fingerprint,
                "selected_params": evidence.selected_params,
                "cv_results": evidence.cv_results,
                "training_user_count": evidence.training_user_count,
                "training_group_count": evidence.training_group_count,
                "dependency_versions": evidence.dependency_versions,
                "fold_map": evidence.fold_map,
            },
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
    if evidence.cv_results:
        aggregates = [row for row in evidence.cv_results if "mean_ndcg_at_10" in row]
        folds = [row for row in evidence.cv_results if "fold" in row]
        if len(aggregates) != 16:
            raise ValueError("validation evidence must contain all 16 CV aggregate results")
        if len(folds) != 48 or {int(row["fold"]) for row in folds} != {0, 1, 2}:
            raise ValueError("validation evidence must contain all 16 CV results per fold")
        expected_params = {
            json.dumps(params, sort_keys=True) for params in DEFAULT_PARAMETER_GRID
        }
        aggregate_params = {
            json.dumps(row["params"], sort_keys=True) for row in aggregates
        }
        fold_cells = {
            (json.dumps(row["params"], sort_keys=True), int(row["fold"]))
            for row in folds
        }
        if aggregate_params != expected_params or fold_cells != {
            (params, fold) for params in expected_params for fold in range(3)
        }:
            raise ValueError("validation evidence CV grid is incomplete or duplicated")
        calculated_report = hashlib.sha256(
            json.dumps(
                evidence.cv_results, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if calculated_report != evidence.report_fingerprint:
            raise ValueError("validation evidence report fingerprint is inconsistent")
        calculated_fold_map = hashlib.sha256(
            json.dumps(
                evidence.fold_map, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if calculated_fold_map != evidence.fold_map_fingerprint:
            raise ValueError("validation evidence fold-map fingerprint is inconsistent")
        selected = max(
            aggregates,
            key=lambda row: (
                float(row["mean_ndcg_at_10"]),
                float(row["mean_recall_at_10"]),
                tuple(-value for value in _complexity_key(row["params"])),
            ),
        )["params"]
        if selected != evidence.selected_params:
            raise ValueError("validation evidence selected parameters are inconsistent")
    comparisons = {
        "dataset_fingerprint": (evidence.dataset_fingerprint, dataset_fingerprint),
        "feature_fingerprint": (evidence.feature_fingerprint, feature_fingerprint),
        "model_fingerprint": (evidence.model_fingerprint, model_fingerprint),
        "candidate_policy_fingerprint": (
            evidence.candidate_policy_fingerprint,
            candidate_policy_fingerprint,
        ),
    }
    if case_fingerprint is not None:
        comparisons["case_fingerprint"] = (
            evidence.case_fingerprint,
            case_fingerprint,
        )
    if config_fingerprint is not None:
        comparisons["config_fingerprint"] = (
            evidence.config_fingerprint,
            config_fingerprint,
        )
    mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError("validation evidence fingerprint mismatch: " + ", ".join(mismatches))
    expected_metric_fingerprint = hashlib.sha256(
        json.dumps(
            {"bootstrap_resamples": 2000, "k": 10, "metric": "ndcg"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if evidence.metric_fingerprint not in {
        "unspecified",
        expected_metric_fingerprint,
    }:
        raise ValueError("validation evidence metric fingerprint mismatch")
    if artifact_provenance is not None:
        shared_fields = (
            "training_rows_fingerprint",
            "history_fingerprint",
            "fold_map_fingerprint",
            "group_fingerprint",
            "candidate_policy_fingerprint",
            "config_fingerprint",
            "metric_fingerprint",
            "case_fingerprint",
            "report_fingerprint",
            "selected_params",
            "cv_results",
            "training_user_count",
            "training_group_count",
            "dependency_versions",
            "fold_map",
        )
        differing = [
            name
            for name in shared_fields
            if getattr(evidence, name) != artifact_provenance.get(name)
        ]
        if differing:
            raise ValueError(
                "validation evidence/artifact provenance mismatch: "
                + ", ".join(differing)
            )
    if case_fingerprint is not None or config_fingerprint is not None:
        provenance_values = (
            evidence.training_rows_fingerprint,
            evidence.history_fingerprint,
            evidence.fold_map_fingerprint,
            evidence.group_fingerprint,
            evidence.config_fingerprint,
            evidence.metric_fingerprint,
            evidence.case_fingerprint,
            evidence.report_fingerprint,
        )
        if any(value in {"", "unspecified"} for value in provenance_values):
            raise ValueError("validation evidence provenance is incomplete")
        if not evidence.dependency_versions or not evidence.cv_results:
            raise ValueError("validation evidence provenance is incomplete")
        if evidence.training_user_count <= 0 or evidence.training_group_count <= 0:
            raise ValueError("validation evidence provenance counts are incomplete")
    if evidence.mean_lambdamart_ndcg_at_10 <= evidence.mean_itemcf_ndcg_at_10:
        raise ValueError("frozen test is locked: LambdaMART did not improve mean NDCG@10")
    if evidence.ndcg_delta_ci_lower <= 0:
        raise ValueError("frozen test is locked: paired bootstrap confidence interval crosses zero")
    if evidence.constraint_satisfaction_rate != 1.0:
        raise ValueError("frozen test is locked: constraints were not satisfied for every user")


def consume_frozen_authorization(
    marker_path: Path,
    *,
    evidence_hash: str,
    case_fingerprint: str,
) -> None:
    """Atomically consume one LambdaMART frozen-test authorization."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "lambdamart-frozen-consumption/v1",
        "evidence_hash": evidence_hash,
        "case_fingerprint": case_fingerprint,
        "consumed_at": datetime.now(UTC).isoformat(),
    }
    try:
        descriptor = os.open(
            marker_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        try:
            if marker_path.stat().st_size > 4096:
                raise ValueError("frozen authorization marker is invalid")
            existing = json.loads(marker_path.read_text())
        except (OSError, ValueError) as read_exc:
            raise ValueError("frozen authorization marker is invalid") from read_exc
        required = set(payload)
        if (
            set(existing) != required
            or existing.get("schema_version")
            != "lambdamart-frozen-consumption/v1"
            or any(type(existing.get(key)) is not str for key in required)
        ):
            raise ValueError("frozen authorization marker is invalid") from exc
        if (
            existing["evidence_hash"] != evidence_hash
            or existing["case_fingerprint"] != case_fingerprint
        ):
            raise ValueError("frozen authorization marker binding mismatch") from exc
        raise ValueError("frozen authorization was already consumed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

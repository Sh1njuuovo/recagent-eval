from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from recagent_eval.evidence import (
    BASELINE_SCHEMA_V2,
    artifact_fingerprint,
    provenance_value,
)

BASELINE_SCORERS: dict[str, Callable[..., Any]] = {}


def register_baseline(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in BASELINE_SCORERS:
            raise ValueError(f"baseline scorer already registered: {name}")
        BASELINE_SCORERS[name] = func
        return func

    return decorator


@dataclass(frozen=True)
class MetricRow:
    user_id: int
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    candidate_recall: float
    constraint_satisfied: bool
    latency_ms: float
    recommended_ids: tuple[int, ...]


def score_ranking(
    *,
    user_id: int = 0,
    ranked_ids: Sequence[int],
    target: int,
    allowed: set[int],
    history: set[int],
    candidate_recall: float,
    latency_ms: float,
) -> MetricRow:
    top = list(ranked_ids[:10])
    target_index = top.index(target) if target in top else None
    recall = float(target_index is not None)
    ndcg = 1.0 / math.log2(target_index + 2) if target_index is not None else 0.0
    mrr = 1.0 / (target_index + 1) if target_index is not None else 0.0
    ranked_set = set(top)
    constraints = ranked_set.issubset(allowed) and ranked_set.isdisjoint(history)
    return MetricRow(
        user_id=user_id,
        recall_at_10=recall,
        ndcg_at_10=ndcg,
        mrr_at_10=mrr,
        candidate_recall=float(candidate_recall),
        constraint_satisfied=bool(constraints),
        latency_ms=float(latency_ms),
        recommended_ids=tuple(top),
    )


def paired_bootstrap_deltas(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("bootstrap values must be finite")
    deltas = c - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    means = deltas[samples].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "resamples": resamples,
        "seed": seed,
    }


def metric_json(
    rows: Sequence[MetricRow],
    *,
    method: str,
    cohort: str,
    universe_size: int,
    config_fingerprint: str,
    dataset_fingerprint: str,
    model_fingerprint: str,
    cohort_ledger_fingerprint: str,
    selected_params: object,
    parameter_grid: object,
    seed: int | str,
    dependency_versions: Mapping[str, str],
    hardware: Mapping[str, object],
    training_seconds: float,
    resource_usage: Mapping[str, object],
    model_size_bytes: int,
    environment: Mapping[str, str],
    bootstrap: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("baseline evidence requires non-empty rows")
    artifact: dict[str, object] = {
        "schema_version": BASELINE_SCHEMA_V2,
        "method": method,
        "cohort": cohort,
        "cohort_ledger_fingerprint": cohort_ledger_fingerprint,
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "model_fingerprint": model_fingerprint,
        "user_count": len(rows),
        "ordered_user_ids": [row.user_id for row in rows],
        "per_user_rows": [
            {
                "user_id": row.user_id,
                "recall_at_10": row.recall_at_10,
                "ndcg_at_10": row.ndcg_at_10,
                "mrr_at_10": row.mrr_at_10,
                "candidate_recall": row.candidate_recall,
                "constraint_satisfied": row.constraint_satisfied,
                "latency_ms": row.latency_ms,
            }
            for row in rows
        ],
        "aggregates": {
            "recall_at_10": _mean([r.recall_at_10 for r in rows]),
            "ndcg_at_10": _mean([r.ndcg_at_10 for r in rows]),
            "mrr_at_10": _mean([r.mrr_at_10 for r in rows]),
            "candidate_recall": _mean([r.candidate_recall for r in rows]),
            "constraint_satisfaction_rate": _mean(
                [float(r.constraint_satisfied) for r in rows]
            ),
            "coverage": _coverage(rows, universe_size=universe_size),
            "latency_ms_p50": _quantile([r.latency_ms for r in rows], 0.5),
            "latency_ms_p95": _quantile([r.latency_ms for r in rows], 0.95),
        },
        "training_seconds": training_seconds,
        "resource_usage": dict(resource_usage),
        "model_size_bytes": model_size_bytes,
        "selected_params": provenance_value(selected_params, source="observed"),
        "parameter_grid": provenance_value(parameter_grid, source="observed"),
        "seed": provenance_value(seed, source="observed"),
        "dependency_versions": provenance_value(
            dict(dependency_versions), source="observed"
        ),
        "hardware": provenance_value(dict(hardware), source="observed"),
        "legacy_environment": provenance_value(dict(environment), source="observed"),
        "bootstrap_vs_itemcf": dict(bootstrap or {}),
    }
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    return artifact


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coverage(rows: Sequence[MetricRow], *, universe_size: int) -> float:
    recommended = {movie_id for row in rows for movie_id in row.recommended_ids}
    return len(recommended) / universe_size if universe_size else 0.0


def _quantile(values: Sequence[float], position: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(position * len(ordered)))
    return float(ordered[index])

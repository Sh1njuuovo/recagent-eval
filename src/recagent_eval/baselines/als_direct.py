from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import time
from collections.abc import Mapping, Sequence

import numpy as np

from recagent_eval.baseline_eval import MetricRow, register_baseline, score_ranking
from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    _state_from_history,
    ranking_dataset_fingerprint,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.retrieval import hard_filter

ALS_PARAMETER_GRID: tuple[Mapping[str, int | float], ...] = tuple(
    {
        "rank": rank,
        "iterations": iterations,
        "alpha": alpha,
        "lambda_reg": lambda_reg,
    }
    for rank in (20, 40)
    for iterations in (10, 12)
    for alpha in (20.0, 40.0)
    for lambda_reg in (0.05, 0.1)
)


def dev_legal_rows(
    split: LeakageSafeRankingSplit, dev_users: Sequence[int]
) -> tuple:
    dev_set = set(dev_users)
    return tuple(row for row in split.legal_retrieval_train if row.user_id in dev_set)


def select_als_params(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    dev_users: Sequence[int],
) -> dict[str, object]:
    """Select ALS hyperparameters on development users only (user-grouped)."""
    histories = _positive_histories(split.legal_retrieval_train, movies)
    dev_rows = dev_legal_rows(split, dev_users)
    results: list[tuple[float, Mapping[str, int | float]]] = []
    for params in ALS_PARAMETER_GRID:
        latent = LatentFactorRetriever.fit(
            dev_rows,
            rank=int(params["rank"]),
            iterations=int(params["iterations"]),
            alpha=float(params["alpha"]),
            lambda_reg=float(params["lambda_reg"]),
            seed=42,
        )
        ndcgs: list[float] = []
        for user_id in dev_users:
            history_ids = {row.movie_id for row in histories.get(user_id, ())}
            state = _state_from_history(history_ids, movies)
            allowed = {
                movie.movie_id for movie in hard_filter(movies.values(), state)
            } - history_ids
            target = split.validation_targets[user_id]
            if target not in allowed or not history_ids:
                continue
            ranked = [
                movie_id
                for movie_id, _score in latent.retrieve(
                    history_ids, top_k=len(allowed), allowed_ids=allowed
                )
            ][:10]
            if target in ranked:
                ndcgs.append(1.0 / math.log2(ranked.index(target) + 2))
        mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        results.append((mean_ndcg, params))
    best_mean, best_params = max(
        results,
        key=lambda item: (
            item[0],
            -int(item[1]["rank"]),
            -int(item[1]["iterations"]),
            -float(item[1]["alpha"]),
            float(item[1]["lambda_reg"]),
        ),
    )
    payload = {
        "selected_params": dict(best_params),
        "mean_ndcg_at_10": best_mean,
        "dev_user_fingerprint": _fingerprint(sorted(dev_users)),
        "seed": 42,
    }
    return {
        **payload,
        "fingerprint": _fingerprint(payload),
        "grid": [dict(params) for params in ALS_PARAMETER_GRID],
    }


@register_baseline("als_direct")
def score_als_direct(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    users: Sequence[int],
    *,
    ledger: Mapping[str, object] | None = None,
    max_training_users: int | None = None,
) -> dict[str, object]:
    del max_training_users
    dev_users = (
        [int(user) for user in ledger["cohorts"]["development"]]
        if ledger is not None
        else sorted(split.validation_targets)[:10]
    )
    selection = select_als_params(movies, split, dev_users)
    params = selection["selected_params"]
    started = time.perf_counter()
    latent = LatentFactorRetriever.fit(
        split.legal_retrieval_train,
        rank=int(params["rank"]),
        iterations=int(params["iterations"]),
        alpha=float(params["alpha"]),
        lambda_reg=float(params["lambda_reg"]),
        seed=42,
    )
    training_seconds = time.perf_counter() - started
    histories = _positive_histories(split.legal_retrieval_train, movies)
    rows: list[MetricRow] = []
    for user_id in users:
        history_ids = {row.movie_id for row in histories.get(user_id, ())}
        state = _state_from_history(history_ids, movies)
        allowed = {
            movie.movie_id for movie in hard_filter(movies.values(), state)
        } - history_ids
        target = split.validation_targets[user_id]
        t0 = time.perf_counter()
        if not history_ids:
            ranked: list[int] = []
        else:
            ranked = [
                movie_id
                for movie_id, _score in latent.retrieve(
                    history_ids, top_k=len(allowed), allowed_ids=allowed
                )
            ][:10]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            score_ranking(
                user_id=user_id,
                ranked_ids=ranked,
                target=target,
                allowed=allowed,
                history=history_ids,
                candidate_recall=1.0 if target in allowed else 0.0,
                latency_ms=latency_ms,
            )
        )
    return {
        "rows": rows,
        "config_fingerprint": selection["fingerprint"],
        "dataset_fingerprint": ranking_dataset_fingerprint(movies, split),
        "model_fingerprint": latent.training_fingerprint,
        "training_seconds": training_seconds,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "model_size_bytes": len(latent.item_factors.tobytes()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

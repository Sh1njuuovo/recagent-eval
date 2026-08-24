from __future__ import annotations

import hashlib
import json
import platform
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
from recagent_eval.resource_usage import read_process_peak_rss
from recagent_eval.retrieval import ItemCFRetriever, hard_filter


@register_baseline("itemcf_direct")
def score_itemcf_direct(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    users: Sequence[int],
    *,
    ledger: Mapping[str, object] | None = None,
    max_training_users: int | None = None,
) -> dict[str, object]:
    del ledger
    del max_training_users
    started = time.perf_counter()
    itemcf = ItemCFRetriever.fit(split.legal_retrieval_train)
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
        scores = itemcf.score_many(history_ids, allowed)
        ranked = sorted(allowed, key=lambda movie_id: (-scores[movie_id], movie_id))[:10]
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
    model_bytes = json.dumps(
        itemcf.similarities, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "rows": rows,
        "config_fingerprint": hashlib.sha256(b"itemcf_direct/v1").hexdigest(),
        "dataset_fingerprint": ranking_dataset_fingerprint(movies, split),
        "model_fingerprint": hashlib.sha256(model_bytes).hexdigest(),
        "selected_params": {},
        "parameter_grid": [],
        "seed": "not_applicable",
        "training_seconds": training_seconds,
        "resource_usage": read_process_peak_rss(),
        "model_size_bytes": len(model_bytes),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

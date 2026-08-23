from __future__ import annotations

import json
import platform
import resource
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from recagent_eval.baseline_eval import MetricRow, register_baseline
from recagent_eval.config import load_experiment_config
from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    lambdamart_config_fingerprint,
    ranking_dataset_fingerprint,
    train_lambdamart_pipeline,
)
from recagent_eval.retrieval import DenseSemanticRetriever

_FIXED_CASE_FINGERPRINT = "bc2f622cd9311bca8509a46f0ee516355bc64db7d91f809273a35d97ce304d88"


@register_baseline("current_v2b")
def score_current_v2b(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    users: Sequence[int],
    *,
    ledger: Mapping[str, object] | None = None,
    max_training_users: int | None = None,
) -> dict[str, object]:
    eligible = sorted(split.validation_targets)
    historical = tuple(eligible[:500])
    dev = (
        tuple(sorted(int(user) for user in ledger["cohorts"]["development"]))
        if ledger is not None
        else ()
    )
    training_users = tuple(sorted(set(historical) | set(dev)))
    if max_training_users is not None:
        training_users = training_users[:max_training_users]
    config = load_experiment_config(Path("configs/v2_dense_latent_bfeat.yaml"))
    workdir = Path(tempfile.mkdtemp(prefix="v2b-baseline-", dir="/private/tmp"))
    config = replace(config, latent_artifact_path=str(workdir / "latent.npz"))
    model_path = workdir / "model.json"
    evidence_path = workdir / "validation.json"
    bundle_path = workdir / "bundle.json"
    semantic = DenseSemanticRetriever.load(
        Path(config.semantic_cache_path),
        movies=movies,
        model_name=config.semantic_model_name,
        model_revision=config.semantic_model_revision,
        device=config.semantic_device,
    )
    started = time.perf_counter()
    summary = train_lambdamart_pipeline(
        movies,
        split,
        semantic,
        config,
        model_output=model_path,
        evidence_output=evidence_path,
        bundle_manifest_output=bundle_path,
        max_users=max(len(training_users), len(users)),
        seed=config.seed,
        registered_case_fingerprint=_FIXED_CASE_FINGERPRINT,
        training_user_ids=training_users,
        eval_user_ids=tuple(users),
    )
    training_seconds = time.perf_counter() - started
    evidence = json.loads(evidence_path.read_text())
    rows: list[MetricRow] = []
    for row in evidence["per_user_rows"]:
        user_id = int(row["user_id"])
        target = split.validation_targets[user_id]
        ranked = [int(movie_id) for movie_id in row["lambdamart_ranked_movie_ids"]]
        mrr = 1.0 / (ranked.index(target) + 1) if target in ranked[:10] else 0.0
        rows.append(
            MetricRow(
                user_id=user_id,
                recall_at_10=float(row["lambdamart_recall_at_10"]),
                ndcg_at_10=float(row["lambdamart_ndcg_at_10"]),
                mrr_at_10=mrr,
                candidate_recall=float(row["union_candidate_recall"]),
                constraint_satisfied=bool(row["constraint_satisfied"]),
                latency_ms=float(row["latency_ms"]),
                recommended_ids=tuple(ranked),
            )
        )
    model_size_bytes = model_path.stat().st_size + (workdir / "latent.npz").stat().st_size
    return {
        "rows": rows,
        "config_fingerprint": lambdamart_config_fingerprint(config),
        "dataset_fingerprint": ranking_dataset_fingerprint(movies, split),
        "model_fingerprint": str(summary["model_checksum"]),
        "training_seconds": training_seconds,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "model_size_bytes": int(model_size_bytes),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

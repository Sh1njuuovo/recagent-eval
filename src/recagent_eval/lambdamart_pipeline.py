from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recagent_eval.agent import build_semantic_profile
from recagent_eval.candidate_features import (
    FEATURE_SCHEMA_FINGERPRINT,
    build_candidate_feature_rows,
)
from recagent_eval.data import LeakageSafeRankingSplit, Movie, Rating
from recagent_eval.learned_ranking import (
    CandidateQuery,
    LearnedRanker,
    artifact_from_estimator,
    build_training_matrix,
    make_lgbm_ranker,
    save_ranker_artifact,
)
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import ItemCFRetriever, SemanticRetriever
from recagent_eval.runner import ExperimentConfig
from recagent_eval.v2_selection import (
    build_validation_evidence,
    cross_validate_lambdamart,
)


def train_lambdamart_pipeline(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    config: ExperimentConfig,
    *,
    model_output: Path,
    evidence_output: Path,
    max_users: int,
    seed: int,
) -> dict[str, Any]:
    dataset_fingerprint = ranking_dataset_fingerprint(movies, split)
    training_queries = build_candidate_queries(
        movies,
        split.ranker_training_history,
        split.histories,
        split.ranker_targets,
        semantic,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        max_users=max_users,
    )
    cv = cross_validate_lambdamart(
        training_queries,
        estimator_factory=lambda params: make_lgbm_ranker(params, seed=seed),
        seed=seed,
    )
    matrix = build_training_matrix(training_queries)
    estimator = make_lgbm_ranker(cv.selected_params, seed=seed)
    estimator.fit(
        list(matrix.features),
        list(matrix.labels),
        group=list(matrix.groups),
    )
    artifact = artifact_from_estimator(
        estimator,
        selected_params=cv.selected_params,
        dataset_fingerprint=dataset_fingerprint,
        training_user_count=matrix.training_users,
        training_group_count=len(matrix.groups),
    )
    save_ranker_artifact(artifact, model_output)

    validation_histories = _positive_histories(split.legal_retrieval_train, movies)
    validation_queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        validation_histories,
        split.validation_targets,
        semantic,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        max_users=max_users,
    )
    learned = LearnedRanker(estimator, legal_train_rows=split.legal_retrieval_train)
    baseline = HybridRanker(kind="itemcf")
    rows: list[dict[str, Any]] = []
    for query in validation_queries:
        started = time.perf_counter()
        feature_rows = query.features_by_movie
        learned_ids = [
            item.movie_id for item in learned.rank_feature_rows(movies, feature_rows, top_k=10)
        ]
        # ItemCF uses the exact same union candidate policy; non-ItemCF route
        # members receive the existing baseline's zero score and ID tie-break.
        itemcf_scores = {movie_id: values[0] for movie_id, values in feature_rows.items()}
        baseline_ids = [
            item.movie_id
            for item in baseline.rank(
                {movie_id: movies[movie_id] for movie_id in feature_rows},
                itemcf_scores=itemcf_scores,
                semantic_scores={},
                state=PreferenceState(),
                top_k=10,
            )
        ]
        target = query.target_movie_id
        rows.append(
            {
                "user_id": query.user_id,
                "itemcf_ndcg_at_10": _single_ndcg(baseline_ids, target),
                "lambdamart_ndcg_at_10": _single_ndcg(learned_ids, target),
                "itemcf_recall_at_10": float(target in baseline_ids),
                "lambdamart_recall_at_10": float(target in learned_ids),
                "itemcf_hit_at_10": float(target in baseline_ids),
                "lambdamart_hit_at_10": float(target in learned_ids),
                "itemcf_candidate_recall": float(
                    target in feature_rows and feature_rows[target][8] == 1.0
                ),
                "dense_candidate_recall": float(
                    target in feature_rows and feature_rows[target][9] == 1.0
                ),
                "union_candidate_recall": float(target in feature_rows),
                "constraint_satisfied": True,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    policy_fingerprint = candidate_policy_fingerprint(config)
    evidence = build_validation_evidence(
        rows,
        dataset_fingerprint=dataset_fingerprint,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT,
        model_fingerprint=artifact.model_checksum,
        candidate_policy_fingerprint=policy_fingerprint,
        seed=seed,
    )
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_params": cv.selected_params,
        "training_users": matrix.training_users,
        "evaluation_users": matrix.evaluation_users,
        "validation_users": len(rows),
        "model_checksum": artifact.model_checksum,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
        "candidate_policy_fingerprint": policy_fingerprint,
    }


def build_candidate_queries(
    movies: dict[int, Movie],
    legal_train_rows: tuple[Rating, ...],
    histories: Mapping[int, tuple[Rating, ...]],
    targets: Mapping[int, int],
    semantic: SemanticRetriever,
    *,
    retrieval_top_k: int,
    history_cap: int,
    max_users: int,
) -> list[CandidateQuery]:
    itemcf = ItemCFRetriever.fit(legal_train_rows)
    queries: list[CandidateQuery] = []
    for user_id, target in sorted(targets.items())[:max_users]:
        history_rows = histories.get(user_id, ())
        history_ids = {
            row.movie_id for row in history_rows if row.rating >= 4 and row.movie_id in movies
        }
        state = _state_from_history(history_ids, movies)
        allowed_ids = set(movies) - history_ids
        itemcf_scores = dict(
            itemcf.retrieve(history_ids, top_k=retrieval_top_k, allowed_ids=allowed_ids)
        )
        dense_scores = dict(
            semantic.retrieve(
                build_semantic_profile("", state, movies, history_cap=history_cap),
                top_k=retrieval_top_k,
                allowed_ids=allowed_ids,
            )
        )
        rows = build_candidate_feature_rows(
            user_id=user_id,
            movies=movies,
            itemcf_scores=itemcf_scores,
            dense_scores=dense_scores,
            history=history_rows,
            train_rows=legal_train_rows,
            state=state,
        )
        queries.append(
            CandidateQuery(
                user_id=user_id,
                target_movie_id=target,
                features_by_movie={row.movie_id: row.values for row in rows},
            )
        )
    return queries


def candidate_policy_fingerprint(config: ExperimentConfig) -> str:
    payload = {
        "schema": "union-candidate-policy/v1",
        "retrieval_top_k": config.retrieval_top_k,
        "semantic_profile_history_cap": config.semantic_profile_history_cap,
        "semantic_kind": config.semantic_kind,
        "semantic_model_name": config.semantic_model_name,
        "semantic_model_revision": config.semantic_model_revision,
        "semantic_cache_path": config.semantic_cache_path,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ranking_dataset_fingerprint(
    movies: Mapping[int, Movie], split: LeakageSafeRankingSplit
) -> str:
    payload = {
        "split_input_fingerprint": split.input_fingerprint,
        "movies": [
            [movie.movie_id, movie.title, list(movie.genres), movie.year]
            for movie in sorted(movies.values(), key=lambda item: item.movie_id)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_histories(
    rows: tuple[Rating, ...], movies: Mapping[int, Movie]
) -> dict[int, tuple[Rating, ...]]:
    grouped: dict[int, list[Rating]] = defaultdict(list)
    for row in rows:
        if row.rating >= 4 and row.movie_id in movies:
            grouped[row.user_id].append(row)
    return {
        user_id: tuple(sorted(history, key=lambda row: (row.timestamp, row.movie_id)))
        for user_id, history in grouped.items()
    }


def _state_from_history(history_ids: set[int], movies: Mapping[int, Movie]) -> PreferenceState:
    counts: Counter[str] = Counter(
        genre for movie_id in history_ids for genre in movies[movie_id].genres
    )
    return PreferenceState(
        liked_movie_ids=history_ids,
        liked_genres={genre for genre, _ in counts.most_common(3)},
    )


def _single_ndcg(ranked_ids: list[int], target: int) -> float:
    if target not in ranked_ids[:10]:
        return 0.0
    return 1.0 / math.log2(ranked_ids.index(target) + 2)

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from recagent_eval.agent import build_semantic_profile
from recagent_eval.bundle import publish_ranker_bundle
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
    serialize_ranker_artifact,
)
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import ItemCFRetriever, SemanticRetriever, hard_filter
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
    bundle_manifest_output: Path,
    max_users: int,
    seed: int,
    registered_case_fingerprint: str = "unregistered",
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
        semantic_top_k=config.semantic_top_k,
        max_users=max_users,
    )
    cv = cross_validate_lambdamart(
        training_queries,
        estimator_factory=lambda params: make_lgbm_ranker(params, seed=seed),
        seed=seed,
        fold_query_builder=lambda train_users, validation_users: build_fold_queries(
            movies,
            split,
            semantic,
            config,
            train_users=train_users,
            validation_users=validation_users,
        ),
    )
    matrix = build_training_matrix(training_queries)
    estimator = make_lgbm_ranker(cv.selected_params, seed=seed)
    estimator.fit(
        list(matrix.features),
        list(matrix.labels),
        group=list(matrix.groups),
    )
    cv_results = [dict(row) for row in cv.parameter_rows] + [
        {
            "params": row.params,
            "fold": row.fold,
            "train_users": list(row.train_users),
            "validation_users": list(row.validation_users),
            "ndcg_at_10": row.ndcg_at_10,
            "recall_at_10": row.recall_at_10,
            "validation_count": row.validation_count,
            "ndcg_sum": row.ndcg_sum,
            "recall_sum": row.recall_sum,
        }
        for row in cv.fold_rows
    ]
    provenance = {
        "training_rows_fingerprint": _fingerprint_ratings(
            split.ranker_training_history
        ),
        "history_fingerprint": _fingerprint(
            {
                user_id: [
                    [row.movie_id, row.rating, row.timestamp]
                    for row in split.histories[user_id]
                ]
                for user_id in sorted(split.histories)
            }
        ),
        "fold_map_fingerprint": _fingerprint(cv.fold_by_user),
        "fold_map": cv.fold_by_user,
        "group_fingerprint": _fingerprint(
            {"groups": matrix.groups, "users": matrix.user_ids}
        ),
        "candidate_policy_fingerprint": candidate_policy_fingerprint(config),
        "config_fingerprint": lambdamart_config_fingerprint(config),
        "metric_fingerprint": _fingerprint(
            {"metric": "ndcg", "k": 10, "bootstrap_resamples": 2000}
        ),
        "case_fingerprint": registered_case_fingerprint,
        "report_fingerprint": _fingerprint(cv_results),
    }
    learned = LearnedRanker(estimator, legal_train_rows=split.legal_retrieval_train)
    rows = build_validation_rows(
        movies, split, semantic, config, learned, max_users=max_users
    )
    provenance["validation_rows_fingerprint"] = _fingerprint(rows)
    provenance["validation_user_count"] = len(rows)
    artifact = artifact_from_estimator(
        estimator,
        selected_params=cv.selected_params,
        dataset_fingerprint=dataset_fingerprint,
        training_user_count=matrix.training_users,
        training_group_count=len(matrix.groups),
        provenance=provenance,
        cv_results=cv_results,
    )
    policy_fingerprint = candidate_policy_fingerprint(config)
    evidence = build_validation_evidence(
        rows,
        dataset_fingerprint=dataset_fingerprint,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT,
        model_fingerprint=artifact.model_checksum,
        candidate_policy_fingerprint=policy_fingerprint,
        seed=seed,
        provenance={
            **provenance,
            "selected_params": cv.selected_params,
            "cv_results": cv_results,
            "training_user_count": matrix.training_users,
            "training_group_count": len(matrix.groups),
            "dependency_versions": artifact.dependency_versions,
        },
    )
    evidence_bytes = (
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode()
    publish_ranker_bundle(
        serialize_ranker_artifact(artifact),
        evidence_bytes,
        model_output,
        evidence_output,
        bundle_manifest_output,
        {
            "run_fingerprint": evidence.evidence_fingerprint,
            "config_fingerprint": provenance["config_fingerprint"],
            "dataset_fingerprint": dataset_fingerprint,
            "candidate_policy_fingerprint": policy_fingerprint,
            "feature_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
        },
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
        "bundle_manifest_path": str(bundle_manifest_output),
    }


def build_validation_rows(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    config: ExperimentConfig,
    learned: LearnedRanker,
    *,
    max_users: int,
) -> list[dict[str, Any]]:
    validation_histories = _positive_histories(split.legal_retrieval_train, movies)
    validation_queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        validation_histories,
        split.validation_targets,
        semantic,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        semantic_top_k=config.semantic_top_k,
        max_users=max_users,
    )
    baseline = HybridRanker(kind="itemcf")
    rows: list[dict[str, Any]] = []
    for query in validation_queries:
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
                "legal_history_movie_ids": sorted(
                    row.movie_id
                    for row in validation_histories.get(query.user_id, ())
                ),
                "allowed_movie_ids": sorted(feature_rows),
                "lambdamart_ranked_movie_ids": learned_ids,
                "latency_ms": 0.0,
            }
        )
    return rows


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
    states: Mapping[int, PreferenceState] | None = None,
    semantic_top_k: int | None = None,
    semantic_query_builder: Callable[[PreferenceState, dict[int, Movie], int], str]
    | None = None,
) -> list[CandidateQuery]:
    if semantic_top_k is not None and semantic_top_k <= 0:
        raise ValueError("semantic_top_k must be positive")
    dense_top_k = retrieval_top_k if semantic_top_k is None else semantic_top_k
    query_builder = semantic_query_builder or (
        lambda state, movies, history_cap: build_semantic_profile(
            "", state, movies, history_cap=history_cap
        )
    )
    itemcf = ItemCFRetriever.fit(legal_train_rows)
    queries: list[CandidateQuery] = []
    for user_id, target in sorted(targets.items())[:max_users]:
        history_rows = histories.get(user_id, ())
        history_ids = {
            row.movie_id for row in history_rows if row.rating >= 4 and row.movie_id in movies
        }
        state = (
            states[user_id]
            if states is not None and user_id in states
            else _state_from_history(history_ids, movies)
        )
        allowed_ids = {
            movie.movie_id for movie in hard_filter(movies.values(), state)
        } - history_ids
        itemcf_scores = dict(
            itemcf.retrieve(history_ids, top_k=retrieval_top_k, allowed_ids=allowed_ids)
        )
        dense_scores = dict(
            semantic.retrieve(
                query_builder(state, movies, history_cap),
                top_k=dense_top_k,
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


def build_fold_queries(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    config: ExperimentConfig,
    *,
    train_users: tuple[int, ...],
    validation_users: tuple[int, ...],
) -> tuple[list[CandidateQuery], list[CandidateQuery]]:
    """Rebuild retrieval and global statistics from training-fold users only."""
    training_user_set = set(train_users)
    fold_train_rows = tuple(
        row
        for row in split.ranker_training_history
        if row.user_id in training_user_set
    )
    common = {
        "movies": movies,
        "legal_train_rows": fold_train_rows,
        "semantic": semantic,
        "retrieval_top_k": config.retrieval_top_k,
        "history_cap": config.semantic_profile_history_cap,
        "semantic_top_k": config.semantic_top_k,
    }
    return (
        build_candidate_queries(
            **common,
            histories={user_id: split.histories[user_id] for user_id in train_users},
            targets={user_id: split.ranker_targets[user_id] for user_id in train_users},
            max_users=len(train_users),
        ),
        build_candidate_queries(
            **common,
            histories={
                user_id: split.histories[user_id] for user_id in validation_users
            },
            targets={
                user_id: split.ranker_targets[user_id]
                for user_id in validation_users
            },
            max_users=len(validation_users),
        ),
    )


def candidate_policy_fingerprint(config: ExperimentConfig) -> str:
    payload = {
        "schema": "union-candidate-policy/v1",
        "retrieval_top_k": config.retrieval_top_k,
        "semantic_profile_history_cap": config.semantic_profile_history_cap,
        "semantic_kind": config.semantic_kind,
        "semantic_model_name": config.semantic_model_name,
        "semantic_model_revision": config.semantic_model_revision,
        "semantic_cache_path": config.semantic_cache_path,
        "semantic_top_k": config.semantic_top_k,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lambdamart_config_fingerprint(config: ExperimentConfig) -> str:
    return _fingerprint(
        {
            "retrieval_top_k": config.retrieval_top_k,
            "semantic_profile_history_cap": config.semantic_profile_history_cap,
            "semantic_kind": config.semantic_kind,
            "semantic_model_name": config.semantic_model_name,
            "semantic_model_revision": config.semantic_model_revision,
            "semantic_cache_path": config.semantic_cache_path,
            "semantic_top_k": config.semantic_top_k,
            "seed": config.seed,
        }
    )


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


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fingerprint_ratings(rows: tuple[Rating, ...]) -> str:
    return _fingerprint(
        [
            [row.user_id, row.movie_id, row.rating, row.timestamp]
            for row in rows
        ]
    )

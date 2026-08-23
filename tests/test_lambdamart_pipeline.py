from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from recagent_eval.bundle import load_ranker_bundle
from recagent_eval.candidate_features import FEATURE_NAMES_V2
from recagent_eval.config import load_experiment_config
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    build_candidate_queries,
    candidate_policy_fingerprint,
    lambdamart_config_fingerprint,
    ranking_dataset_fingerprint,
    train_lambdamart_pipeline,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.learned_ranking import parse_ranker_artifact
from recagent_eval.runner import ExperimentConfig
from recagent_eval.v2_selection import LearnedValidationEvidence, validate_learned_gate


class _CatalogSemanticRetriever:
    kind = "synthetic"

    def retrieve(self, query, *, top_k=100, allowed_ids=None):
        del query
        return [
            (movie_id, 1.0 / index)
            for index, movie_id in enumerate(sorted(allowed_ids or ()), start=1)
        ][:top_k]


def test_training_pipeline_publishes_bound_model_evidence_and_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    movies = {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            1990 + movie_id,
        )
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 6)
    ]
    split = leakage_safe_ranking_split(ratings)
    config = ExperimentConfig(
        name="synthetic-lambdamart",
        semantic_kind="tfidf",
        retrieval_top_k=8,
        semantic_profile_history_cap=2,
        seed=17,
    )
    model_path = tmp_path / "ranker.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"

    summary = train_lambdamart_pipeline(
        movies,
        split,
        _CatalogSemanticRetriever(),
        config,
        model_output=model_path,
        evidence_output=evidence_path,
        bundle_manifest_output=manifest_path,
        max_users=6,
        seed=17,
        registered_case_fingerprint="registered-cases",
    )

    bundle = load_ranker_bundle(
        model_path, evidence_path, manifest_path
    )
    model_bytes = bundle.model_bytes
    evidence_bytes = bundle.evidence_bytes
    artifact = parse_ranker_artifact(
        model_bytes,
        expected_dataset_fingerprint=ranking_dataset_fingerprint(movies, split),
        expected_candidate_policy_fingerprint=candidate_policy_fingerprint(config),
        expected_config_fingerprint=lambdamart_config_fingerprint(config),
        expected_case_fingerprint="registered-cases",
    )
    evidence = LearnedValidationEvidence.model_validate_json(evidence_bytes)
    manifest = json.loads(manifest_path.read_text())

    assert summary["training_users"] == 6
    assert summary["validation_users"] == 6
    assert summary["model_checksum"] == artifact.model_checksum
    assert evidence.model_fingerprint == artifact.model_checksum
    assert evidence.validation_user_count == 6
    assert evidence.training_group_count == 6
    assert evidence.fold_map == artifact.fold_map
    assert evidence.case_fingerprint == "registered-cases"
    assert manifest["run_fingerprint"] == evidence.evidence_fingerprint
    assert manifest["model_sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert all(row["constraint_satisfied"] for row in evidence.per_user_rows)

    with pytest.raises(ValueError, match="did not improve"):
        validate_learned_gate(
            evidence,
            dataset_fingerprint=summary["dataset_fingerprint"],
            feature_fingerprint=summary["feature_fingerprint"],
            model_fingerprint=summary["model_checksum"],
            candidate_policy_fingerprint=summary["candidate_policy_fingerprint"],
            case_fingerprint="registered-cases",
            config_fingerprint=lambdamart_config_fingerprint(config),
            artifact_provenance=artifact.model_dump(mode="python"),
        )


def test_candidate_policy_and_config_fingerprints_include_semantic_top_k() -> None:
    base = ExperimentConfig(
        name="dense",
        semantic_kind="dense",
        semantic_cache_path="cache.npz",
        retrieval_top_k=500,
    )
    widened = ExperimentConfig(
        name="dense",
        semantic_kind="dense",
        semantic_cache_path="cache.npz",
        retrieval_top_k=500,
        semantic_top_k=1500,
    )
    assert candidate_policy_fingerprint(base) != candidate_policy_fingerprint(widened)
    assert lambdamart_config_fingerprint(base) != lambdamart_config_fingerprint(widened)


def test_build_candidate_queries_threads_latent_scores() -> None:
    movies = {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            1990 + movie_id,
        )
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 9)
    ]
    split = leakage_safe_ranking_split(ratings)
    latent = LatentFactorRetriever.fit(split.legal_retrieval_train, seed=42)
    queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        _positive_histories(split.legal_retrieval_train, movies),
        split.validation_targets,
        _CatalogSemanticRetriever(),
        retrieval_top_k=10,
        history_cap=5,
        max_users=3,
        semantic_top_k=20,
        latent=latent,
        latent_top_k=10,
        feature_version="v2",
    )
    assert queries
    row = next(iter(queries[0].features_by_movie.values()))
    assert len(row) == len(FEATURE_NAMES_V2)


def test_fingerprints_change_only_when_latent_enabled() -> None:
    base = load_experiment_config(Path("configs/v2_dense_recall1500.yaml"))
    assert candidate_policy_fingerprint(base) == (
        "a3c3475fec9b49b3e67923a73e97d10c2017031050abcbc8f1e468824b52eb41"
    )
    enabled = replace(
        base,
        latent_enabled=True,
        latent_artifact_path="artifacts/experiments/x/latent.npz",
        ranker_feature_version="v2",
        ranker_negative_policy="route_balanced",
        ranker_max_negatives=200,
    )
    assert candidate_policy_fingerprint(enabled) != candidate_policy_fingerprint(base)
    assert lambdamart_config_fingerprint(enabled) != lambdamart_config_fingerprint(base)


def test_build_candidate_queries_v2b_threads_recent_itemcf_scores() -> None:
    from recagent_eval.candidate_features import FEATURE_NAMES_V2B

    movies = {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            1990 + movie_id,
        )
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 9)
    ]
    split = leakage_safe_ranking_split(ratings)
    latent = LatentFactorRetriever.fit(split.legal_retrieval_train, seed=42)
    queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        _positive_histories(split.legal_retrieval_train, movies),
        split.validation_targets,
        _CatalogSemanticRetriever(),
        retrieval_top_k=10,
        history_cap=5,
        max_users=3,
        semantic_top_k=20,
        latent=latent,
        latent_top_k=10,
        feature_version="v2b",
    )
    assert queries
    row = next(iter(queries[0].features_by_movie.values()))
    assert len(row) == len(FEATURE_NAMES_V2B)
    recent_index = FEATURE_NAMES_V2B.index("recent_itemcf_score")
    assert row[recent_index] >= 0.0


def test_train_pipeline_restricts_training_users_and_eval_user_ids(tmp_path) -> None:
    movies = {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            1990 + movie_id,
        )
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 7)
    ]
    split = leakage_safe_ranking_split(ratings)
    config = ExperimentConfig(
        name="synthetic",
        semantic_kind="tfidf",
        retrieval_top_k=8,
        semantic_profile_history_cap=2,
        seed=17,
    )
    eligible = sorted(split.validation_targets)
    train_users = tuple(eligible[:4])
    eval_users = tuple(eligible[4:6])
    model_path = tmp_path / "ranker.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"
    summary = train_lambdamart_pipeline(
        movies,
        split,
        _CatalogSemanticRetriever(),
        config,
        model_output=model_path,
        evidence_output=evidence_path,
        bundle_manifest_output=manifest_path,
        max_users=6,
        seed=17,
        registered_case_fingerprint="registered-cases",
        training_user_ids=train_users,
        eval_user_ids=eval_users,
    )
    assert summary["validation_users"] == 2
    bundle = load_ranker_bundle(model_path, evidence_path, manifest_path)
    evidence = LearnedValidationEvidence.model_validate_json(bundle.evidence_bytes)
    assert [row["user_id"] for row in evidence.per_user_rows] == list(eval_users)

from __future__ import annotations

import hashlib
import json

import pytest

from recagent_eval.bundle import load_ranker_bundle
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.lambdamart_pipeline import (
    candidate_policy_fingerprint,
    lambdamart_config_fingerprint,
    ranking_dataset_fingerprint,
    train_lambdamart_pipeline,
)
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

    model_bytes, evidence_bytes = load_ranker_bundle(
        model_path, evidence_path, manifest_path
    )
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

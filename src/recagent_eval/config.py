from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import yaml

from recagent_eval.models import ToolName
from recagent_eval.ranking import RankerKind
from recagent_eval.runner import ExperimentConfig

RETRIEVAL_TOOLS = {"itemcf_retrieve", "semantic_retrieve"}


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text()) or {}
    weights = tuple(float(value) for value in payload.get("weights", (0.5, 0.3, 0.2)))
    if len(weights) != 3 or not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
        raise ValueError("weights must contain three values that sum to 1")
    ranker_payload = payload.get("ranker") or {}
    ranker_kind = str(ranker_payload.get("kind", "minmax_linear"))
    allowed_rankers = {"itemcf", "minmax_linear", "rrf", "percentile_linear", "lambdamart"}
    if ranker_kind not in allowed_rankers:
        raise ValueError(
            "ranker.kind must be itemcf, minmax_linear, rrf, percentile_linear, or lambdamart"
        )
    rrf_k = int(ranker_payload.get("rrf_k", 60))
    if rrf_k <= 0:
        raise ValueError("ranker.rrf_k must be positive")
    model_path_value = ranker_payload.get("model_path")
    learned_model_path = str(model_path_value).strip() if model_path_value is not None else None
    if model_path_value is not None and not learned_model_path:
        raise ValueError("ranker.model_path must not be empty")
    evidence_path_value = ranker_payload.get("evidence_path")
    learned_evidence_path = (
        str(evidence_path_value).strip() if evidence_path_value is not None else None
    )
    if evidence_path_value is not None and not learned_evidence_path:
        raise ValueError("ranker.evidence_path must not be empty")
    learned_values = {
        name: (
            str(ranker_payload.get(name)).strip()
            if ranker_payload.get(name) is not None
            else None
        )
        for name in (
            "dataset_fingerprint",
            "candidate_policy_fingerprint",
            "config_fingerprint",
            "case_fingerprint",
            "gate_fingerprint",
            "consumption_dir",
        )
    }
    if any(value == "" for value in learned_values.values()):
        raise ValueError("learned ranker provenance values must not be empty")
    if "weights" in ranker_payload:
        route_weights = tuple(float(value) for value in ranker_payload["weights"])
        if any(value < 0 for value in route_weights):
            raise ValueError("ranker.weights must be non-negative")
        if len(route_weights) != 2 or not math.isclose(sum(route_weights), 1.0, abs_tol=1e-8):
            raise ValueError("ranker.weights must contain two values that sum to 1")
        weights = (route_weights[0], route_weights[1], 0.0)
    required_retrieval_tools: tuple[ToolName, ...] = tuple(  # type: ignore[assignment]
        str(value)
        for value in payload.get(
            "required_retrieval_tools",
            ("itemcf_retrieve",),
        )
    )
    if not required_retrieval_tools or not set(required_retrieval_tools).issubset(RETRIEVAL_TOOLS):
        raise ValueError(
            "required_retrieval_tools must contain itemcf_retrieve and/or semantic_retrieve"
        )
    semantic_payload = payload.get("semantic", {})
    if not isinstance(semantic_payload, dict):
        raise ValueError("semantic must be a mapping")
    semantic_kind = str(semantic_payload.get("kind", "tfidf"))
    if semantic_kind not in {"tfidf", "dense"}:
        raise ValueError("semantic.kind must be tfidf or dense")
    semantic_model_name = str(
        semantic_payload.get(
            "model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    ).strip()
    if not semantic_model_name:
        raise ValueError("semantic.model_name must not be empty")
    revision_value = semantic_payload.get("model_revision")
    semantic_model_revision = str(revision_value).strip() if revision_value is not None else None
    if revision_value is not None and not semantic_model_revision:
        raise ValueError("semantic.model_revision must not be empty")
    cache_value = semantic_payload.get("cache_path")
    semantic_cache_path = str(cache_value).strip() if cache_value is not None else None
    if cache_value is not None and not semantic_cache_path:
        raise ValueError("semantic.cache_path must not be empty")
    semantic_device = str(semantic_payload.get("device", "cpu"))
    if semantic_device not in {"cpu", "cuda"}:
        raise ValueError("semantic.device must be cpu or cuda")
    return ExperimentConfig(
        name=str(payload.get("name") or path.stem),
        weights=weights,
        ranker_kind=cast(RankerKind, ranker_kind),
        rrf_k=rrf_k,
        retrieval_top_k=int(payload.get("retrieval_top_k", 100)),
        enable_memory=bool(payload.get("enable_memory", True)),
        enable_semantic_retrieval=bool(payload.get("enable_semantic_retrieval", True)),
        structured_planning=bool(payload.get("structured_planning", True)),
        required_retrieval_tools=required_retrieval_tools,
        semantic_profile_history_cap=int(payload.get("semantic_profile_history_cap", 20)),
        semantic_kind=semantic_kind,
        semantic_model_name=semantic_model_name,
        semantic_model_revision=semantic_model_revision,
        semantic_cache_path=semantic_cache_path,
        semantic_device=semantic_device,
        learned_model_path=learned_model_path,
        learned_evidence_path=learned_evidence_path,
        learned_dataset_fingerprint=learned_values["dataset_fingerprint"],
        learned_candidate_policy_fingerprint=learned_values[
            "candidate_policy_fingerprint"
        ],
        learned_config_fingerprint=learned_values["config_fingerprint"],
        learned_case_fingerprint=learned_values["case_fingerprint"],
        learned_gate_fingerprint=learned_values["gate_fingerprint"],
        learned_consumption_dir=learned_values["consumption_dir"],
        seed=int(payload.get("seed", 42)),
    )

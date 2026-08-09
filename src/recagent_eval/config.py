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
    allowed_rankers = {"itemcf", "minmax_linear", "rrf", "percentile_linear"}
    if ranker_kind not in allowed_rankers:
        raise ValueError(
            "ranker.kind must be itemcf, minmax_linear, rrf, or percentile_linear"
        )
    rrf_k = int(ranker_payload.get("rrf_k", 60))
    if rrf_k <= 0:
        raise ValueError("ranker.rrf_k must be positive")
    if "weights" in ranker_payload:
        route_weights = tuple(float(value) for value in ranker_payload["weights"])
        if any(value < 0 for value in route_weights):
            raise ValueError("ranker.weights must be non-negative")
        if len(route_weights) != 2 or not math.isclose(
            sum(route_weights), 1.0, abs_tol=1e-8
        ):
            raise ValueError("ranker.weights must contain two values that sum to 1")
        weights = (route_weights[0], route_weights[1], 0.0)
    required_retrieval_tools: tuple[ToolName, ...] = tuple(  # type: ignore[assignment]
        str(value)
        for value in payload.get(
            "required_retrieval_tools",
            ("itemcf_retrieve",),
        )
    )
    if not required_retrieval_tools or not set(required_retrieval_tools).issubset(
        RETRIEVAL_TOOLS
    ):
        raise ValueError(
            "required_retrieval_tools must contain itemcf_retrieve "
            "and/or semantic_retrieve"
        )
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
        semantic_profile_history_cap=int(
            payload.get("semantic_profile_history_cap", 20)
        ),
        seed=int(payload.get("seed", 42)),
    )

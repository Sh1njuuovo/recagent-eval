from __future__ import annotations

import math
from pathlib import Path

import yaml

from recagent_eval.models import ToolName
from recagent_eval.runner import ExperimentConfig

RETRIEVAL_TOOLS = {"itemcf_retrieve", "semantic_retrieve"}


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text()) or {}
    weights = tuple(float(value) for value in payload.get("weights", (0.5, 0.3, 0.2)))
    if len(weights) != 3 or not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
        raise ValueError("weights must contain three values that sum to 1")
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

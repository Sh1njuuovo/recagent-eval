from __future__ import annotations

import math
from pathlib import Path

import yaml

from recagent_eval.runner import ExperimentConfig


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text()) or {}
    weights = tuple(float(value) for value in payload.get("weights", (0.5, 0.3, 0.2)))
    if len(weights) != 3 or not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
        raise ValueError("weights must contain three values that sum to 1")
    return ExperimentConfig(
        name=str(payload.get("name") or path.stem),
        weights=weights,
        retrieval_top_k=int(payload.get("retrieval_top_k", 100)),
        enable_memory=bool(payload.get("enable_memory", True)),
        enable_semantic_retrieval=bool(payload.get("enable_semantic_retrieval", True)),
        structured_planning=bool(payload.get("structured_planning", True)),
        seed=int(payload.get("seed", 42)),
    )

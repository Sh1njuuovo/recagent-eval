from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recagent_eval.agent import AgentConfig, RecommendationAgent
from recagent_eval.cases import EvaluationCase, validate_cases_relevance
from recagent_eval.data import Movie, Rating
from recagent_eval.evaluation import (
    EvaluationRecord,
    aggregate_metrics,
    build_candidate_diagnostics,
    pipeline_compliant,
)
from recagent_eval.models import ToolName
from recagent_eval.provider import LLMProvider
from recagent_eval.ranking import HybridRanker, RankerKind
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    ranker_kind: RankerKind = "minmax_linear"
    rrf_k: int = 60
    retrieval_top_k: int = 100
    enable_memory: bool = True
    enable_semantic_retrieval: bool = True
    structured_planning: bool = True
    required_retrieval_tools: tuple[ToolName, ...] = ("itemcf_retrieve",)
    semantic_profile_history_cap: int = 20
    seed: int = 42


def run_experiment(
    *,
    movies: dict[int, Movie],
    ratings: list[Rating],
    cases: list[EvaluationCase],
    provider: LLMProvider,
    config: ExperimentConfig,
    output_dir: Path,
) -> dict[str, float | int]:
    validate_cases_relevance(cases, movies)
    random.seed(config.seed)
    np.random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    itemcf = ItemCFRetriever.fit(ratings)
    semantic = TfidfSemanticRetriever.fit(movies)
    agent = RecommendationAgent(
        movies=movies,
        itemcf=itemcf,
        semantic=semantic,
        ranker=HybridRanker(config.weights),
        provider=provider,
        config=AgentConfig(
            retrieval_top_k=config.retrieval_top_k,
            enable_memory=config.enable_memory,
            enable_semantic_retrieval=config.enable_semantic_retrieval,
            structured_planning=config.structured_planning,
            required_retrieval_tools=config.required_retrieval_tools,
            semantic_profile_history_cap=config.semantic_profile_history_cap,
        ),
    )

    records: list[EvaluationRecord] = []
    serialized: list[dict[str, object]] = []
    for case in cases:
        state = case.initial_state
        result = None
        turn_results = []
        for turn in case.turns:
            result = agent.recommend(turn, state)
            turn_results.append(result)
            state = result.preference_state
        if result is None:
            continue
        final_turn_traces = turn_results[-1].traces
        candidate_diagnostics = build_candidate_diagnostics(
            case.relevant_movie_ids,
            movies,
            result.preference_state,
            final_turn_traces,
        )
        is_pipeline_compliant = pipeline_compliant(
            final_turn_traces,
            config.required_retrieval_tools,
        )
        result = result.model_copy(
            update={
                "traces": [trace for turn_result in turn_results for trace in turn_result.traces],
                "latency_ms": sum(item.latency_ms for item in turn_results),
                "llm_calls": sum(item.llm_calls for item in turn_results),
                "prompt_tokens": sum(item.prompt_tokens for item in turn_results),
                "completion_tokens": sum(item.completion_tokens for item in turn_results),
                "errors": [error for turn_result in turn_results for error in turn_result.errors],
                "fallback_used": any(item.fallback_used for item in turn_results),
                "plan_valid": all(item.plan_valid for item in turn_results),
            }
        )
        records.append(
            EvaluationRecord(
                result=result,
                relevant_movie_ids=case.relevant_movie_ids,
                expected_preferences=case.expected_preferences,
                metadata={
                    "case_id": case.case_id,
                    "tags": case.tags,
                    "label_eligible": True,
                    "candidate_diagnostics": candidate_diagnostics,
                    "pipeline_compliant": is_pipeline_compliant,
                },
            )
        )
        serialized.append(
            {
                "case_id": case.case_id,
                "user_id": case.user_id,
                "recommended_movie_ids": [movie.movie_id for movie in result.movies],
                "relevant_movie_ids": sorted(case.relevant_movie_ids),
                "candidate_diagnostics": candidate_diagnostics,
                "pipeline_compliant": is_pipeline_compliant,
                "turn_results": [
                    {
                        "turn_index": index,
                        "plan_valid": turn_result.plan_valid,
                        "fallback_used": turn_result.fallback_used,
                        "errors": turn_result.errors,
                    }
                    for index, turn_result in enumerate(turn_results, start=1)
                ],
                "result": result.model_dump(mode="json"),
            }
        )

    metrics = aggregate_metrics(records, movies, k=10)
    _write_jsonl(output_dir / "episodes.jsonl", serialized)
    _write_json(output_dir / "metrics.json", metrics)
    manifest = {
        "experiment": config.name,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": config.seed,
        "weights": config.weights,
        "ranker": {
            "kind": config.ranker_kind,
            "rrf_k": config.rrf_k,
            "weights": config.weights,
        },
        "retrieval_top_k": config.retrieval_top_k,
        "enable_memory": config.enable_memory,
        "enable_semantic_retrieval": config.enable_semantic_retrieval,
        "structured_planning": config.structured_planning,
        "required_retrieval_tools": config.required_retrieval_tools,
        "semantic_profile_history_cap": config.semantic_profile_history_cap,
        "movie_count": len(movies),
        "rating_count": len(ratings),
        "case_count": len(cases),
        "case_fingerprint": _case_fingerprint(cases),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return metrics


def _case_fingerprint(cases: list[EvaluationCase]) -> str:
    canonical = json.dumps(
        canonical_case_payload(cases),
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def canonical_case_payload(cases: list[EvaluationCase]) -> list[dict[str, Any]]:
    return [_canonicalize(case.model_dump(mode="python")) for case in cases]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )

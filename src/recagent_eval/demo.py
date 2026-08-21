from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Protocol

from pydantic import BaseModel, Field

from recagent_eval.agent import AgentConfig, RecommendationAgent
from recagent_eval.config import load_experiment_config
from recagent_eval.data import chronological_split, load_movielens_movies, load_movielens_ratings
from recagent_eval.learned_ranking import (
    LearnedRanker,
    estimator_from_artifact,
    parse_ranker_artifact,
)
from recagent_eval.models import PreferenceState, RecommendationResult
from recagent_eval.provider import ProviderSelection, ProviderStatus, build_provider
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import DenseSemanticRetriever, ItemCFRetriever, TfidfSemanticRetriever


class DemoAgent(Protocol):
    def recommend(
        self, message: str, state: PreferenceState | None = None
    ) -> RecommendationResult: ...


class DemoSessionState(BaseModel):
    preference_state: PreferenceState = Field(default_factory=PreferenceState)
    history: list[dict[str, str]] = Field(default_factory=list)


@dataclass(frozen=True)
class DemoTurn:
    output: str
    session: DemoSessionState
    diagnostics: dict[str, Any]


def format_recommendations(result: RecommendationResult) -> str:
    """Keep the original text formatter stable for scripts and interviews."""
    if not result.movies:
        detail = "; ".join(result.errors) or "No eligible movies found."
        return f"No recommendations. {detail}"
    lines = []
    for index, movie in enumerate(result.movies, start=1):
        genres = ", ".join(movie.genres)
        lines.append(
            f"{index}. **{movie.title}** · score={movie.score.final:.3f} "
            f"· {genres}\n   {movie.reason}"
        )
    if result.fallback_used:
        lines.append("\n_Note: deterministic fallback was used after plan validation failed._")
    return "\n".join(lines)


def select_demo_provider(
    name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderSelection:
    env = os.environ if environ is None else environ
    requested = name or env.get("RECAGENT_DEMO_PROVIDER", "deepseek")
    return build_provider(requested, allow_fallback=True, environ=env)


def handle_message(
    message: str,
    session: DemoSessionState | Mapping[str, Any] | None,
    *,
    agent: DemoAgent,
    provider_status: ProviderStatus,
) -> DemoTurn:
    """Process one request without retaining mutable cross-session state."""
    current = _session_from_value(session)
    if not message.strip():
        diagnostics = _empty_diagnostics(
            provider_status,
            current.preference_state,
            agent,
            "Please enter a movie request.",
        )
        return DemoTurn(
            output="Please enter a movie request so I can recommend something.",
            session=current,
            diagnostics=diagnostics,
        )
    result = agent.recommend(message.strip(), current.preference_state)
    output = format_recommendations(result)
    if provider_status.fallback and provider_status.message:
        output = f"{output}\n\n_Provider status: {provider_status.message}_"
    updated = DemoSessionState(
        preference_state=result.preference_state,
        history=[
            *current.history,
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": output},
        ],
    )
    return DemoTurn(
        output=output,
        session=updated,
        diagnostics=serialize_diagnostics(result, provider_status, agent=agent),
    )


def reset_session() -> DemoSessionState:
    return DemoSessionState()


def serialize_diagnostics(
    result: RecommendationResult,
    provider_status: ProviderStatus,
    *,
    agent: DemoAgent | None = None,
) -> dict[str, Any]:
    sources_by_movie: dict[int, list[str]] = {}
    for trace in result.traces:
        for movie_id in trace.candidate_movie_ids:
            sources_by_movie.setdefault(movie_id, []).append(trace.tool)
    recommendations = []
    for movie in result.movies:
        contributions = sorted(
            movie.score.feature_contributions.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:5]
        recommendations.append(
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "candidate_sources": sources_by_movie.get(movie.movie_id, []),
                "score_breakdown": movie.score.model_dump(mode="json"),
                "top_feature_contributions": [
                    {"feature": name, "contribution": value}
                    for name, value in contributions
                ],
                "reason": movie.reason,
            }
        )
    provider = _safe_provider_status(provider_status)
    provider["fallback"] = provider_status.fallback or result.fallback_used
    return {
        "preference_state": result.preference_state.model_dump(mode="json"),
        "validated_tool_plan": (
            result.plan.model_dump(mode="json") if result.plan is not None else None
        ),
        "tool_traces": [trace.model_dump(mode="json") for trace in result.traces],
        "recommendations": recommendations,
        "provider": provider,
        "pipeline": _pipeline_identity(agent),
        "fallback_used": provider_status.fallback or result.fallback_used,
        "errors": [error[:500] for error in result.errors],
        "latency_ms": result.latency_ms,
        "llm_calls": result.llm_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def build_agent(
    data_dir: Path,
    *,
    provider_name: str | None = None,
    semantic_config_path: Path | None = None,
    ranker_config_path: Path | None = None,
) -> RecommendationAgent:
    selection = select_demo_provider(provider_name)
    return _build_agent(
        data_dir,
        selection,
        semantic_config_path=semantic_config_path,
        ranker_config_path=ranker_config_path,
    )


def _build_agent(
    data_dir: Path,
    selection: ProviderSelection,
    *,
    semantic_config_path: Path | None,
    ranker_config_path: Path | None,
) -> RecommendationAgent:
    movies = load_movielens_movies(data_dir / "movies.dat")
    ratings = load_movielens_ratings(data_dir / "ratings.dat")
    split = chronological_split(ratings)
    semantic_config = (
        load_experiment_config(semantic_config_path)
        if semantic_config_path is not None
        else None
    )
    ranker_config = (
        load_experiment_config(ranker_config_path)
        if ranker_config_path is not None
        else None
    )
    if semantic_config is not None and semantic_config.semantic_kind == "dense":
        if semantic_config.semantic_cache_path is None:
            semantic = DenseSemanticRetriever.fit(
                movies,
                model_name=semantic_config.semantic_model_name,
                model_revision=semantic_config.semantic_model_revision,
                device=semantic_config.semantic_device,
            )
        else:
            semantic = DenseSemanticRetriever.load(
                Path(semantic_config.semantic_cache_path),
                movies=movies,
                model_name=semantic_config.semantic_model_name,
                model_revision=semantic_config.semantic_model_revision,
                device=semantic_config.semantic_device,
            )
    else:
        semantic = TfidfSemanticRetriever.fit(movies)
    config = ranker_config or semantic_config
    if config is not None and config.ranker_kind == "lambdamart":
        if config.learned_model_path is None:
            raise ValueError("ranker.model_path is required for the LambdaMART demo")
        artifact = parse_ranker_artifact(Path(config.learned_model_path).read_bytes())
        ranker = LearnedRanker(
            estimator_from_artifact(artifact),
            legal_train_rows=split.train,
        )
        ranker.artifact_id = f"sha256:{artifact.model_checksum[:12]}"
    else:
        ranker = HybridRanker(
            config.weights if config is not None else (0.7, 0.3, 0.0),
            kind=config.ranker_kind if config is not None else "minmax_linear",
            rrf_k=config.rrf_k if config is not None else 60,
        )
    return RecommendationAgent(
        movies=movies,
        itemcf=ItemCFRetriever.fit(split.train),
        semantic=semantic,
        ranker=ranker,
        provider=selection.provider,
        config=AgentConfig(
            retrieval_top_k=config.retrieval_top_k if config is not None else 100,
            required_retrieval_tools=(
                config.required_retrieval_tools
                if config is not None
                else ("itemcf_retrieve", "semantic_retrieve")
            ),
        ),
    )


def build_demo(
    data_dir: Path = Path("data/raw/ml-1m"),
    *,
    provider_name: str | None = None,
    semantic_config_path: Path | None = None,
    ranker_config_path: Path | None = None,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install demo dependencies with: uv sync --extra demo") from exc

    selection = select_demo_provider(provider_name)
    agent = _build_agent(
        data_dir,
        selection,
        semantic_config_path=semantic_config_path,
        ranker_config_path=ranker_config_path,
    )

    with gr.Blocks(title="RecAgent-Eval") as blocks:
        gr.Markdown("# RecAgent-Eval\nSession-isolated, explainable movie recommendations.")
        session_state = gr.State(reset_session().model_dump(mode="json"))
        chatbot = gr.Chatbot(type="messages", label="Conversation")
        message = gr.Textbox(label="Movie request", placeholder="Try: thoughtful space movies")
        with gr.Row():
            submit = gr.Button("Recommend", variant="primary")
            reset = gr.Button("Reset session")
        with gr.Row():
            preference_panel = gr.JSON(label="PreferenceState")
            plan_panel = gr.JSON(label="Validated ToolPlan")
        trace_panel = gr.JSON(label="Tool trace")
        recommendation_panel = gr.JSON(label="Candidate sources and ScoreBreakdown")
        runtime_panel = gr.JSON(label="Provider, fallback, errors, latency and tokens")

        def respond(text: str, serialized_session: dict[str, Any]):
            turn = handle_message(
                text,
                serialized_session,
                agent=agent,
                provider_status=selection.status,
            )
            diagnostics = turn.diagnostics
            runtime = {
                key: diagnostics[key]
                for key in (
                    "provider",
                    "pipeline",
                    "fallback_used",
                    "errors",
                    "latency_ms",
                    "llm_calls",
                    "prompt_tokens",
                    "completion_tokens",
                )
            }
            return (
                turn.session.history,
                "",
                turn.session.model_dump(mode="json"),
                diagnostics["preference_state"],
                diagnostics["validated_tool_plan"],
                diagnostics["tool_traces"],
                diagnostics["recommendations"],
                runtime,
            )

        outputs = [
            chatbot,
            message,
            session_state,
            preference_panel,
            plan_panel,
            trace_panel,
            recommendation_panel,
            runtime_panel,
        ]
        submit.click(respond, [message, session_state], outputs)
        message.submit(respond, [message, session_state], outputs)
        reset.click(
            lambda: ([], "", reset_session().model_dump(mode="json"), {}, {}, [], [], {}),
            outputs=outputs,
        )
    return blocks


def launch(
    data_dir: Path = Path("data/raw/ml-1m"),
    *,
    provider_name: str | None = None,
    semantic_config_path: Path | None = None,
    ranker_config_path: Path | None = None,
) -> None:
    build_demo(
        data_dir,
        provider_name=provider_name,
        semantic_config_path=semantic_config_path,
        ranker_config_path=ranker_config_path,
    ).launch()


def _session_from_value(
    session: DemoSessionState | Mapping[str, Any] | None,
) -> DemoSessionState:
    if session is None:
        return DemoSessionState()
    if isinstance(session, DemoSessionState):
        return session.model_copy(deep=True)
    return DemoSessionState.model_validate(dict(session))


def _empty_diagnostics(
    status: ProviderStatus,
    state: PreferenceState,
    agent: DemoAgent,
    error: str,
) -> dict[str, Any]:
    return {
        "preference_state": state.model_dump(mode="json"),
        "validated_tool_plan": None,
        "tool_traces": [],
        "recommendations": [],
        "provider": _safe_provider_status(status),
        "pipeline": _pipeline_identity(agent),
        "fallback_used": status.fallback,
        "errors": [error],
        "latency_ms": 0.0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def _pipeline_identity(agent: DemoAgent | None) -> dict[str, Any]:
    if agent is None:
        return {
            "ranker": {"kind": "unknown", "artifact_id": None},
            "semantic": {"kind": "unknown", "model": None, "revision": None},
        }
    ranker = getattr(agent, "ranker", None)
    semantic = getattr(agent, "semantic", None)
    ranker_kind = str(getattr(ranker, "kind", type(ranker).__name__ if ranker else "unknown"))
    artifact_id = getattr(ranker, "artifact_id", None)
    artifact_digest = artifact_id.removeprefix("sha256:") if isinstance(artifact_id, str) else ""
    if (
        not isinstance(artifact_id, str)
        or not artifact_id.startswith("sha256:")
        or not 12 <= len(artifact_digest) <= 64
        or any(character not in "0123456789abcdef" for character in artifact_digest)
    ):
        artifact_id = None
    semantic_kind = str(
        getattr(semantic, "kind", type(semantic).__name__ if semantic else "unknown")
    )
    model = _safe_model_identity(getattr(semantic, "model_name", None))
    revision = _safe_model_identity(getattr(semantic, "model_revision", None))
    return {
        "ranker": {"kind": ranker_kind, "artifact_id": artifact_id},
        "semantic": {"kind": semantic_kind, "model": model, "revision": revision},
    }


def _safe_provider_status(status: ProviderStatus) -> dict[str, Any]:
    provider = asdict(status)
    provider["model"] = _safe_model_identity(status.model)
    return provider


def _safe_model_identity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if (
        Path(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or text.startswith(("./", "../", "~", "\\\\", "//"))
    ):
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"local-model:sha256:{digest}"
    return text[:200]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the RecAgent-Eval demo")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--provider", choices=("rule-based", "deepseek", "vllm", "qwen"))
    parser.add_argument("--semantic-config", type=Path)
    parser.add_argument("--ranker-config", type=Path)
    args = parser.parse_args()
    launch(
        args.data_dir,
        provider_name=args.provider,
        semantic_config_path=args.semantic_config,
        ranker_config_path=args.ranker_config,
    )


if __name__ == "__main__":
    main()

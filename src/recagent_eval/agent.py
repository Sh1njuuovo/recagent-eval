from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from recagent_eval.data import Movie
from recagent_eval.models import (
    PreferencePatch,
    PreferenceState,
    RecommendationResult,
    ToolName,
    ToolPlan,
    ToolStep,
    ToolTrace,
)
from recagent_eval.provider import LLMProvider, LLMResponse
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import (
    ItemCFRetriever,
    TfidfSemanticRetriever,
    hard_filter,
)


@dataclass(frozen=True)
class AgentConfig:
    retrieval_top_k: int = 100
    provider_timeout_seconds: float = 30
    enable_memory: bool = True
    enable_semantic_retrieval: bool = True
    structured_planning: bool = True
    required_retrieval_tools: tuple[ToolName, ...] = ("itemcf_retrieve",)
    semantic_profile_history_cap: int = 20


class RecommendationAgent:
    def __init__(
        self,
        *,
        movies: dict[int, Movie],
        itemcf: ItemCFRetriever,
        semantic: TfidfSemanticRetriever,
        ranker: HybridRanker,
        provider: LLMProvider,
        config: AgentConfig | None = None,
    ) -> None:
        self.movies = movies
        self.itemcf = itemcf
        self.semantic = semantic
        self.ranker = ranker
        self.provider = provider
        self.config = config or AgentConfig()

    def recommend(
        self,
        message: str,
        state: PreferenceState | None = None,
    ) -> RecommendationResult:
        started = time.perf_counter()
        current_state = state or PreferenceState()
        errors: list[str] = []
        responses: list[LLMResponse] = []

        if not self.config.structured_planning:
            response = self.provider.chat(
                [{"role": "user", "content": message}],
                response_schema=None,
                timeout=self.config.provider_timeout_seconds,
            )
            responses.append(response)
            if response.error is not None:
                errors.append(_response_error(response, "provider request failed"))
            parsed = (
                PreferencePatch(),
                _fallback_plan(
                    self.config.retrieval_top_k,
                    current_state.requested_count,
                    self.config.required_retrieval_tools,
                ),
            )
            plan_valid = False
        else:
            response = self.provider.chat(
                _planning_messages(
                    message,
                    current_state,
                    self.config.required_retrieval_tools,
                ),
                response_schema=_planning_schema(),
                timeout=self.config.provider_timeout_seconds,
            )
            responses.append(response)
            parsed = _parse_planning_response(
                response,
                self.config.required_retrieval_tools,
            )
            plan_valid = parsed is not None

            if parsed is None:
                errors.append(_response_error(response, "invalid tool plan"))
                repair = self.provider.chat(
                    _repair_messages(
                        message,
                        response,
                        self.config.required_retrieval_tools,
                    ),
                    response_schema=_planning_schema(),
                    timeout=self.config.provider_timeout_seconds,
                )
                responses.append(repair)
                parsed = _parse_planning_response(
                    repair,
                    self.config.required_retrieval_tools,
                )
                plan_valid = parsed is not None
                if parsed is None:
                    errors.append(_response_error(repair, "tool plan repair failed"))

        fallback_used = parsed is None
        if parsed is None:
            patch = PreferencePatch()
            plan = _fallback_plan(
                self.config.retrieval_top_k,
                current_state.requested_count,
                self.config.required_retrieval_tools,
            )
        else:
            patch, plan = parsed

        updated_state = (
            current_state.apply(patch)
            if self.config.enable_memory
            else PreferenceState().apply(patch)
        )
        movies, traces, execution_errors = self._execute(message, updated_state, plan)
        errors.extend(execution_errors)

        return RecommendationResult(
            movies=movies,
            preference_state=updated_state,
            plan=plan,
            traces=traces,
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_calls=len(responses),
            prompt_tokens=sum(response.usage.prompt_tokens for response in responses),
            completion_tokens=sum(response.usage.completion_tokens for response in responses),
            errors=errors,
            fallback_used=fallback_used,
            plan_valid=plan_valid,
        )

    def _execute(
        self,
        message: str,
        state: PreferenceState,
        plan: ToolPlan,
    ) -> tuple[list[Any], list[ToolTrace], list[str]]:
        traces: list[ToolTrace] = []
        errors: list[str] = []
        allowed_movies: dict[int, Movie] = dict(self.movies)
        itemcf_scores: dict[int, float] = {}
        semantic_scores: dict[int, float] = {}
        ranked: list[Any] = []

        for step in plan.steps:
            step_started = time.perf_counter()
            candidate_movie_ids: list[int] = []
            try:
                if step.tool == "lookup":
                    candidate_count = len(allowed_movies)
                elif step.tool == "hard_filter":
                    allowed_movies = {
                        movie.movie_id: movie for movie in hard_filter(self.movies.values(), state)
                    }
                    candidate_count = len(allowed_movies)
                elif step.tool == "itemcf_retrieve":
                    history = state.liked_movie_ids
                    retrieved = self.itemcf.retrieve(
                        history,
                        top_k=_top_k(step, self.config.retrieval_top_k),
                        allowed_ids=set(allowed_movies),
                    )
                    itemcf_scores = dict(retrieved)
                    candidate_movie_ids = [movie_id for movie_id, _ in retrieved]
                    candidate_count = len(itemcf_scores)
                elif step.tool == "semantic_retrieve":
                    if self.config.enable_semantic_retrieval:
                        retrieved = self.semantic.retrieve(
                            build_semantic_profile(
                                message,
                                state,
                                self.movies,
                                history_cap=self.config.semantic_profile_history_cap,
                            ),
                            top_k=_top_k(step, self.config.retrieval_top_k),
                            allowed_ids=set(allowed_movies),
                        )
                        semantic_scores = dict(retrieved)
                        candidate_movie_ids = [
                            movie_id for movie_id, _ in retrieved
                        ]
                    candidate_count = len(semantic_scores)
                elif step.tool == "rerank":
                    ranked = self.ranker.rank(
                        allowed_movies,
                        itemcf_scores=itemcf_scores,
                        semantic_scores=semantic_scores,
                        state=state,
                        top_k=_top_k(step, state.requested_count),
                    )
                    candidate_movie_ids = [movie.movie_id for movie in ranked]
                    candidate_count = len(ranked)
                else:
                    for item in ranked:
                        item.reason = _reason(item, state)
                    candidate_count = len(ranked)
                traces.append(
                    ToolTrace(
                        tool=step.tool,
                        args=step.args,
                        candidate_count=candidate_count,
                        candidate_movie_ids=candidate_movie_ids,
                        latency_ms=(time.perf_counter() - step_started) * 1000,
                    )
                )
            except Exception as exc:  # keep a batch evaluation alive per episode
                message_text = f"{step.tool} failed: {type(exc).__name__}: {exc}"
                errors.append(message_text)
                traces.append(
                    ToolTrace(
                        tool=step.tool,
                        args=step.args,
                        success=False,
                        error=message_text,
                        latency_ms=(time.perf_counter() - step_started) * 1000,
                    )
                )

        if not ranked:
            errors.append("no candidates satisfy hard constraints")
        return ranked, traces, errors


def _parse_planning_response(
    response: LLMResponse,
    required_retrieval_tools: tuple[ToolName, ...],
) -> tuple[PreferencePatch, ToolPlan] | None:
    if response.error is not None or response.structured is None:
        return None
    try:
        patch = PreferencePatch.model_validate(response.structured.get("preference_patch") or {})
        plan = ToolPlan.model_validate({"steps": response.structured.get("steps")})
    except (ValidationError, TypeError):
        return None
    if not _plan_has_required_retrieval(plan, required_retrieval_tools):
        return None
    return patch, plan


def _planning_messages(
    message: str,
    state: PreferenceState,
    required_retrieval_tools: tuple[ToolName, ...],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                'Return one JSON object with keys "preference_patch" and "steps". '
                'Each step is {"tool": <name>, "args": {}}. Allowed tools are '
                "lookup, hard_filter, itemcf_retrieve, semantic_retrieve, rerank, "
                f"and explain. {_plan_safety_instructions(required_retrieval_tools)} "
                "Extract only preferences "
                "stated or implied by the user. Never invent tools or movie IDs."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current preference state: {state.model_dump_json()}\nUser request: {message}"
            ),
        },
    ]


def _repair_messages(
    message: str,
    response: LLMResponse,
    required_retrieval_tools: tuple[ToolName, ...],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the response into one JSON object with preference_patch and "
                f"steps. {_plan_safety_instructions(required_retrieval_tools)} JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"Request: {message}\nInvalid response: {response.text[:1000]}",
        },
    ]


def _planning_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["preference_patch", "steps"],
        "properties": {
            "preference_patch": {"type": "object"},
            "steps": {"type": "array"},
        },
    }


def _plan_safety_instructions(
    required_retrieval_tools: tuple[ToolName, ...],
) -> str:
    required = ", ".join(required_retrieval_tools)
    return (
        "Every recommendation plan MUST include hard_filter before retrieval, "
        "including requests that exclude watched movies. "
        f"It MUST include these configured retrieval tools: {required}. "
        "Then include rerank, and only then explain."
    )


def _fallback_plan(
    retrieval_top_k: int,
    result_top_k: int,
    required_retrieval_tools: tuple[ToolName, ...],
) -> ToolPlan:
    retrieval_steps = [
        ToolStep(tool=tool, args={"top_k": retrieval_top_k})
        for tool in required_retrieval_tools
    ]
    return ToolPlan(
        steps=[
            ToolStep(tool="lookup"),
            ToolStep(tool="hard_filter"),
            *retrieval_steps,
            ToolStep(tool="rerank", args={"top_k": result_top_k}),
            ToolStep(tool="explain"),
        ]
    )


def _plan_has_required_retrieval(
    plan: ToolPlan,
    required_retrieval_tools: tuple[ToolName, ...],
) -> bool:
    names = {step.tool for step in plan.steps}
    return set(required_retrieval_tools).issubset(names)


def _response_error(response: LLMResponse, fallback: str) -> str:
    if response.error is None:
        return fallback
    return f"{response.error.code}: {response.error.message}"


def _top_k(step: ToolStep, default: int) -> int:
    value = step.args.get("top_k", default)
    return max(1, min(int(value), 1000))


def build_semantic_profile(
    message: str,
    state: PreferenceState,
    movies: dict[int, Movie],
    *,
    history_cap: int,
) -> str:
    parts = [message]
    parts.extend(sorted(state.liked_genres))
    for movie_id in sorted(state.liked_movie_ids)[: max(history_cap, 0)]:
        movie = movies.get(movie_id)
        if movie is not None:
            parts.append(movie.text)
    return " ".join(parts)


def _reason(movie: Any, state: PreferenceState) -> str:
    matched = sorted(set(movie.genres) & state.liked_genres)
    if matched:
        return f"Matches preferred genres: {', '.join(matched)}."
    return "Selected by collaborative and semantic relevance."

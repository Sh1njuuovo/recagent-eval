from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolName = Literal[
    "lookup",
    "hard_filter",
    "itemcf_retrieve",
    "semantic_retrieve",
    "rerank",
    "explain",
]


class PreferencePatch(BaseModel):
    liked_movie_ids: set[int] = Field(default_factory=set)
    disliked_movie_ids: set[int] = Field(default_factory=set)
    liked_genres: set[str] = Field(default_factory=set)
    disliked_genres: set[str] = Field(default_factory=set)
    required_genres: set[str] = Field(default_factory=set)
    excluded_genres: set[str] = Field(default_factory=set)
    excluded_movie_ids: set[int] = Field(default_factory=set)
    year_min: int | None = None
    year_max: int | None = None
    requested_count: int | None = Field(default=None, ge=1, le=100)
    ranking_mode: Literal["relevance", "novelty", "balanced"] | None = None
    replace_soft_preferences: bool = False


class PreferenceState(BaseModel):
    liked_movie_ids: set[int] = Field(default_factory=set)
    disliked_movie_ids: set[int] = Field(default_factory=set)
    liked_genres: set[str] = Field(default_factory=set)
    disliked_genres: set[str] = Field(default_factory=set)
    required_genres: set[str] = Field(default_factory=set)
    excluded_genres: set[str] = Field(default_factory=set)
    excluded_movie_ids: set[int] = Field(default_factory=set)
    year_min: int | None = None
    year_max: int | None = None
    requested_count: int = Field(default=10, ge=1, le=100)
    ranking_mode: Literal["relevance", "novelty", "balanced"] = "balanced"

    @model_validator(mode="after")
    def validate_year_range(self) -> PreferenceState:
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError("year_min cannot be greater than year_max")
        return self

    def apply(self, patch: PreferencePatch) -> PreferenceState:
        liked_genres = (
            set(patch.liked_genres)
            if patch.replace_soft_preferences
            else self.liked_genres | patch.liked_genres
        )
        disliked_genres = (
            set(patch.disliked_genres)
            if patch.replace_soft_preferences
            else self.disliked_genres | patch.disliked_genres
        )
        values: dict[str, Any] = {
            "liked_movie_ids": self.liked_movie_ids | patch.liked_movie_ids,
            "disliked_movie_ids": self.disliked_movie_ids | patch.disliked_movie_ids,
            "liked_genres": liked_genres,
            "disliked_genres": disliked_genres,
            "required_genres": self.required_genres | patch.required_genres,
            "excluded_genres": self.excluded_genres | patch.excluded_genres,
            "excluded_movie_ids": self.excluded_movie_ids | patch.excluded_movie_ids,
            "year_min": patch.year_min if patch.year_min is not None else self.year_min,
            "year_max": patch.year_max if patch.year_max is not None else self.year_max,
            "requested_count": (
                patch.requested_count if patch.requested_count is not None else self.requested_count
            ),
            "ranking_mode": patch.ranking_mode or self.ranking_mode,
        }
        return PreferenceState.model_validate(values)


class ToolStep(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    steps: list[ToolStep] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_order(self) -> ToolPlan:
        names = [step.tool for step in self.steps]
        retrieval_tools = {"itemcf_retrieve", "semantic_retrieve"}
        retrieval_indexes = [index for index, name in enumerate(names) if name in retrieval_tools]
        if "rerank" not in names:
            raise ValueError("tool plan requires rerank")
        if not retrieval_indexes:
            raise ValueError("rerank requires at least one retrieval tool")
        if "hard_filter" not in names or names.index("hard_filter") > min(retrieval_indexes):
            raise ValueError("hard_filter must run before retrieval")
        if names.index("rerank") < max(retrieval_indexes):
            raise ValueError("rerank must run after retrieval")
        if "explain" in names and (
            "rerank" not in names or names.index("explain") < names.index("rerank")
        ):
            raise ValueError("explain requires rerank to run first")
        return self


class ScoreBreakdown(BaseModel):
    itemcf: float = 0.0
    semantic: float = 0.0
    preference: float = 0.0
    final: float = 0.0
    feature_contributions: dict[str, float] = Field(default_factory=dict)


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    genres: tuple[str, ...] = ()
    year: int | None = None
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    reason: str = ""


class ToolTrace(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    latency_ms: float = 0.0
    candidate_count: int | None = None
    candidate_movie_ids: list[int] = Field(default_factory=list)
    error: str | None = None


class RecommendationResult(BaseModel):
    movies: list[RecommendedMovie] = Field(default_factory=list)
    preference_state: PreferenceState = Field(default_factory=PreferenceState)
    plan: ToolPlan | None = None
    traces: list[ToolTrace] = Field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    plan_valid: bool = True

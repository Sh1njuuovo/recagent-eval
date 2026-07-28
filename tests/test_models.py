import pytest
from pydantic import ValidationError

from recagent_eval.models import (
    PreferencePatch,
    PreferenceState,
    ToolPlan,
    ToolStep,
)


def test_preference_state_merges_feedback_and_keeps_exclusions_hard() -> None:
    state = PreferenceState(
        liked_movie_ids={1},
        liked_genres={"Drama"},
        excluded_movie_ids={8},
        requested_count=5,
    )

    updated = state.apply(
        PreferencePatch(
            liked_movie_ids={2},
            disliked_movie_ids={3},
            liked_genres={"Sci-Fi"},
            disliked_genres={"Horror"},
            required_genres={"Sci-Fi"},
            excluded_genres={"Musical"},
            excluded_movie_ids={9},
            year_min=1990,
        )
    )

    assert updated.liked_movie_ids == {1, 2}
    assert updated.disliked_movie_ids == {3}
    assert updated.liked_genres == {"Drama", "Sci-Fi"}
    assert updated.disliked_genres == {"Horror"}
    assert updated.required_genres == {"Sci-Fi"}
    assert updated.excluded_genres == {"Musical"}
    assert updated.excluded_movie_ids == {8, 9}
    assert updated.year_min == 1990
    assert updated.requested_count == 5


def test_preference_patch_can_replace_soft_preferences_without_clearing_exclusions() -> None:
    state = PreferenceState(
        liked_genres={"Comedy"},
        excluded_movie_ids={10},
    )

    updated = state.apply(
        PreferencePatch(
            liked_genres={"Thriller"},
            replace_soft_preferences=True,
        )
    )

    assert updated.liked_genres == {"Thriller"}
    assert updated.excluded_movie_ids == {10}


def test_tool_plan_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ToolStep.model_validate({"tool": "shell", "args": {}})


def test_tool_plan_requires_rerank_before_explain() -> None:
    with pytest.raises(ValueError, match="rerank"):
        ToolPlan(
            steps=[
                ToolStep(tool="lookup", args={}),
                ToolStep(tool="explain", args={}),
            ]
        )


def test_tool_plan_accepts_supported_recommendation_pipeline() -> None:
    plan = ToolPlan(
        steps=[
            ToolStep(tool="lookup", args={}),
            ToolStep(tool="hard_filter", args={}),
            ToolStep(tool="itemcf_retrieve", args={"top_k": 100}),
            ToolStep(tool="semantic_retrieve", args={"top_k": 100}),
            ToolStep(tool="rerank", args={"top_k": 10}),
            ToolStep(tool="explain", args={}),
        ]
    )

    assert plan.steps[-1].tool == "explain"


def test_tool_plan_rejects_retrieval_before_hard_filter() -> None:
    with pytest.raises(ValueError, match="hard_filter"):
        ToolPlan(
            steps=[
                ToolStep(tool="itemcf_retrieve"),
                ToolStep(tool="rerank"),
            ]
        )

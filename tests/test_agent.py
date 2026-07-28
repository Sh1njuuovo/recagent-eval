from collections import deque

from recagent_eval.agent import (
    AgentConfig,
    RecommendationAgent,
    build_semantic_profile,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState
from recagent_eval.provider import LLMResponse, ProviderError, TokenUsage
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever


class SequenceProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = deque(responses)

    def chat(self, messages, response_schema=None, timeout=30) -> LLMResponse:
        return self.responses.popleft()


class CapturingProvider(SequenceProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.schemas = []
        self.messages = []

    def chat(self, messages, response_schema=None, timeout=30) -> LLMResponse:
        self.schemas.append(response_schema)
        self.messages.append(messages)
        return super().chat(messages, response_schema, timeout)


MOVIES = {
    1: Movie(1, "Space Quest (1999)", ("Sci-Fi", "Adventure"), 1999),
    2: Movie(2, "Galaxy War (2001)", ("Sci-Fi", "Action"), 2001),
    3: Movie(3, "Quiet Drama (2000)", ("Drama",), 2000),
}
RATINGS = [
    Rating(1, 1, 5, 1),
    Rating(1, 2, 5, 2),
    Rating(2, 1, 5, 1),
    Rating(2, 2, 4, 2),
]


def make_agent(provider: SequenceProvider) -> RecommendationAgent:
    return RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((0.5, 0.3, 0.2)),
        provider=provider,
        config=AgentConfig(retrieval_top_k=100),
    )


def valid_response() -> LLMResponse:
    return LLMResponse(
        structured={
            "preference_patch": {
                "liked_genres": ["Sci-Fi"],
                "excluded_movie_ids": [3],
                "requested_count": 2,
            },
            "steps": [
                {"tool": "lookup", "args": {}},
                {"tool": "hard_filter", "args": {}},
                {"tool": "itemcf_retrieve", "args": {"top_k": 100}},
                {"tool": "semantic_retrieve", "args": {"top_k": 100}},
                {"tool": "rerank", "args": {"top_k": 2}},
                {"tool": "explain", "args": {}},
            ],
        },
        usage=TokenUsage(10, 5, 15),
    )


def itemcf_only_response() -> LLMResponse:
    return LLMResponse(
        structured={
            "preference_patch": {},
            "steps": [
                {"tool": "hard_filter", "args": {}},
                {"tool": "itemcf_retrieve", "args": {"top_k": 100}},
                {"tool": "rerank", "args": {"top_k": 2}},
                {"tool": "explain", "args": {}},
            ],
        }
    )


def test_full_profile_repairs_plan_missing_semantic_route() -> None:
    provider = CapturingProvider([itemcf_only_response(), valid_response()])
    agent = RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((0.5, 0.3, 0.2)),
        provider=provider,
        config=AgentConfig(
            required_retrieval_tools=(
                "itemcf_retrieve",
                "semantic_retrieve",
            )
        ),
    )

    result = agent.recommend("recommend", PreferenceState(liked_movie_ids={1}))

    assert result.llm_calls == 2
    assert result.plan_valid is True
    assert "semantic_retrieve" in provider.messages[0][0]["content"]


def test_full_profile_fallback_retains_both_required_routes() -> None:
    provider = SequenceProvider([itemcf_only_response(), itemcf_only_response()])
    agent = RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((0.5, 0.3, 0.2)),
        provider=provider,
        config=AgentConfig(
            required_retrieval_tools=(
                "itemcf_retrieve",
                "semantic_retrieve",
            )
        ),
    )

    result = agent.recommend("recommend", PreferenceState(liked_movie_ids={1}))

    assert result.fallback_used is True
    assert result.plan is not None
    assert [step.tool for step in result.plan.steps].count("semantic_retrieve") == 1


def test_semantic_profile_is_ordered_capped_and_omits_negative_tokens() -> None:
    state = PreferenceState(
        liked_movie_ids={3, 1, 2},
        liked_genres={"Sci-Fi"},
        disliked_genres={"Horror"},
        excluded_genres={"Musical"},
    )

    profile = build_semantic_profile(
        "recommend something",
        state,
        MOVIES,
        history_cap=2,
    )

    assert "Space Quest" in profile
    assert "Galaxy War" in profile
    assert "Quiet Drama" not in profile
    assert "Sci-Fi" in profile
    assert "Horror" not in profile
    assert "Musical" not in profile


def test_retrieval_traces_save_ordered_candidate_ids() -> None:
    result = make_agent(SequenceProvider([valid_response()])).recommend(
        "science fiction",
        PreferenceState(liked_movie_ids={1}),
    )

    traces = {trace.tool: trace for trace in result.traces}
    assert traces["itemcf_retrieve"].candidate_movie_ids == [2]
    assert traces["semantic_retrieve"].candidate_movie_ids
    assert traces["rerank"].candidate_movie_ids == [2]


def test_agent_updates_memory_executes_tools_and_respects_exclusions() -> None:
    agent = make_agent(SequenceProvider([valid_response()]))

    result = agent.recommend(
        "I liked Space Quest and want more science fiction",
        PreferenceState(liked_movie_ids={1}),
    )

    assert result.plan_valid is True
    assert result.llm_calls == 1
    assert result.preference_state.liked_genres == {"Sci-Fi"}
    assert result.preference_state.excluded_movie_ids == {3}
    assert [movie.movie_id for movie in result.movies] == [2]
    assert all(trace.success for trace in result.traces)
    assert result.movies[0].reason


def test_agent_repairs_once_then_uses_deterministic_fallback() -> None:
    invalid = LLMResponse(
        structured={"steps": [{"tool": "shell", "args": {}}]},
        error=None,
    )
    provider_error = LLMResponse(
        error=ProviderError("invalid_json", "not json"),
    )
    agent = make_agent(SequenceProvider([invalid, provider_error]))

    result = agent.recommend("recommend a movie", PreferenceState(liked_movie_ids={1}))

    assert result.llm_calls == 2
    assert result.fallback_used is True
    assert result.plan_valid is False
    assert result.movies[0].movie_id == 2
    assert len(result.errors) == 2


def test_successful_repair_counts_as_valid_without_fallback() -> None:
    invalid = LLMResponse(
        structured={"steps": [{"tool": "shell", "args": {}}]},
    )
    agent = make_agent(SequenceProvider([invalid, valid_response()]))

    result = agent.recommend("recommend a movie", PreferenceState(liked_movie_ids={1}))

    assert result.llm_calls == 2
    assert result.plan_valid is True
    assert result.fallback_used is False
    assert len(result.errors) == 1


def test_agent_returns_empty_result_instead_of_breaking_hard_constraints() -> None:
    response = valid_response()
    agent = make_agent(SequenceProvider([response]))
    state = PreferenceState(
        liked_movie_ids={1},
        excluded_movie_ids={2, 3},
    )

    result = agent.recommend("anything", state)

    assert result.movies == []
    assert "no candidates satisfy hard constraints" in result.errors


def test_unstructured_baseline_uses_free_text_without_memory_or_repair() -> None:
    provider = CapturingProvider([LLMResponse(text="Try Galaxy War.")])
    agent = RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((1.0, 0.0, 0.0)),
        provider=provider,
        config=AgentConfig(
            structured_planning=False,
            enable_memory=False,
            enable_semantic_retrieval=False,
        ),
    )

    result = agent.recommend(
        "I like science fiction",
        PreferenceState(liked_movie_ids={1}, liked_genres={"Sci-Fi"}),
    )

    assert provider.schemas == [None]
    assert result.llm_calls == 1
    assert result.plan_valid is False
    assert result.fallback_used is False
    assert result.preference_state == PreferenceState()


def test_structured_prompt_names_the_schema_and_tool_allowlist() -> None:
    provider = CapturingProvider([valid_response()])
    agent = make_agent(provider)

    agent.recommend("recommend movies", PreferenceState(liked_movie_ids={1}))

    system_prompt = provider.messages[0][0]["content"]
    assert "preference_patch" in system_prompt
    assert '"steps"' in system_prompt
    assert "itemcf_retrieve" in system_prompt
    assert "semantic_retrieve" in system_prompt
    assert "hard_filter" in system_prompt
    assert "MUST include hard_filter" in system_prompt


def test_repair_prompt_repeats_the_safety_order_constraint() -> None:
    invalid = LLMResponse(
        structured={
            "preference_patch": {},
            "steps": [
                {"tool": "itemcf_retrieve", "args": {}},
                {"tool": "rerank", "args": {}},
            ],
        },
        text='{"preference_patch":{},"steps":[]}',
    )
    provider = CapturingProvider([invalid, valid_response()])
    agent = make_agent(provider)

    agent.recommend("exclude watched movies", PreferenceState(liked_movie_ids={1}))

    repair_prompt = provider.messages[1][0]["content"]
    assert "MUST include hard_filter" in repair_prompt
    assert "hard_filter before retrieval" in repair_prompt

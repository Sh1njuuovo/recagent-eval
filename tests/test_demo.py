import builtins
import importlib

from recagent_eval import demo
from recagent_eval.demo import (
    DemoSessionState,
    build_agent,
    format_recommendations,
    handle_message,
    reset_session,
    select_demo_provider,
    serialize_diagnostics,
)
from recagent_eval.learned_ranking import LearnedRanker
from recagent_eval.models import (
    PreferenceState,
    RecommendationResult,
    RecommendedMovie,
    ScoreBreakdown,
    ToolPlan,
    ToolTrace,
)
from recagent_eval.provider import ProviderStatus, RuleBasedProvider


def test_demo_formatter_shows_scores_reasons_and_fallback_state() -> None:
    result = RecommendationResult(
        movies=[
            RecommendedMovie(
                movie_id=2,
                title="Galaxy War",
                genres=("Sci-Fi",),
                score=ScoreBreakdown(final=0.876),
                reason="Matches preferred genres: Sci-Fi.",
            )
        ],
        fallback_used=True,
    )

    text = format_recommendations(result)

    assert "Galaxy War" in text
    assert "0.876" in text
    assert "Sci-Fi" in text
    assert "deterministic fallback" in text


class SessionAwareAgent:
    def __init__(self) -> None:
        self.calls = 0

    def recommend(self, message: str, state: PreferenceState) -> RecommendationResult:
        self.calls += 1
        genre = "Sci-Fi" if "space" in message.lower() else "Drama"
        updated = state.model_copy(
            update={"liked_genres": state.liked_genres | {genre}}
        )
        return RecommendationResult(
            movies=[
                RecommendedMovie(
                    movie_id=2,
                    title=f"{genre} Pick",
                    genres=(genre,),
                    score=ScoreBreakdown(
                        final=0.8,
                        itemcf=0.5,
                        semantic=0.4,
                        feature_contributions={"genre_match": 0.4, "bias": 0.1},
                    ),
                    reason=f"Matches {genre}.",
                )
            ],
            preference_state=updated,
            plan=ToolPlan.model_validate(
                {
                    "steps": [
                        {"tool": "hard_filter"},
                        {"tool": "itemcf_retrieve"},
                        {"tool": "rerank"},
                        {"tool": "explain"},
                    ]
                }
            ),
            traces=[
                ToolTrace(tool="itemcf_retrieve", candidate_movie_ids=[2]),
                ToolTrace(tool="rerank", candidate_movie_ids=[2]),
            ],
            latency_ms=12.5,
        )


def _status() -> ProviderStatus:
    return ProviderStatus("rule-based", "rule-based", "deterministic-offline")


def test_demo_sessions_remain_isolated_when_interleaved_and_reset() -> None:
    agent = SessionAwareAgent()

    first_a = handle_message("space movies", None, agent=agent, provider_status=_status())
    first_b = handle_message("drama movies", None, agent=agent, provider_status=_status())
    second_a = handle_message(
        "space follow-up",
        first_a.session.model_dump(mode="json"),
        agent=agent,
        provider_status=_status(),
    )

    assert first_a.session.preference_state.liked_genres == {"Sci-Fi"}
    assert first_b.session.preference_state.liked_genres == {"Drama"}
    assert second_a.session.preference_state.liked_genres == {"Sci-Fi"}
    assert len(second_a.session.history) == 4
    assert len(first_b.session.history) == 2
    assert reset_session() == DemoSessionState()


def test_empty_demo_input_is_actionable_and_does_not_call_agent() -> None:
    agent = SessionAwareAgent()

    turn = handle_message("   ", None, agent=agent, provider_status=_status())

    assert "enter" in turn.output.lower()
    assert turn.diagnostics["errors"]
    assert turn.session == DemoSessionState()
    assert agent.calls == 0


def test_diagnostics_include_plan_trace_sources_scores_and_provider() -> None:
    result = SessionAwareAgent().recommend("space", PreferenceState())

    diagnostics = serialize_diagnostics(result, _status())

    assert diagnostics["preference_state"]["liked_genres"] == ["Sci-Fi"]
    assert diagnostics["validated_tool_plan"]["steps"][-1]["tool"] == "explain"
    assert diagnostics["tool_traces"][0]["tool"] == "itemcf_retrieve"
    recommendation = diagnostics["recommendations"][0]
    assert recommendation["candidate_sources"] == ["itemcf_retrieve", "rerank"]
    assert recommendation["score_breakdown"]["final"] == 0.8
    assert recommendation["top_feature_contributions"][0] == {
        "feature": "genre_match",
        "contribution": 0.4,
    }
    assert diagnostics["provider"]["active"] == "rule-based"
    assert diagnostics["latency_ms"] == 12.5
    assert diagnostics["fallback_used"] is False
    assert diagnostics["errors"] == []


def test_no_key_demo_provider_uses_visible_offline_fallback() -> None:
    selection = select_demo_provider("deepseek", environ={})

    assert isinstance(selection.provider, RuleBasedProvider)
    assert selection.status.fallback is True
    assert "DEEPSEEK_API_KEY" in selection.status.message


def test_no_key_demo_agent_builds_offline_from_local_data(tmp_path, monkeypatch) -> None:
    (tmp_path / "movies.dat").write_text(
        "1::Space One (2000)::Sci-Fi\n2::Drama One (2001)::Drama\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "1::1::5::1\n1::2::4::2\n2::1::5::1\n2::2::4::2\n",
        encoding="latin-1",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    agent = build_agent(tmp_path, provider_name="deepseek")

    assert isinstance(agent.provider, RuleBasedProvider)


def test_demo_ranker_config_loads_validated_lambdamart_for_explanations(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "movies.dat").write_text(
        "1::Space One (2000)::Sci-Fi\n2::Drama One (2001)::Drama\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "1::1::5::1\n1::2::4::2\n2::1::5::1\n2::2::4::2\n",
        encoding="latin-1",
    )
    model_path = tmp_path / "ranker.json"
    model_path.write_bytes(b"validated-artifact")
    config_path = tmp_path / "ranker.yaml"
    config_path.write_text(
        f"ranker:\n  kind: lambdamart\n  model_path: {model_path}\n",
        encoding="utf-8",
    )
    artifact = object()
    estimator = object()
    monkeypatch.setattr(
        "recagent_eval.demo.parse_ranker_artifact", lambda raw: artifact
    )
    monkeypatch.setattr(
        "recagent_eval.demo.estimator_from_artifact", lambda value: estimator
    )

    agent = build_agent(
        tmp_path,
        provider_name="rule-based",
        ranker_config_path=config_path,
    )

    assert isinstance(agent.ranker, LearnedRanker)
    assert agent.ranker.estimator is estimator


def test_demo_module_import_does_not_require_gradio(monkeypatch) -> None:
    real_import = builtins.__import__

    def without_gradio(name, *args, **kwargs):
        if name == "gradio":
            raise ImportError("optional dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gradio)
    importlib.reload(demo)

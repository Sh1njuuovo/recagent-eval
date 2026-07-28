from __future__ import annotations

import os
from pathlib import Path

from recagent_eval.agent import RecommendationAgent
from recagent_eval.data import (
    chronological_split,
    load_movielens_movies,
    load_movielens_ratings,
)
from recagent_eval.models import PreferenceState, RecommendationResult
from recagent_eval.provider import OpenAICompatibleProvider
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever


def format_recommendations(result: RecommendationResult) -> str:
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


def build_agent(data_dir: Path) -> RecommendationAgent:
    movies = load_movielens_movies(data_dir / "movies.dat")
    ratings = load_movielens_ratings(data_dir / "ratings.dat")
    split = chronological_split(ratings)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for the interactive demo")
    provider = OpenAICompatibleProvider(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=key,
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    return RecommendationAgent(
        movies=movies,
        itemcf=ItemCFRetriever.fit(split.train),
        semantic=TfidfSemanticRetriever.fit(movies),
        ranker=HybridRanker((0.7, 0.3, 0.0)),
        provider=provider,
    )


def launch(data_dir: Path = Path("data/raw/ml-1m")) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install demo dependencies with: uv sync --extra demo") from exc

    agent = build_agent(data_dir)
    state = PreferenceState()

    def respond(message: str, history: list[dict[str, str]]) -> str:
        del history
        nonlocal state
        result = agent.recommend(message, state)
        state = result.preference_state
        return format_recommendations(result)

    gr.ChatInterface(
        fn=respond,
        title="RecAgent-Eval",
        description="Structured planning, persistent preferences, and measurable ranking.",
    ).launch()


if __name__ == "__main__":
    launch()

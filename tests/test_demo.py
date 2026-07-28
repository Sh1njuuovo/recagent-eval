from recagent_eval.demo import format_recommendations
from recagent_eval.models import (
    RecommendationResult,
    RecommendedMovie,
    ScoreBreakdown,
)


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

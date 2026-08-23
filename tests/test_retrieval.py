import pytest

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState
from recagent_eval.retrieval import (
    ItemCFRetriever,
    TfidfSemanticRetriever,
    hard_filter,
)

MOVIES = {
    1: Movie(1, "Space Quest (1999)", ("Sci-Fi", "Adventure"), 1999),
    2: Movie(2, "Galaxy War (2001)", ("Sci-Fi", "Action"), 2001),
    3: Movie(3, "Quiet Drama (2000)", ("Drama",), 2000),
    4: Movie(4, "Space Comedy (1980)", ("Sci-Fi", "Comedy"), 1980),
}


def test_hard_filter_never_returns_excluded_items_or_genres() -> None:
    state = PreferenceState(
        required_genres={"Sci-Fi"},
        excluded_genres={"Comedy"},
        excluded_movie_ids={2},
        year_min=1990,
    )

    kept = hard_filter(MOVIES.values(), state)

    assert [movie.movie_id for movie in kept] == [1]


def test_itemcf_retrieves_items_co_liked_with_history() -> None:
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(2, 1, 4, 1),
        Rating(2, 2, 5, 2),
        Rating(3, 1, 5, 1),
        Rating(3, 3, 5, 2),
    ]
    retriever = ItemCFRetriever.fit(ratings)

    scores = retriever.retrieve({1}, top_k=10)

    assert scores[0][0] == 2
    assert {movie_id for movie_id, _ in scores} == {2, 3}

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve({1}, top_k=0)


def test_itemcf_score_many_scores_requested_ids_and_falls_back_to_popularity() -> None:
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(2, 1, 4, 1),
        Rating(2, 2, 5, 2),
        Rating(3, 1, 5, 1),
        Rating(3, 3, 5, 2),
    ]
    retriever = ItemCFRetriever.fit(ratings)
    scores = retriever.score_many({1}, [2, 3, 99])
    assert scores[2] > 0.0
    assert scores[99] == 0.0
    empty = retriever.score_many(set(), [1, 2])
    assert empty[1] > 0.0  # popularity fallback for empty history
    assert empty[2] > 0.0


def test_tfidf_retrieval_uses_title_and_genre_text() -> None:
    retriever = TfidfSemanticRetriever.fit(MOVIES)

    scores = retriever.retrieve("space science fiction adventure", top_k=2)

    assert scores[0][0] == 1
    assert 3 not in {movie_id for movie_id, _ in scores}

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("space", top_k=0)

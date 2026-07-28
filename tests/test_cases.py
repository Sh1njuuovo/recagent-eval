import pytest

from recagent_eval.cases import (
    EvaluationCase,
    generate_cases,
    select_stratified_cases,
    validate_case_relevance,
)
from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.models import PreferenceState


def test_generate_cases_is_deterministic_and_respects_requested_mix() -> None:
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id} (2000)", ("Drama",), 2000)
        for movie_id in range(1, 13)
    }
    movies[13] = Movie(13, "Action Catalog (2000)", ("Action",), 2000)
    ratings = []
    for user_id in range(1, 4):
        for offset in range(4):
            movie_id = (user_id - 1) * 4 + offset + 1
            ratings.append(Rating(user_id, movie_id, 5, offset + 1))
    split = chronological_split(ratings)

    cases = generate_cases(
        movies,
        split,
        ratings,
        single_turn_count=2,
        multi_turn_count=1,
        seed=7,
    )

    assert [case.case_id for case in cases] == [
        "single-001",
        "single-002",
        "multi-001",
    ]
    assert len(cases[0].turns) == 1
    assert cases[0].turns[0].startswith("I tend to enjoy ")
    assert "matching" not in cases[0].turns[0]
    assert len(cases[-1].turns) == 3
    assert cases[-1].relevant_movie_ids
    assert cases[-1].expected_preferences is not None
    assert cases[-1].expected_preferences.excluded_genres
    assert not cases[-1].expected_preferences.disliked_genres


def test_multi_turn_negative_genre_does_not_conflict_with_target() -> None:
    movies = {
        1: Movie(1, "History", ("Comedy",), 1998),
        2: Movie(2, "Validation", ("Drama",), 1999),
        3: Movie(3, "Target", ("Comedy", "Romance"), 2000),
        4: Movie(4, "Catalog Action", ("Action",), 2001),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(1, 3, 5, 3),
    ]
    split = chronological_split(ratings)

    case = generate_cases(
        movies,
        split,
        ratings,
        single_turn_count=0,
        multi_turn_count=1,
        seed=7,
    )[0]

    assert case.expected_preferences is not None
    excluded = case.expected_preferences.excluded_genres
    assert excluded
    assert not (excluded & set(movies[3].genres))
    validate_case_relevance(case, movies)


def test_case_preflight_reports_conflicting_relevance_target() -> None:
    movies = {9: Movie(9, "Target", ("Action",), 2000)}
    case = EvaluationCase(
        case_id="bad-case",
        user_id=1,
        turns=("Avoid Action.",),
        relevant_movie_ids={9},
        initial_state=PreferenceState(excluded_genres={"Action"}),
    )

    with pytest.raises(
        ValueError,
        match=r"bad-case.*movie 9.*excluded genre Action",
    ):
        validate_case_relevance(case, movies)


def test_select_stratified_cases_keeps_single_and_multi_mix() -> None:
    cases = [
        EvaluationCase(
            case_id=case_id,
            user_id=index,
            turns=("turn",),
            relevant_movie_ids={index},
            initial_state=PreferenceState(),
        )
        for index, case_id in enumerate(
            ["single-001", "single-002", "multi-001", "multi-002"],
            start=1,
        )
    ]

    selected = select_stratified_cases(cases, single_turn_count=1, multi_turn_count=1)

    assert [case.case_id for case in selected] == ["single-001", "multi-001"]

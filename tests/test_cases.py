from recagent_eval.cases import EvaluationCase, generate_cases, select_stratified_cases
from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.models import PreferenceState


def test_generate_cases_is_deterministic_and_respects_requested_mix() -> None:
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id} (2000)", ("Drama",), 2000)
        for movie_id in range(1, 13)
    }
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
    assert len(cases[-1].turns) == 3
    assert cases[-1].relevant_movie_ids


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

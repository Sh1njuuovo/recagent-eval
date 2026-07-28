from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from recagent_eval.data import DatasetSplit, Movie, Rating
from recagent_eval.models import PreferenceState


class EvaluationCase(BaseModel):
    case_id: str
    user_id: int
    turns: tuple[str, ...] = Field(min_length=1)
    relevant_movie_ids: set[int] = Field(min_length=1)
    initial_state: PreferenceState = Field(default_factory=PreferenceState)
    expected_preferences: PreferenceState | None = None
    tags: tuple[str, ...] = ()


def generate_cases(
    movies: dict[int, Movie],
    split: DatasetSplit,
    ratings: list[Rating],
    *,
    single_turn_count: int = 40,
    multi_turn_count: int = 10,
    seed: int = 42,
) -> list[EvaluationCase]:
    positive_history: dict[int, list[int]] = defaultdict(list)
    for row in split.train:
        if row.rating >= 4 and row.movie_id in movies:
            positive_history[row.user_id].append(row.movie_id)
    eligible = sorted(
        set(split.test_targets)
        & {user_id for user_id, history in positive_history.items() if history}
    )
    random.Random(seed).shuffle(eligible)
    single_users = (
        [eligible[index % len(eligible)] for index in range(single_turn_count)]
        if eligible
        else []
    )

    cases: list[EvaluationCase] = []
    for index, user_id in enumerate(single_users, start=1):
        history = positive_history[user_id]
        genres = _favorite_genres(history, movies)
        state = PreferenceState(
            liked_movie_ids=set(history),
            liked_genres=set(genres[:2]),
            requested_count=10,
        )
        genre_text = ", ".join(genres[:2]) or "movies similar to my history"
        cases.append(
            EvaluationCase(
                case_id=f"single-{index:03d}",
                user_id=user_id,
                turns=(
                    f"I tend to enjoy {genre_text}. "
                    "Please recommend ten movies based on those preferences.",
                ),
                relevant_movie_ids={split.test_targets[user_id]},
                initial_state=state,
                tags=("single-turn", "history-aware"),
            )
        )

    split_index = min(single_turn_count, len(eligible))
    multi_user_order = eligible[split_index:] + eligible[:split_index]
    multi_candidates: list[tuple[int, str]] = []
    for user_id in multi_user_order:
        history = positive_history[user_id]
        genres = _favorite_genres(history, movies)
        liked_genre = genres[0] if genres else "Drama"
        target = movies[split.test_targets[user_id]]
        disliked_genre = _different_genre(
            liked_genre,
            movies,
            forbidden_genres=set(target.genres),
        )
        if disliked_genre is not None:
            multi_candidates.append((user_id, disliked_genre))

    multi_users = (
        [
            multi_candidates[index % len(multi_candidates)]
            for index in range(multi_turn_count)
        ]
        if multi_candidates
        else []
    )
    for index, (user_id, disliked_genre) in enumerate(multi_users, start=1):
        history = positive_history[user_id]
        genres = _favorite_genres(history, movies)
        liked_genre = genres[0] if genres else "Drama"
        initial = PreferenceState(
            liked_movie_ids=set(history),
            requested_count=10,
        )
        expected = initial.apply(_genre_patch(liked_genre, disliked_genre))
        cases.append(
            EvaluationCase(
                case_id=f"multi-{index:03d}",
                user_id=user_id,
                turns=(
                    f"I usually like {liked_genre} movies.",
                    f"Please avoid {disliked_genre}.",
                    "Now give me ten recommendations and exclude movies I have seen.",
                ),
                relevant_movie_ids={split.test_targets[user_id]},
                initial_state=initial,
                expected_preferences=expected,
                tags=("multi-turn", "preference-retention", "negative-feedback"),
            )
        )
    validate_cases_relevance(cases, movies)
    return cases


def save_cases(cases: list[EvaluationCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [case.model_dump(mode="json") for case in cases]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text())
    return [EvaluationCase.model_validate(item) for item in payload]


def select_stratified_cases(
    cases: list[EvaluationCase],
    *,
    single_turn_count: int,
    multi_turn_count: int,
) -> list[EvaluationCase]:
    single = [case for case in cases if case.case_id.startswith("single-")]
    multi = [case for case in cases if case.case_id.startswith("multi-")]
    if len(single) < single_turn_count or len(multi) < multi_turn_count:
        raise ValueError("not enough cases for requested stratified subset")
    return single[:single_turn_count] + multi[:multi_turn_count]


def _favorite_genres(history: list[int], movies: dict[int, Movie]) -> list[str]:
    counts: Counter[str] = Counter()
    for movie_id in history:
        movie = movies.get(movie_id)
        if movie:
            counts.update(movie.genres)
    return [genre for genre, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _different_genre(
    liked_genre: str,
    movies: dict[int, Movie],
    *,
    forbidden_genres: set[str],
) -> str | None:
    genres = sorted({genre for movie in movies.values() for genre in movie.genres})
    return next(
        (
            genre
            for genre in genres
            if genre != liked_genre and genre not in forbidden_genres
        ),
        None,
    )


def _genre_patch(liked: str, disliked: str):
    from recagent_eval.models import PreferencePatch

    return PreferencePatch(liked_genres={liked}, excluded_genres={disliked})


def validate_case_relevance(
    case: EvaluationCase,
    movies: dict[int, Movie],
) -> None:
    state = case.expected_preferences or case.initial_state
    blocked_ids = (
        state.liked_movie_ids
        | state.disliked_movie_ids
        | state.excluded_movie_ids
    )
    for movie_id in sorted(case.relevant_movie_ids):
        movie = movies.get(movie_id)
        if movie is None:
            raise ValueError(
                f"{case.case_id}: relevant movie {movie_id} is missing from metadata"
            )
        reasons: list[str] = []
        if movie_id in blocked_ids:
            reasons.append("blocked movie ID")
        genres = set(movie.genres)
        excluded = sorted(state.excluded_genres & genres)
        if excluded:
            reasons.append(f"excluded genre {', '.join(excluded)}")
        if state.required_genres and not state.required_genres.issubset(genres):
            reasons.append("missing required genre")
        if state.year_min is not None and (
            movie.year is None or movie.year < state.year_min
        ):
            reasons.append("below minimum year")
        if state.year_max is not None and (
            movie.year is None or movie.year > state.year_max
        ):
            reasons.append("above maximum year")
        if reasons:
            raise ValueError(
                f"{case.case_id}: relevant movie {movie_id} violates "
                + "; ".join(reasons)
            )


def validate_cases_relevance(
    cases: list[EvaluationCase],
    movies: dict[int, Movie],
) -> None:
    for case in cases:
        validate_case_relevance(case, movies)

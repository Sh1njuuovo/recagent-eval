from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState

FEATURE_SCHEMA_VERSION = "candidate-features/v1"
FEATURE_NAMES = (
    "itemcf_score",
    "itemcf_reciprocal_rank",
    "dense_score",
    "dense_reciprocal_rank",
    "log1p_popularity",
    "history_genre_jaccard",
    "history_year_match",
    "preference_affinity",
    "in_itemcf",
    "in_dense",
)
FEATURE_SCHEMA_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {"version": FEATURE_SCHEMA_VERSION, "features": FEATURE_NAMES},
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True)
class CandidateFeatureRow:
    user_id: int
    movie_id: int
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))


def build_candidate_feature_rows(
    *,
    user_id: int,
    movies: Mapping[int, Movie],
    candidate_ids: Iterable[int] | None = None,
    itemcf_scores: Mapping[int, float],
    dense_scores: Mapping[int, float],
    history: Iterable[Rating],
    train_rows: Iterable[Rating],
    state: PreferenceState,
) -> tuple[CandidateFeatureRow, ...]:
    """Build fixed-order features exclusively from explicitly legal inputs."""
    candidates = (
        set(itemcf_scores) | set(dense_scores) if candidate_ids is None else set(candidate_ids)
    )
    history_rows = tuple(history)
    statistics_rows = tuple(train_rows)
    popularity = Counter(row.movie_id for row in statistics_rows if row.rating >= 4)
    history_movies = [movies[row.movie_id] for row in history_rows if row.movie_id in movies]
    history_genres = {genre for movie in history_movies for genre in movie.genres}
    history_years = {movie.year for movie in history_movies if movie.year is not None}
    itemcf_ranks = _ranks(itemcf_scores)
    dense_ranks = _ranks(dense_scores)

    result: list[CandidateFeatureRow] = []
    for movie_id in sorted(candidates):
        movie = movies.get(movie_id)
        if movie is None:
            continue
        movie_genres = set(movie.genres)
        union = history_genres | movie_genres
        genre_jaccard = len(history_genres & movie_genres) / len(union) if union else 0.0
        year_match = float(movie.year is not None and movie.year in history_years)
        values = (
            float(itemcf_scores.get(movie_id, 0.0)),
            1.0 / itemcf_ranks[movie_id] if movie_id in itemcf_ranks else 0.0,
            float(dense_scores.get(movie_id, 0.0)),
            1.0 / dense_ranks[movie_id] if movie_id in dense_ranks else 0.0,
            math.log1p(popularity[movie_id]),
            genre_jaccard,
            year_match,
            _preference_affinity(movie, state),
            float(movie_id in itemcf_scores),
            float(movie_id in dense_scores),
        )
        for name, value in zip(FEATURE_NAMES, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    "candidate feature must be finite: "
                    f"user={user_id}, movie={movie_id}, feature={name}, value={value!r}"
                )
        result.append(CandidateFeatureRow(user_id, movie_id, values))
    return tuple(result)


def _ranks(scores: Mapping[int, float]) -> dict[int, int]:
    for _movie_id, value in scores.items():
        if not math.isfinite(value):
            # The caller adds full row context when materializing the feature.
            continue
    return {
        movie_id: rank
        for rank, movie_id in enumerate(
            sorted(scores, key=lambda item: (-scores[item], item)), start=1
        )
    }


def _preference_affinity(movie: Movie, state: PreferenceState) -> float:
    genres = set(movie.genres)
    liked = len(genres & state.liked_genres)
    disliked = len(genres & state.disliked_genres)
    liked_movie = float(movie.movie_id in state.liked_movie_ids)
    disliked_movie = float(movie.movie_id in state.disliked_movie_ids)
    score = 0.5 + 0.25 * liked - 0.5 * disliked + 0.5 * liked_movie - disliked_movie
    return max(0.0, min(1.0, score))

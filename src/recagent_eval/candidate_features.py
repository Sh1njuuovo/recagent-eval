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

FEATURE_SCHEMA_VERSION_V2 = "candidate-features/v2"
FEATURE_NAMES_V2 = FEATURE_NAMES + (
    "latent_score",
    "latent_reciprocal_rank",
    "in_latent",
)
FEATURE_SCHEMA_FINGERPRINT_V2 = hashlib.sha256(
    json.dumps(
        {"version": FEATURE_SCHEMA_VERSION_V2, "features": FEATURE_NAMES_V2},
        separators=(",", ":"),
    ).encode()
).hexdigest()

FEATURE_SCHEMA_VERSION_V2B = "candidate-features/v2b"
FEATURE_NAMES_V2B = FEATURE_NAMES_V2 + (
    "itemcf_latent_cross",
    "recent_itemcf_score",
    "year_recency",
)
FEATURE_SCHEMA_FINGERPRINT_V2B = hashlib.sha256(
    json.dumps(
        {"version": FEATURE_SCHEMA_VERSION_V2B, "features": FEATURE_NAMES_V2B},
        separators=(",", ":"),
    ).encode()
).hexdigest()

_SCHEMA_BY_VERSION = {
    "v1": (FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT),
    "v2": (FEATURE_NAMES_V2, FEATURE_SCHEMA_FINGERPRINT_V2),
    "v2b": (FEATURE_NAMES_V2B, FEATURE_SCHEMA_FINGERPRINT_V2B),
}


@dataclass(frozen=True)
class CandidateFeatureRow:
    user_id: int
    movie_id: int
    values: tuple[float, ...]
    names: tuple[str, ...] = FEATURE_NAMES

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


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
    score_calibration: str = "raw",
    latent_scores: Mapping[int, float] | None = None,
    recent_itemcf_scores: Mapping[int, float] | None = None,
    feature_version: str = "v1",
) -> tuple[CandidateFeatureRow, ...]:
    """Build fixed-order features exclusively from explicitly legal inputs."""
    if score_calibration not in {"raw", "percentile"}:
        raise ValueError("score_calibration must be raw or percentile")
    if feature_version not in _SCHEMA_BY_VERSION:
        raise ValueError("feature_version must be v1, v2, or v2b")
    names, _fingerprint = _SCHEMA_BY_VERSION[feature_version]
    latent = dict(latent_scores or {})
    recent = dict(recent_itemcf_scores or {})
    candidates = (
        set(itemcf_scores) | set(dense_scores) | set(latent)
        if candidate_ids is None
        else set(candidate_ids)
    )
    history_rows = tuple(history)
    statistics_rows = tuple(train_rows)
    popularity = Counter(row.movie_id for row in statistics_rows if row.rating >= 4)
    history_movies = [movies[row.movie_id] for row in history_rows if row.movie_id in movies]
    history_genres = {genre for movie in history_movies for genre in movie.genres}
    history_years = {movie.year for movie in history_movies if movie.year is not None}
    recent_years = [
        movie.year for movie in history_movies if movie.year is not None
    ]
    median_recent_year = (
        sorted(recent_years)[len(recent_years) // 2] if recent_years else None
    )
    itemcf_ranks = _ranks(itemcf_scores)
    dense_ranks = _ranks(dense_scores)
    latent_ranks = _ranks(latent)
    itemcf_score_values = (
        _route_percentile(itemcf_scores)
        if score_calibration == "percentile"
        else itemcf_scores
    )
    dense_score_values = (
        _route_percentile(dense_scores)
        if score_calibration == "percentile"
        else dense_scores
    )
    latent_score_values = (
        _route_percentile(latent)
        if score_calibration == "percentile"
        else latent
    )

    result: list[CandidateFeatureRow] = []
    for movie_id in sorted(candidates):
        movie = movies.get(movie_id)
        if movie is None:
            continue
        movie_genres = set(movie.genres)
        union = history_genres | movie_genres
        genre_jaccard = len(history_genres & movie_genres) / len(union) if union else 0.0
        year_match = float(movie.year is not None and movie.year in history_years)
        itemcf_value = float(itemcf_score_values.get(movie_id, 0.0))
        latent_score = float(latent_score_values.get(movie_id, 0.0))
        values = [
            itemcf_value,
            1.0 / itemcf_ranks[movie_id] if movie_id in itemcf_ranks else 0.0,
            float(dense_score_values.get(movie_id, 0.0)),
            1.0 / dense_ranks[movie_id] if movie_id in dense_ranks else 0.0,
            math.log1p(popularity[movie_id]),
            genre_jaccard,
            year_match,
            _preference_affinity(movie, state),
            float(movie_id in itemcf_scores),
            float(movie_id in dense_scores),
        ]
        if feature_version != "v1":
            values += [
                latent_score,
                1.0 / latent_ranks[movie_id] if movie_id in latent_ranks else 0.0,
                float(movie_id in latent),
            ]
        if feature_version == "v2b":
            year_recency = (
                float(abs(movie.year - median_recent_year))
                if movie.year is not None and median_recent_year is not None
                else 0.0
            )
            values += [
                itemcf_value * latent_score,
                float(recent.get(movie_id, 0.0)),
                year_recency,
            ]
        row_values = tuple(float(value) for value in values)
        if len(row_values) != len(names):
            raise ValueError("candidate feature row length does not match schema")
        for name, value in zip(names, row_values, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    "candidate feature must be finite: "
                    f"user={user_id}, movie={movie_id}, feature={name}, value={value!r}"
                )
        result.append(CandidateFeatureRow(user_id, movie_id, row_values, names))
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


def _route_percentile(scores: Mapping[int, float]) -> dict[int, float]:
    """Map finite route scores to within-route rank percentiles.

    Percentile is ``(rank_count - rank + 1) / rank_count`` with competition
    ranking: equal scores share the same rank, ordered by movie ID for
    determinism. Missing route members stay zero in the caller.
    """
    ordered = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))
    count = len(ordered)
    if count == 0:
        return {}
    result: dict[int, float] = {}
    index = 0
    while index < count:
        end = index
        while (
            end < count
            and scores[ordered[end]] == scores[ordered[index]]
        ):
            end += 1
        percentile = (count - (index + 1) + 1) / count
        for movie_id in ordered[index:end]:
            result[movie_id] = percentile
        index = end
    return result


def _preference_affinity(movie: Movie, state: PreferenceState) -> float:
    genres = set(movie.genres)
    liked = len(genres & state.liked_genres)
    disliked = len(genres & state.disliked_genres)
    liked_movie = float(movie.movie_id in state.liked_movie_ids)
    disliked_movie = float(movie.movie_id in state.disliked_movie_ids)
    score = 0.5 + 0.25 * liked - 0.5 * disliked + 0.5 * liked_movie - disliked_movie
    return max(0.0, min(1.0, score))

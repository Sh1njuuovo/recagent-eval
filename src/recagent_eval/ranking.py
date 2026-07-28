from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from recagent_eval.data import Movie
from recagent_eval.models import (
    PreferenceState,
    RecommendedMovie,
    ScoreBreakdown,
)


@dataclass(frozen=True)
class HybridRanker:
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)

    def rank(
        self,
        movies: dict[int, Movie],
        *,
        itemcf_scores: dict[int, float],
        semantic_scores: dict[int, float],
        state: PreferenceState,
        top_k: int = 10,
    ) -> list[RecommendedMovie]:
        candidate_ids = set(itemcf_scores) | set(semantic_scores)
        normalized_cf = normalize_scores(itemcf_scores)
        normalized_semantic = normalize_scores(semantic_scores)
        preference_scores = {
            movie_id: _preference_affinity(movies[movie_id], state)
            for movie_id in candidate_ids
            if movie_id in movies
        }
        w_cf, w_semantic, w_preference = self.weights
        ranked: list[RecommendedMovie] = []
        for movie_id in candidate_ids:
            movie = movies.get(movie_id)
            if movie is None:
                continue
            cf_score = normalized_cf.get(movie_id, 0.0)
            semantic_score = normalized_semantic.get(movie_id, 0.0)
            preference_score = preference_scores.get(movie_id, 0.0)
            final = w_cf * cf_score + w_semantic * semantic_score + w_preference * preference_score
            ranked.append(
                RecommendedMovie(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    year=movie.year,
                    score=ScoreBreakdown(
                        itemcf=cf_score,
                        semantic=semantic_score,
                        preference=preference_score,
                        final=final,
                    ),
                )
            )
        return sorted(
            ranked,
            key=lambda item: (-item.score.final, item.movie_id),
        )[:top_k]


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def tune_weights(
    examples: list[
        tuple[
            dict[int, float],
            dict[int, float],
            dict[int, float],
            set[int],
        ]
    ],
    *,
    step: float = 0.1,
    top_k: int = 10,
) -> tuple[float, float, float]:
    units = round(1 / step)
    candidates = [
        (left * step, middle * step, right * step)
        for left, middle, right in product(range(units + 1), repeat=3)
        if left + middle + right == units
    ]

    def quality(weights: tuple[float, float, float]) -> float:
        values = []
        for itemcf, semantic, preference, relevant in examples:
            ids = set(itemcf) | set(semantic) | set(preference)
            scored = sorted(
                ids,
                key=lambda movie_id: (
                    -(
                        weights[0] * itemcf.get(movie_id, 0.0)
                        + weights[1] * semantic.get(movie_id, 0.0)
                        + weights[2] * preference.get(movie_id, 0.0)
                    ),
                    movie_id,
                ),
            )[:top_k]
            values.append(_ndcg(scored, relevant))
        return sum(values) / len(values) if values else 0.0

    return max(candidates, key=lambda weights: (quality(weights), weights[1], weights[0]))


def _preference_affinity(movie: Movie, state: PreferenceState) -> float:
    genres = set(movie.genres)
    positive = len(genres & state.liked_genres)
    negative = len(genres & state.disliked_genres)
    if positive == 0 and negative == 0:
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.25 * positive - 0.5 * negative))


def _ndcg(ranked_ids: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1 / math.log2(index + 2)
        for index, movie_id in enumerate(ranked_ids)
        if movie_id in relevant
    )
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), len(ranked_ids))))
    return dcg / ideal if ideal else 0.0

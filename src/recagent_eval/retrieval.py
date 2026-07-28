from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState


def hard_filter(
    movies: Iterable[Movie],
    state: PreferenceState,
) -> list[Movie]:
    blocked_ids = state.excluded_movie_ids | state.disliked_movie_ids | state.liked_movie_ids
    kept: list[Movie] = []
    for movie in movies:
        genres = set(movie.genres)
        if movie.movie_id in blocked_ids:
            continue
        if state.year_min is not None and (movie.year is None or movie.year < state.year_min):
            continue
        if state.year_max is not None and (movie.year is None or movie.year > state.year_max):
            continue
        if state.required_genres and not state.required_genres.issubset(genres):
            continue
        if state.excluded_genres & genres:
            continue
        kept.append(movie)
    return kept


@dataclass(frozen=True)
class ItemCFRetriever:
    similarities: dict[int, dict[int, float]]
    popularity: dict[int, int]

    @classmethod
    def fit(
        cls,
        ratings: Iterable[Rating],
        *,
        positive_threshold: int = 4,
    ) -> ItemCFRetriever:
        user_items: dict[int, set[int]] = defaultdict(set)
        popularity: Counter[int] = Counter()
        for row in ratings:
            if row.rating >= positive_threshold:
                user_items[row.user_id].add(row.movie_id)
                popularity[row.movie_id] += 1

        cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
        for items in user_items.values():
            for left in items:
                for right in items:
                    if left != right:
                        cooccurrence[left][right] += 1

        similarities: dict[int, dict[int, float]] = defaultdict(dict)
        for left, neighbors in cooccurrence.items():
            for right, count in neighbors.items():
                denominator = math.sqrt(popularity[left] * popularity[right])
                similarities[left][right] = count / denominator if denominator else 0.0
        return cls(dict(similarities), dict(popularity))

    def retrieve(
        self,
        history: set[int],
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        scores: Counter[int] = Counter()
        for source in history:
            for movie_id, similarity in self.similarities.get(source, {}).items():
                if movie_id not in history:
                    scores[movie_id] += similarity
        if not scores:
            for movie_id, count in self.popularity.items():
                if movie_id not in history:
                    scores[movie_id] = float(count)
        ranked = [
            (movie_id, float(score))
            for movie_id, score in scores.items()
            if allowed_ids is None or movie_id in allowed_ids
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_k]


@dataclass(frozen=True)
class TfidfSemanticRetriever:
    movies: dict[int, Movie]
    vectors: dict[int, dict[str, float]]
    idf: dict[str, float]

    @classmethod
    def fit(cls, movies: dict[int, Movie]) -> TfidfSemanticRetriever:
        documents = {movie_id: _tokens(movie.text) for movie_id, movie in movies.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))
        count = max(len(documents), 1)
        idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        vectors = {movie_id: _tfidf_vector(tokens, idf) for movie_id, tokens in documents.items()}
        return cls(movies=dict(movies), vectors=vectors, idf=idf)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        query_vector = _tfidf_vector(_tokens(query), self.idf)
        scores = []
        for movie_id, vector in self.vectors.items():
            if allowed_ids is not None and movie_id not in allowed_ids:
                continue
            score = sum(query_vector.get(key, 0.0) * value for key, value in vector.items())
            if score > 0:
                scores.append((movie_id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    vector = {token: count * idf[token] for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        return {token: value / norm for token, value in vector.items()}
    return {}

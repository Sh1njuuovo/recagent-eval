from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]
    year: int | None = None

    @property
    def text(self) -> str:
        return f"{self.title} {' '.join(self.genres)}"


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[Rating, ...]
    validation_targets: dict[int, int]
    test_targets: dict[int, int]


@dataclass(frozen=True)
class LeakageSafeRankingSplit:
    """Three-target split used only by the learned-ranking pipeline.

    ``legal_retrieval_train`` may be used to fit retrieval for validation: it
    includes the ranker target and excludes validation/frozen-test targets.
    ``histories`` is the stricter per-query view used to construct ranker rows.
    """

    ranker_training_history: tuple[Rating, ...]
    legal_retrieval_train: tuple[Rating, ...]
    ranker_targets: dict[int, int]
    validation_targets: dict[int, int]
    test_targets: dict[int, int]
    histories: dict[int, tuple[Rating, ...]]
    input_fingerprint: str

    @property
    def train(self) -> tuple[Rating, ...]:
        return self.ranker_training_history


def load_movielens_movies(path: Path) -> dict[int, Movie]:
    movies: dict[int, Movie] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line:
            continue
        movie_id_text, title, genre_text = line.split("::")
        year_match = re.search(r"\((\d{4})\)\s*$", title)
        movies[int(movie_id_text)] = Movie(
            movie_id=int(movie_id_text),
            title=title,
            genres=tuple(genre_text.split("|")),
            year=int(year_match.group(1)) if year_match else None,
        )
    return movies


def load_movielens_ratings(path: Path) -> list[Rating]:
    ratings: list[Rating] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line:
            continue
        user_id, movie_id, rating, timestamp = line.split("::")
        ratings.append(
            Rating(
                user_id=int(user_id),
                movie_id=int(movie_id),
                rating=int(rating),
                timestamp=int(timestamp),
            )
        )
    return ratings


def chronological_split(
    ratings: list[Rating],
    *,
    positive_threshold: int = 4,
) -> DatasetSplit:
    by_user: dict[int, list[Rating]] = defaultdict(list)
    for rating in ratings:
        by_user[rating.user_id].append(rating)

    train: list[Rating] = []
    validation_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    for user_id in sorted(by_user):
        rows = sorted(by_user[user_id], key=lambda item: (item.timestamp, item.movie_id))
        positive = [row for row in rows if row.rating >= positive_threshold]
        if len(positive) < 3:
            train.extend(rows)
            continue
        validation_row, test_row = positive[-2:]
        validation_targets[user_id] = validation_row.movie_id
        test_targets[user_id] = test_row.movie_id
        held_out = {id(validation_row), id(test_row)}
        train.extend(row for row in rows if id(row) not in held_out)
    return DatasetSplit(
        train=tuple(sorted(train, key=lambda item: (item.user_id, item.timestamp))),
        validation_targets=validation_targets,
        test_targets=test_targets,
    )


def leakage_safe_ranking_split(
    ratings: list[Rating] | tuple[Rating, ...],
    *,
    positive_threshold: int = 4,
) -> LeakageSafeRankingSplit:
    """Create deterministic ranker/validation/test targets without changing v1."""
    ordered_input = sorted(
        ratings,
        key=lambda row: (
            row.user_id,
            row.timestamp,
            row.movie_id,
            row.rating,
        ),
    )
    by_user: dict[int, list[Rating]] = defaultdict(list)
    for row in ordered_input:
        by_user[row.user_id].append(row)

    history_rows: list[Rating] = []
    retrieval_rows: list[Rating] = []
    histories: dict[int, tuple[Rating, ...]] = {}
    ranker_targets: dict[int, int] = {}
    validation_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        positives = [row for row in rows if row.rating >= positive_threshold]
        if len(positives) < 4:
            history_rows.extend(rows)
            retrieval_rows.extend(rows)
            continue
        ranker_row, validation_row, test_row = positives[-3:]
        history = tuple(
            row
            for row in rows
            if row.timestamp < ranker_row.timestamp
            or (
                row.timestamp == ranker_row.timestamp
                and (row.movie_id, row.rating) < (ranker_row.movie_id, ranker_row.rating)
            )
        )
        # A row equal by value to a target is intentionally treated as a target;
        # MovieLens has one interaction per user/movie, and this keeps duplicate
        # input handling deterministic rather than dependent on object identity.
        held_out = {ranker_row, validation_row, test_row}
        validation_key = (
            validation_row.timestamp,
            validation_row.movie_id,
            validation_row.rating,
        )
        legal_retrieval = [
            row
            for row in rows
            if (row.timestamp, row.movie_id, row.rating) < validation_key
            and row not in {validation_row, test_row}
        ]
        histories[user_id] = history
        history_rows.extend(history)
        retrieval_rows.extend(legal_retrieval)
        ranker_targets[user_id] = ranker_row.movie_id
        validation_targets[user_id] = validation_row.movie_id
        test_targets[user_id] = test_row.movie_id
        if len(held_out) != 3:
            raise ValueError(f"ranking targets are not disjoint for user={user_id}")

    payload = [[row.user_id, row.movie_id, row.rating, row.timestamp] for row in ordered_input]
    fingerprint = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return LeakageSafeRankingSplit(
        ranker_training_history=tuple(history_rows),
        legal_retrieval_train=tuple(retrieval_rows),
        ranker_targets=ranker_targets,
        validation_targets=validation_targets,
        test_targets=test_targets,
        histories=histories,
        input_fingerprint=fingerprint,
    )

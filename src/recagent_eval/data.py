from __future__ import annotations

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

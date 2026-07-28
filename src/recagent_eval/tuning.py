from __future__ import annotations

from collections import Counter, defaultdict

from recagent_eval.data import DatasetSplit, Movie
from recagent_eval.ranking import normalize_scores, tune_weights
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever


def tune_on_validation(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    step: float = 0.1,
    max_users: int = 500,
) -> tuple[float, float, float]:
    examples = build_validation_examples(movies, split, max_users=max_users)
    return tune_weights(examples, step=step)


def build_validation_examples(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    max_users: int = 500,
):
    itemcf = ItemCFRetriever.fit(split.train)
    semantic = TfidfSemanticRetriever.fit(movies)
    histories: dict[int, set[int]] = defaultdict(set)
    for row in split.train:
        if row.rating >= 4:
            histories[row.user_id].add(row.movie_id)

    examples = []
    for user_id, target in sorted(split.validation_targets.items())[:max_users]:
        history = histories[user_id]
        genre_counts: Counter[str] = Counter()
        for movie_id in history:
            movie = movies.get(movie_id)
            if movie:
                genre_counts.update(movie.genres)
        query = " ".join(genre for genre, _ in genre_counts.most_common(3))
        allowed = set(movies) - history
        itemcf_scores = normalize_scores(
            dict(itemcf.retrieve(history, top_k=100, allowed_ids=allowed))
        )
        semantic_scores = normalize_scores(
            dict(semantic.retrieve(query, top_k=100, allowed_ids=allowed))
        )
        candidate_ids = set(itemcf_scores) | set(semantic_scores)
        preference_scores = {
            movie_id: float(
                bool(set(movie.genres) & {genre for genre, _ in genre_counts.most_common(3)})
            )
            for movie_id, movie in movies.items()
            if movie_id in candidate_ids
        }
        examples.append((itemcf_scores, semantic_scores, preference_scores, {target}))
    return examples

from __future__ import annotations

import time
from collections import Counter, defaultdict

from recagent_eval.agent import build_semantic_profile
from recagent_eval.data import DatasetSplit, Movie
from recagent_eval.evaluation import ndcg_at_k
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import HybridRanker, normalize_scores, tune_weights
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever


def build_retrieval_ablation(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    depths: tuple[int, ...] = (100, 200, 500),
    history_caps: tuple[int, ...] = (10, 20, 50),
    max_users: int = 500,
) -> list[dict[str, float | int]]:
    itemcf = ItemCFRetriever.fit(split.train)
    semantic = TfidfSemanticRetriever.fit(movies)
    ranker = HybridRanker((0.7, 0.3, 0.0))
    histories: dict[int, set[int]] = defaultdict(set)
    for row in split.train:
        if row.rating >= 4 and row.movie_id in movies:
            histories[row.user_id].add(row.movie_id)

    validation_users = [
        (user_id, target)
        for user_id, target in sorted(split.validation_targets.items())
        if histories[user_id] and target in movies
    ][:max_users]
    rows: list[dict[str, float | int]] = []
    for depth in depths:
        for history_cap in history_caps:
            started = time.perf_counter()
            itemcf_hits = 0
            semantic_hits = 0
            union_hits = 0
            ndcgs: list[float] = []
            for user_id, target in validation_users:
                history = histories[user_id]
                genre_counts: Counter[str] = Counter(
                    genre
                    for movie_id in history
                    for genre in movies[movie_id].genres
                )
                state = PreferenceState(
                    liked_movie_ids=history,
                    liked_genres={
                        genre for genre, _ in genre_counts.most_common(3)
                    },
                )
                allowed_movies = {
                    movie_id: movie
                    for movie_id, movie in movies.items()
                    if movie_id not in history
                }
                itemcf_scores = dict(
                    itemcf.retrieve(
                        history,
                        top_k=depth,
                        allowed_ids=set(allowed_movies),
                    )
                )
                semantic_scores = dict(
                    semantic.retrieve(
                        build_semantic_profile(
                            "",
                            state,
                            movies,
                            history_cap=history_cap,
                        ),
                        top_k=depth,
                        allowed_ids=set(allowed_movies),
                    )
                )
                itemcf_hits += target in itemcf_scores
                semantic_hits += target in semantic_scores
                union_hits += target in set(itemcf_scores) | set(semantic_scores)
                ranked = ranker.rank(
                    allowed_movies,
                    itemcf_scores=itemcf_scores,
                    semantic_scores=semantic_scores,
                    state=state,
                    top_k=10,
                )
                ndcgs.append(
                    ndcg_at_k(
                        [movie.movie_id for movie in ranked],
                        {target},
                        10,
                    )
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            users = len(validation_users)
            denominator = users or 1
            rows.append(
                {
                    "retrieval_top_k": depth,
                    "semantic_profile_history_cap": history_cap,
                    "itemcf_candidate_recall": itemcf_hits / denominator,
                    "semantic_candidate_recall": semantic_hits / denominator,
                    "union_candidate_recall": union_hits / denominator,
                    "ndcg_at_10": sum(ndcgs) / denominator,
                    "latency_ms_per_user": elapsed_ms / denominator,
                    "users": users,
                }
            )
    return rows


def select_retrieval_parameters(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    rows: list[dict[str, float | int]] | None = None,
) -> dict[str, float | int | str]:
    rows = rows if rows is not None else build_retrieval_ablation(movies, split)
    best = max(
        rows,
        key=lambda row: (
            row["ndcg_at_10"],
            row["union_candidate_recall"],
            -row["retrieval_top_k"],
            -row["semantic_profile_history_cap"],
        ),
    )
    return {
        **best,
        "selection_metric": "validation_ndcg_at_10",
    }


def tune_on_validation(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    step: float = 0.1,
    max_users: int = 500,
    retrieval_top_k: int = 100,
    semantic_profile_history_cap: int = 20,
) -> tuple[float, float, float]:
    examples = build_validation_examples(
        movies,
        split,
        max_users=max_users,
        retrieval_top_k=retrieval_top_k,
        semantic_profile_history_cap=semantic_profile_history_cap,
    )
    return tune_weights(examples, step=step)


def build_validation_examples(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    max_users: int = 500,
    retrieval_top_k: int = 100,
    semantic_profile_history_cap: int = 20,
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
        state = PreferenceState(
            liked_movie_ids=history,
            liked_genres={genre for genre, _ in genre_counts.most_common(3)},
        )
        allowed = set(movies) - history
        itemcf_scores = normalize_scores(
            dict(
                itemcf.retrieve(
                    history,
                    top_k=retrieval_top_k,
                    allowed_ids=allowed,
                )
            )
        )
        semantic_scores = normalize_scores(
            dict(
                semantic.retrieve(
                    build_semantic_profile(
                        "",
                        state,
                        movies,
                        history_cap=semantic_profile_history_cap,
                    ),
                    top_k=retrieval_top_k,
                    allowed_ids=allowed,
                )
            )
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

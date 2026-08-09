from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from recagent_eval.agent import build_semantic_profile
from recagent_eval.data import DatasetSplit, Movie
from recagent_eval.evaluation import hit_rate_at_k, ndcg_at_k, recall_at_k
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import HybridRanker, RankerKind
from recagent_eval.retrieval import ItemCFRetriever, TfidfSemanticRetriever

SELECTABLE_METHOD_PRIORITY = {
    "itemcf": 0,
    "rrf": 1,
    "percentile_linear": 2,
}


class RankerSelectionEvidence(BaseModel):
    rows: list[dict[str, Any]]
    selected: dict[str, Any]
    itemcf_ndcg_at_10: float
    selected_ndcg_at_10: float
    margin: float
    test_unlocked: bool
    dataset_fingerprint: str
    retrieval_top_k: int
    semantic_profile_history_cap: int
    max_users: int


@dataclass(frozen=True)
class RankerExample:
    itemcf_scores: dict[int, float]
    semantic_scores: dict[int, float]
    relevant_movie_ids: set[int]


def build_ranker_ablation(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    rrf_ks: tuple[int, ...] = (10, 30, 60, 100),
    weight_step: float = 0.1,
    max_users: int = 500,
    retrieval_top_k: int = 500,
    history_cap: int = 50,
) -> list[dict[str, Any]]:
    examples = _build_validation_examples(
        movies,
        split,
        max_users=max_users,
        retrieval_top_k=retrieval_top_k,
        history_cap=history_cap,
    )
    specifications: list[tuple[RankerKind, dict[str, Any]]] = [
        ("itemcf", {}),
        ("minmax_linear", {"weights": [0.7, 0.3]}),
    ]
    specifications.extend(("rrf", {"rrf_k": k}) for k in rrf_ks)
    units = round(1 / weight_step)
    specifications.extend(
        (
            "percentile_linear",
            {
                "weights": [
                    round(left * weight_step, 10),
                    round((units - left) * weight_step, 10),
                ]
            },
        )
        for left in range(units, -1, -1)
    )
    return [
        _evaluate_ranker(movies, examples, kind=kind, parameters=parameters)
        for kind, parameters in specifications
    ]


def select_ranker(
    rows: list[dict[str, Any]],
    *,
    dataset_fingerprint: str,
    retrieval_top_k: int,
    history_cap: int,
    max_users: int,
) -> RankerSelectionEvidence:
    itemcf_rows = [row for row in rows if row.get("kind") == "itemcf"]
    if len(itemcf_rows) != 1:
        raise ValueError("ranker ablation requires exactly one ItemCF row")
    eligible = [
        row for row in rows if str(row.get("kind")) in SELECTABLE_METHOD_PRIORITY
    ]
    best = max(
        eligible,
        key=lambda row: (
            float(row["ndcg_at_10"]),
            float(row["recall_at_10"]),
            -SELECTABLE_METHOD_PRIORITY[str(row["kind"])],
            json.dumps(row.get("parameters", {}), sort_keys=True),
        ),
    )
    itemcf_ndcg = float(itemcf_rows[0]["ndcg_at_10"])
    selected_ndcg = float(best["ndcg_at_10"])
    margin = selected_ndcg - itemcf_ndcg
    unlocked = (
        str(best["kind"]) in {"rrf", "percentile_linear"}
        and margin > 1e-12
    )
    return RankerSelectionEvidence(
        rows=rows,
        selected=best,
        itemcf_ndcg_at_10=itemcf_ndcg,
        selected_ndcg_at_10=selected_ndcg,
        margin=margin,
        test_unlocked=unlocked,
        dataset_fingerprint=dataset_fingerprint,
        retrieval_top_k=retrieval_top_k,
        semantic_profile_history_cap=history_cap,
        max_users=max_users,
    )


def validate_test_gate(
    evidence: RankerSelectionEvidence,
    *,
    dataset_fingerprint: str,
    retrieval_top_k: int,
    semantic_profile_history_cap: int,
    ranker_kind: RankerKind,
    ranker_parameters: dict[str, Any],
) -> None:
    if not evidence.test_unlocked:
        raise ValueError("frozen test is locked: validation did not beat ItemCF")
    expected_parameters = evidence.selected.get("parameters", {})
    comparisons = [
        ("dataset_fingerprint", evidence.dataset_fingerprint, dataset_fingerprint),
        ("retrieval_top_k", evidence.retrieval_top_k, retrieval_top_k),
        (
            "semantic_profile_history_cap",
            evidence.semantic_profile_history_cap,
            semantic_profile_history_cap,
        ),
        ("ranker_kind", evidence.selected.get("kind"), ranker_kind),
        ("ranker_parameters", expected_parameters, ranker_parameters),
    ]
    mismatches = [
        f"{name}: evidence={expected!r}, actual={actual!r}"
        for name, expected, actual in comparisons
        if expected != actual
    ]
    if mismatches:
        raise ValueError("ranker selection evidence mismatch: " + "; ".join(mismatches))


def ranker_dataset_fingerprint(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    max_users: int,
    retrieval_top_k: int,
    history_cap: int,
) -> str:
    payload = {
        "movies": [
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": list(movie.genres),
                "year": movie.year,
            }
            for movie in sorted(movies.values(), key=lambda item: item.movie_id)
        ],
        "train": [
            {
                "user_id": row.user_id,
                "movie_id": row.movie_id,
                "rating": row.rating,
                "timestamp": row.timestamp,
            }
            for row in sorted(
                split.train,
                key=lambda item: (
                    item.user_id,
                    item.timestamp,
                    item.movie_id,
                    item.rating,
                ),
            )
        ],
        "validation_targets": sorted(split.validation_targets.items()),
        "max_users": max_users,
        "retrieval_top_k": retrieval_top_k,
        "semantic_profile_history_cap": history_cap,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_validation_examples(
    movies: dict[int, Movie],
    split: DatasetSplit,
    *,
    max_users: int,
    retrieval_top_k: int,
    history_cap: int,
) -> list[RankerExample]:
    itemcf = ItemCFRetriever.fit(split.train)
    semantic = TfidfSemanticRetriever.fit(movies)
    histories: dict[int, set[int]] = defaultdict(set)
    for row in split.train:
        if row.rating >= 4 and row.movie_id in movies:
            histories[row.user_id].add(row.movie_id)
    validation_users = [
        (user_id, target)
        for user_id, target in sorted(split.validation_targets.items())
        if histories[user_id] and target in movies
    ][:max_users]
    examples: list[RankerExample] = []
    for user_id, target in validation_users:
        history = histories[user_id]
        genre_counts: Counter[str] = Counter(
            genre
            for movie_id in history
            for genre in movies[movie_id].genres
        )
        state = PreferenceState(
            liked_movie_ids=history,
            liked_genres={genre for genre, _ in genre_counts.most_common(3)},
        )
        allowed_movies = {
            movie_id: movie
            for movie_id, movie in movies.items()
            if movie_id not in history
        }
        allowed_ids = set(allowed_movies)
        itemcf_scores = dict(
            itemcf.retrieve(
                history,
                top_k=retrieval_top_k,
                allowed_ids=allowed_ids,
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
                top_k=retrieval_top_k,
                allowed_ids=allowed_ids,
            )
        )
        examples.append(
            RankerExample(
                itemcf_scores=itemcf_scores,
                semantic_scores=semantic_scores,
                relevant_movie_ids={target},
            )
        )
    return examples


def _evaluate_ranker(
    movies: dict[int, Movie],
    examples: list[RankerExample],
    *,
    kind: RankerKind,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    route_weights = parameters.get("weights", [0.7, 0.3])
    ranker = HybridRanker(
        weights=(float(route_weights[0]), float(route_weights[1]), 0.0),
        kind=kind,
        rrf_k=int(parameters.get("rrf_k", 60)),
    )
    recalls: list[float] = []
    ndcgs: list[float] = []
    hits: list[float] = []
    itemcf_hits: list[float] = []
    semantic_hits: list[float] = []
    union_hits: list[float] = []
    started = time.perf_counter()
    for example in examples:
        ranked = ranker.rank(
            movies,
            itemcf_scores=example.itemcf_scores,
            semantic_scores=example.semantic_scores,
            state=PreferenceState(),
            top_k=10,
        )
        ranked_ids = [movie.movie_id for movie in ranked]
        relevant = example.relevant_movie_ids
        recalls.append(recall_at_k(ranked_ids, relevant, 10))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant, 10))
        hits.append(hit_rate_at_k(ranked_ids, relevant, 10))
        itemcf_hits.append(float(bool(set(example.itemcf_scores) & relevant)))
        semantic_hits.append(float(bool(set(example.semantic_scores) & relevant)))
        union_hits.append(
            float(
                bool(
                    (set(example.itemcf_scores) | set(example.semantic_scores))
                    & relevant
                )
            )
        )
    users = len(examples)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "kind": kind,
        "parameters": parameters,
        "recall_at_10": _mean(recalls),
        "ndcg_at_10": _mean(ndcgs),
        "hit_rate_at_10": _mean(hits),
        "itemcf_candidate_recall": _mean(itemcf_hits),
        "semantic_candidate_recall": _mean(semantic_hits),
        "union_candidate_recall": _mean(union_hits),
        "latency_ms_per_user": elapsed_ms / users if users else 0.0,
        "users": users,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

from __future__ import annotations

import pytest

from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.ranker_selection import (
    RankerSelectionEvidence,
    build_ranker_ablation,
    ranker_dataset_fingerprint,
    select_ranker,
    validate_test_gate,
)


def _row(kind: str, ndcg: float, recall: float = 0.1, **parameters):
    return {
        "kind": kind,
        "parameters": parameters,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "hit_rate_at_10": recall,
        "users": 10,
    }


def test_tie_with_itemcf_never_unlocks_test() -> None:
    evidence = select_ranker(
        [
            _row("itemcf", 0.2),
            _row("rrf", 0.2, rrf_k=30),
        ],
        dataset_fingerprint="abc",
        retrieval_top_k=500,
        history_cap=50,
        max_users=10,
    )

    assert evidence.selected["kind"] == "itemcf"
    assert evidence.test_unlocked is False
    assert evidence.margin == 0.0


def test_strict_improvement_unlocks_exact_selected_ranker() -> None:
    evidence = select_ranker(
        [
            _row("itemcf", 0.2),
            _row("rrf", 0.21, rrf_k=30),
            _row("percentile_linear", 0.205, weights=[0.8, 0.2]),
        ],
        dataset_fingerprint="abc",
        retrieval_top_k=500,
        history_cap=50,
        max_users=10,
    )

    assert evidence.selected["kind"] == "rrf"
    assert evidence.test_unlocked is True
    assert evidence.margin == pytest.approx(0.01)


def test_minmax_control_cannot_unlock_test() -> None:
    evidence = select_ranker(
        [
            _row("itemcf", 0.2),
            _row("minmax_linear", 0.4, weights=[0.7, 0.3]),
            _row("rrf", 0.19, rrf_k=30),
        ],
        dataset_fingerprint="abc",
        retrieval_top_k=500,
        history_cap=50,
        max_users=10,
    )

    assert evidence.selected["kind"] == "itemcf"
    assert evidence.test_unlocked is False


def test_gate_reports_all_evidence_mismatches() -> None:
    evidence = RankerSelectionEvidence.model_validate(
        {
            "rows": [_row("itemcf", 0.2), _row("rrf", 0.21, rrf_k=30)],
            "selected": _row("rrf", 0.21, rrf_k=30),
            "itemcf_ndcg_at_10": 0.2,
            "selected_ndcg_at_10": 0.21,
            "margin": 0.01,
            "test_unlocked": True,
            "dataset_fingerprint": "abc",
            "retrieval_top_k": 500,
            "semantic_profile_history_cap": 50,
            "max_users": 10,
        }
    )

    with pytest.raises(ValueError, match="dataset_fingerprint.*retrieval_top_k"):
        validate_test_gate(
            evidence,
            dataset_fingerprint="different",
            retrieval_top_k=200,
            semantic_profile_history_cap=50,
            ranker_kind="rrf",
            ranker_parameters={"rrf_k": 30},
        )


def test_retrieval_ablation_has_all_rankers_and_stable_fingerprint() -> None:
    movies = {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            2000,
        )
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, (user_id - 1) * 4 + offset + 1, 5, offset + 1)
        for user_id in range(1, 3)
        for offset in range(4)
    ]
    split = chronological_split(ratings)

    rows = build_ranker_ablation(
        movies,
        split,
        rrf_ks=(10, 30, 60, 100),
        weight_step=0.1,
        max_users=1,
        retrieval_top_k=2,
        history_cap=1,
    )
    fingerprint = ranker_dataset_fingerprint(
        movies,
        split,
        max_users=1,
        retrieval_top_k=2,
        history_cap=1,
    )
    second_fingerprint = ranker_dataset_fingerprint(
        movies,
        split,
        max_users=1,
        retrieval_top_k=2,
        history_cap=1,
    )

    assert [row["kind"] for row in rows] == [
        "itemcf",
        "minmax_linear",
        "rrf",
        "rrf",
        "rrf",
        "rrf",
        *["percentile_linear"] * 11,
    ]
    assert all(row["users"] == 1 for row in rows)
    assert all(0.0 <= row["ndcg_at_10"] <= 1.0 for row in rows)
    assert fingerprint == second_fingerprint

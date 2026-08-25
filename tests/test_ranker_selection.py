from __future__ import annotations

import pytest

from recagent_eval.cases import EvaluationCase
from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.models import PreferenceState
from recagent_eval.ranker_selection import (
    RankerSelectionEvidence,
    build_ranker_ablation,
    evaluate_frozen_cases,
    ranker_dataset_fingerprint,
    select_ranker,
    validate_test_gate,
)
from recagent_eval.ranking import HybridRanker


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
        case_fingerprint="cases",
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
        case_fingerprint="cases",
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
        case_fingerprint="cases",
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
            "case_fingerprint": "cases",
            "retrieval_top_k": 500,
            "semantic_profile_history_cap": 50,
            "max_users": 10,
        }
    )

    with pytest.raises(ValueError, match="dataset_fingerprint.*retrieval_top_k"):
        validate_test_gate(
            evidence,
            dataset_fingerprint="different",
            case_fingerprint="cases",
            retrieval_top_k=200,
            semantic_profile_history_cap=50,
            ranker_kind="rrf",
            ranker_parameters={"rrf_k": 30},
        )


def test_gate_recomputes_and_rejects_contradictory_evidence() -> None:
    evidence = RankerSelectionEvidence.model_validate(
        {
            "rows": [_row("itemcf", 0.2), _row("rrf", 0.19, rrf_k=30)],
            "selected": _row("rrf", 0.19, rrf_k=30),
            "itemcf_ndcg_at_10": 0.2,
            "selected_ndcg_at_10": 0.19,
            "margin": -0.01,
            "test_unlocked": True,
            "dataset_fingerprint": "abc",
            "case_fingerprint": "cases",
            "retrieval_top_k": 500,
            "semantic_profile_history_cap": 50,
            "max_users": 10,
        }
    )

    with pytest.raises(ValueError, match="inconsistent"):
        validate_test_gate(
            evidence,
            dataset_fingerprint="abc",
            case_fingerprint="cases",
            retrieval_top_k=500,
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


def test_repeated_ablation_has_deterministic_ranking_metrics() -> None:
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id}", ("Drama",), 2000)
        for movie_id in range(1, 5)
    }
    split = chronological_split(
        [Rating(1, movie_id, 5, movie_id) for movie_id in range(1, 5)]
    )

    first = build_ranker_ablation(
        movies,
        split,
        rrf_ks=(10,),
        weight_step=1.0,
        max_users=1,
        retrieval_top_k=2,
        history_cap=1,
    )
    second = build_ranker_ablation(
        movies,
        split,
        rrf_ks=(10,),
        weight_step=1.0,
        max_users=1,
        retrieval_top_k=2,
        history_cap=1,
    )

    def deterministic_fields(rows):
        return [
            {key: value for key, value in row.items() if key != "latency_ms_per_user"}
            for row in rows
        ]

    assert deterministic_fields(first) == deterministic_fields(second)


def test_frozen_case_evaluation_uses_expected_multi_turn_preferences(
    monkeypatch,
) -> None:
    movies = {
        1: Movie(1, "History", ("Drama",), 2000),
        2: Movie(2, "Target", ("Comedy",), 2001),
        3: Movie(3, "Blocked", ("Horror",), 2002),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(2, 1, 5, 1),
        Rating(2, 2, 5, 2),
    ]
    initial = PreferenceState(liked_movie_ids={1})
    expected = PreferenceState(liked_movie_ids={1}, excluded_genres={"Horror"})
    cases = [
        EvaluationCase(
            case_id="multi-001",
            user_id=1,
            turns=("I like drama", "Avoid horror"),
            relevant_movie_ids={2},
            initial_state=initial,
            expected_preferences=expected,
        )
    ]
    seen_states = []

    def capture_filter(catalog, state):
        seen_states.append(state)
        return list(catalog)

    monkeypatch.setattr("recagent_eval.ranker_selection.hard_filter", capture_filter)

    metrics = evaluate_frozen_cases(
        movies,
        ratings,
        cases,
        ranker=HybridRanker(kind="itemcf"),
        retrieval_top_k=10,
        history_cap=10,
    )

    assert seen_states == [expected]
    assert metrics["cases"] == 1
    assert metrics["ranker_kind"] == "itemcf"
    assert metrics["recall_at_10"] == 1.0


def test_frozen_v2b_contract_uses_three_routes_and_only_state_visible_history(
    monkeypatch,
) -> None:
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id}", ("Drama",), 2000 + movie_id)
        for movie_id in range(1, 10)
    }
    ratings = [
        Rating(1, 1, 5, 10),
        Rating(1, 3, 5, 90),  # Dataset interaction hidden from case state.
        Rating(1, 9, 5, 100),  # Synthetic hidden target must not enter history.
        Rating(2, 1, 5, 11),
        Rating(2, 4, 5, 12),
    ]
    state = PreferenceState(liked_movie_ids={1, 2})
    case = EvaluationCase(
        case_id="synthetic-001",
        user_id=1,
        turns=("recommend",),
        relevant_movie_ids={9},
        initial_state=state,
    )
    observed: dict[str, object] = {}

    class FakeItemCF:
        def retrieve(self, history, *, top_k, allowed_ids):
            observed["itemcf"] = (set(history), top_k, set(allowed_ids))
            return [(4, 0.4)]

        def score_many(self, history, movie_ids):
            observed["recent"] = (set(history), set(movie_ids))
            return {movie_id: 0.1 for movie_id in movie_ids}

    def fit_itemcf(rows):
        observed["fit_rows"] = tuple(rows)
        return FakeItemCF()

    monkeypatch.setattr(
        "recagent_eval.ranker_selection.ItemCFRetriever.fit", fit_itemcf
    )

    class FakeSemantic:
        def retrieve(self, query, *, top_k, allowed_ids):
            observed["semantic"] = (query, top_k, set(allowed_ids))
            return [(5, 0.5)]

    class FakeLatent:
        def retrieve(self, history, *, top_k, allowed_ids):
            observed["latent"] = (set(history), top_k, set(allowed_ids))
            return [(6, 0.6)]

    class FakeLearnedRanker:
        kind = "lambdamart"

        def rank(self, movies, **kwargs):
            observed["rank"] = (set(movies), kwargs)
            return []

    metrics = evaluate_frozen_cases(
        movies,
        ratings,
        [case],
        ranker=FakeLearnedRanker(),
        retrieval_top_k=500,
        semantic_top_k=1500,
        latent_top_k=500,
        history_cap=50,
        semantic_retriever=FakeSemantic(),
        latent_retriever=FakeLatent(),
        feature_version="v2b",
    )

    assert observed["itemcf"][0:2] == ({1, 2}, 500)
    assert observed["semantic"][1] == 1500
    assert observed["latent"][0:2] == ({1, 2}, 500)
    assert observed["recent"] == ({1, 2}, {4, 5, 6})
    fit_rows = observed["fit_rows"]
    assert not any(row.user_id == 1 and row.movie_id in {3, 9} for row in fit_rows)
    _, rank_kwargs = observed["rank"]
    assert rank_kwargs["latent_scores"] == {6: 0.6}
    assert rank_kwargs["recent_itemcf_scores"] == {4: 0.1, 5: 0.1, 6: 0.1}
    assert {row.movie_id for row in rank_kwargs["history_rows"]} == {1, 2}
    assert not any(
        row.user_id == 1 and row.movie_id in {3, 9}
        for row in rank_kwargs["statistics_rows"]
    )
    assert metrics["latent_candidate_recall"] == 0.0

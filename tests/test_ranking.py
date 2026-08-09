import math

import pytest

from recagent_eval.data import Movie
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import (
    HybridRanker,
    percentile_scores,
    reciprocal_rank_scores,
    tune_weights,
)

MOVIES = {
    1: Movie(1, "A", ("Sci-Fi",), 2000),
    2: Movie(2, "B", ("Drama",), 2001),
    3: Movie(3, "C", ("Sci-Fi", "Adventure"), 2002),
}


def test_hybrid_ranker_normalizes_scores_and_applies_preference_affinity() -> None:
    ranker = HybridRanker(weights=(0.5, 0.3, 0.2))
    state = PreferenceState(liked_genres={"Sci-Fi"}, disliked_genres={"Drama"})

    ranked = ranker.rank(
        MOVIES,
        itemcf_scores={1: 1.0, 2: 0.0, 3: 0.8},
        semantic_scores={1: 0.2, 2: 1.0, 3: 0.8},
        state=state,
        top_k=3,
    )

    assert [item.movie_id for item in ranked] == [3, 1, 2]
    assert ranked[0].score.final >= ranked[1].score.final


def test_tune_weights_selects_validation_ndcg_and_is_deterministic() -> None:
    examples = [
        (
            {1: 0.1, 2: 1.0},
            {1: 1.0, 2: 0.0},
            {1: 0.0, 2: 0.0},
            {1},
        )
    ]

    weights = tune_weights(examples, step=0.5)

    assert weights == (0.0, 1.0, 0.0)


def test_rrf_sums_route_rank_contributions() -> None:
    itemcf = reciprocal_rank_scores({1: 9.0, 2: 8.0}, k=10)
    semantic = reciprocal_rank_scores({2: 0.9, 3: 0.8}, k=10)

    assert itemcf == {1: 1 / 11, 2: 1 / 12}
    assert semantic == {2: 1 / 11, 3: 1 / 12}


def test_percentiles_handle_empty_singleton_and_ties() -> None:
    assert percentile_scores({}) == {}
    assert percentile_scores({7: 2.0}) == {7: 1.0}
    assert percentile_scores({1: 5.0, 2: 5.0, 3: 1.0}) == {
        1: 1.0,
        2: 1.0,
        3: 0.0,
    }


def test_rrf_ranker_promotes_cross_route_support() -> None:
    ranked = HybridRanker(kind="rrf", rrf_k=10).rank(
        MOVIES,
        itemcf_scores={1: 10.0, 2: 9.0, 3: 8.0},
        semantic_scores={3: 1.0},
        state=PreferenceState(),
        top_k=3,
    )

    assert ranked[0].movie_id == 3


def test_itemcf_ranker_does_not_promote_semantic_only_candidates() -> None:
    ranked = HybridRanker(kind="itemcf").rank(
        MOVIES,
        itemcf_scores={2: 1.0},
        semantic_scores={1: 10.0},
        state=PreferenceState(),
        top_k=3,
    )

    assert [item.movie_id for item in ranked] == [2]


def test_ranker_rejects_non_finite_route_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        HybridRanker(kind="rrf").rank(
            MOVIES,
            itemcf_scores={1: math.nan},
            semantic_scores={},
            state=PreferenceState(),
        )


def test_ranker_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown ranker kind"):
        HybridRanker(kind="unknown").rank(  # type: ignore[arg-type]
            MOVIES,
            itemcf_scores={1: 1.0},
            semantic_scores={},
            state=PreferenceState(),
        )

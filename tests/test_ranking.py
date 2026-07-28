from recagent_eval.data import Movie
from recagent_eval.models import PreferenceState
from recagent_eval.ranking import HybridRanker, tune_weights

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

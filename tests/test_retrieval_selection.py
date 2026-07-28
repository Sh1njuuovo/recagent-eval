from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.tuning import (
    build_retrieval_ablation,
    select_retrieval_parameters,
)


def test_selection_uses_validation_targets_and_deterministic_ties(monkeypatch) -> None:
    rows = [
        {
            "retrieval_top_k": 100,
            "semantic_profile_history_cap": 10,
            "ndcg_at_10": 0.2,
            "union_candidate_recall": 0.5,
        },
        {
            "retrieval_top_k": 200,
            "semantic_profile_history_cap": 20,
            "ndcg_at_10": 0.2,
            "union_candidate_recall": 0.6,
        },
    ]
    monkeypatch.setattr(
        "recagent_eval.tuning.build_retrieval_ablation",
        lambda *args, **kwargs: rows,
    )

    selection = select_retrieval_parameters(object(), object())

    assert selection["retrieval_top_k"] == 200
    assert selection["semantic_profile_history_cap"] == 20
    assert selection["selection_metric"] == "validation_ndcg_at_10"


def test_retrieval_ablation_reports_each_validation_configuration() -> None:
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

    rows = build_retrieval_ablation(
        movies,
        split,
        depths=(1, 2),
        history_caps=(1,),
        max_users=1,
    )

    assert [
        (row["retrieval_top_k"], row["semantic_profile_history_cap"])
        for row in rows
    ] == [(1, 1), (2, 1)]
    assert all(row["users"] == 1 for row in rows)
    assert all(0.0 <= row["union_candidate_recall"] <= 1.0 for row in rows)
    assert all(row["latency_ms_per_user"] >= 0.0 for row in rows)

from recagent_eval.data import Movie, Rating, chronological_split
from recagent_eval.tuning import build_validation_examples


def test_validation_examples_are_bounded_and_do_not_use_test_targets() -> None:
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id}", ("Drama",), 2000)
        for movie_id in range(1, 13)
    }
    ratings = [
        Rating(user_id, (user_id - 1) * 4 + offset + 1, 5, offset + 1)
        for user_id in range(1, 4)
        for offset in range(4)
    ]
    split = chronological_split(ratings)

    examples = build_validation_examples(movies, split, max_users=2)

    assert len(examples) == 2
    test_targets = set(split.test_targets.values())
    assert all(not (set(itemcf) & test_targets) for itemcf, _, _, _ in examples)

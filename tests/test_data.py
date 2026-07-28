from pathlib import Path

from recagent_eval.data import (
    Rating,
    chronological_split,
    load_movielens_movies,
    load_movielens_ratings,
)


def test_loads_movielens_double_colon_files(tmp_path: Path) -> None:
    (tmp_path / "movies.dat").write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n2::Heat (1995)::Action|Crime|Thriller\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "7::1::5::100\n7::2::2::200\n",
        encoding="ascii",
    )

    movies = load_movielens_movies(tmp_path / "movies.dat")
    ratings = load_movielens_ratings(tmp_path / "ratings.dat")

    assert movies[1].title == "Toy Story (1995)"
    assert movies[1].year == 1995
    assert movies[1].genres == ("Animation", "Children's", "Comedy")
    assert ratings[1] == Rating(user_id=7, movie_id=2, rating=2, timestamp=200)


def test_chronological_split_holds_out_latest_positive_items() -> None:
    ratings = [
        Rating(1, 1, 5, 100),
        Rating(1, 2, 2, 200),
        Rating(1, 3, 4, 300),
        Rating(1, 4, 5, 400),
        Rating(2, 5, 4, 100),
        Rating(2, 6, 5, 200),
    ]

    split = chronological_split(ratings)

    assert [row.movie_id for row in split.train if row.user_id == 1] == [1, 2]
    assert split.validation_targets == {1: 3}
    assert split.test_targets == {1: 4}
    assert 2 not in split.test_targets

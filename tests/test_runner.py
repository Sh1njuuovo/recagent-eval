import json
from pathlib import Path

from recagent_eval.cases import EvaluationCase
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState
from recagent_eval.provider import RuleBasedProvider
from recagent_eval.runner import (
    ExperimentConfig,
    canonical_case_payload,
    run_experiment,
)


def test_run_experiment_writes_reproducible_records_and_metrics(
    tmp_path: Path,
) -> None:
    movies = {
        1: Movie(1, "Space One (2000)", ("Sci-Fi",), 2000),
        2: Movie(2, "Space Two (2001)", ("Sci-Fi",), 2001),
        3: Movie(3, "Drama (2002)", ("Drama",), 2002),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(2, 1, 5, 1),
        Rating(2, 2, 4, 2),
    ]
    cases = [
        EvaluationCase(
            case_id="single-001",
            user_id=1,
            turns=("Recommend science fiction",),
            relevant_movie_ids={2},
            initial_state=PreferenceState(
                liked_movie_ids={1},
                liked_genres={"Sci-Fi"},
                requested_count=2,
            ),
        )
    ]

    metrics = run_experiment(
        movies=movies,
        ratings=ratings,
        cases=cases,
        provider=RuleBasedProvider(),
        config=ExperimentConfig(name="full", weights=(0.5, 0.3, 0.2)),
        output_dir=tmp_path,
    )

    record = json.loads((tmp_path / "episodes.jsonl").read_text().strip())
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert record["case_id"] == "single-001"
    assert record["recommended_movie_ids"] == [2]
    assert metrics == saved_metrics
    assert metrics["hit_rate_at_10"] == 1.0
    assert (tmp_path / "run_manifest.json").exists()


def test_case_fingerprint_payload_sorts_set_like_fields() -> None:
    case = EvaluationCase(
        case_id="order",
        user_id=1,
        turns=("hello",),
        relevant_movie_ids={9, 2},
        initial_state=PreferenceState(
            liked_movie_ids={7, 1},
            liked_genres={"Sci-Fi", "Action"},
        ),
    )

    payload = canonical_case_payload([case])

    assert payload[0]["relevant_movie_ids"] == [2, 9]
    assert payload[0]["initial_state"]["liked_movie_ids"] == [1, 7]
    assert payload[0]["initial_state"]["liked_genres"] == ["Action", "Sci-Fi"]


def test_multi_turn_run_aggregates_calls_and_traces(tmp_path: Path) -> None:
    movies = {
        1: Movie(1, "One", ("Drama",), 2000),
        2: Movie(2, "Two", ("Drama",), 2001),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
    ]
    case = EvaluationCase(
        case_id="multi",
        user_id=1,
        turns=("first turn", "second turn"),
        relevant_movie_ids={2},
        initial_state=PreferenceState(liked_movie_ids={1}),
    )

    metrics = run_experiment(
        movies=movies,
        ratings=ratings,
        cases=[case],
        provider=RuleBasedProvider(),
        config=ExperimentConfig(name="full"),
        output_dir=tmp_path,
    )

    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert metrics["llm_calls"] == 2
    assert len(episode["result"]["traces"]) == 12

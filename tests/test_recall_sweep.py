from __future__ import annotations

import json

from typer.testing import CliRunner

from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.recall_sweep import (
    DEFAULT_VARIANTS,
    RecallResult,
    RecallVariant,
    run_recall_sweep,
    select_recall_winner,
)


def _tiny_movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(
            movie_id,
            f"Movie {movie_id}",
            ("Drama",) if movie_id % 2 else ("Comedy",),
            1990 + movie_id,
        )
        for movie_id in range(1, 9)
    }


def _tiny_ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id)
        for user_id in range(1, 7)
        for movie_id in range(1, 6)
    ]


class _AllCandidatesSemantic:
    """Synthetic retriever that admits every allowed movie."""

    kind = "synthetic"

    def retrieve(self, query, *, top_k=100, allowed_ids=None):
        del query
        return [
            (movie_id, 1.0 / index)
            for index, movie_id in enumerate(sorted(allowed_ids or ()), start=1)
        ][:top_k]


def _baseline_result(**overrides) -> RecallResult:
    values = {
        "variant": RecallVariant("baseline-top500", 500, 50),
        "user_count": 100,
        "dense_recall": 0.288,
        "itemcf_recall": 0.696,
        "union_recall": 0.776,
        "fingerprint": "f",
    }
    values.update(overrides)
    return RecallResult(**values)


def test_recall_sweep_baseline_counts_all_union_and_dense_candidates() -> None:
    movies = _tiny_movies()
    ratings = _tiny_ratings()
    split = leakage_safe_ranking_split(ratings)
    results, decision = run_recall_sweep(
        movies,
        split,
        _AllCandidatesSemantic(),
        retrieval_top_k=8,
        max_users=6,
        dataset_fingerprint="dataset",
        variants=(RecallVariant("baseline-top500", 500, 50),),
    )
    assert len(results) == 1
    result = results[0]
    assert result.user_count == 6
    assert result.dense_recall == 1.0
    assert result.union_recall == 1.0
    assert decision.baseline == result


def test_recall_variant_fingerprint_changes_with_top_k() -> None:
    movies = _tiny_movies()
    split = leakage_safe_ranking_split(_tiny_ratings())
    results, _ = run_recall_sweep(
        movies,
        split,
        _AllCandidatesSemantic(),
        retrieval_top_k=8,
        max_users=6,
        dataset_fingerprint="dataset",
        variants=(
            RecallVariant("baseline-top500", 500, 50),
            RecallVariant("top750", 750, 50),
        ),
    )
    assert results[0].fingerprint != results[1].fingerprint
    assert results[0].fingerprint == results[0].fingerprint


def test_recall_sweep_uses_only_validation_targets() -> None:
    movies = _tiny_movies()
    split = leakage_safe_ranking_split(_tiny_ratings())
    queries = []
    from recagent_eval.lambdamart_pipeline import (
        _positive_histories,
        build_candidate_queries,
    )

    histories = _positive_histories(split.legal_retrieval_train, movies)
    queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        histories,
        split.validation_targets,
        _AllCandidatesSemantic(),
        retrieval_top_k=8,
        history_cap=2,
        max_users=6,
    )
    assert {query.user_id for query in queries} == set(split.validation_targets)
    assert set(split.validation_targets.values()).isdisjoint(
        {row.movie_id for row in split.legal_retrieval_train}
    )


def test_select_recall_winner_requires_dense_lift_and_union_gain() -> None:
    baseline = _baseline_result()
    lifted = _baseline_result(
        variant=RecallVariant("top1000", 1000, 50),
        dense_recall=0.35,
        union_recall=0.80,
        fingerprint="lifted",
    )
    decision = select_recall_winner([baseline, lifted], dense_lift=0.05)
    assert decision.passed
    assert decision.winner == lifted


def test_select_recall_winner_rejects_small_lift() -> None:
    baseline = _baseline_result()
    small = _baseline_result(
        variant=RecallVariant("top750", 750, 50),
        dense_recall=0.30,
        union_recall=0.78,
        fingerprint="small",
    )
    decision = select_recall_winner([baseline, small], dense_lift=0.05)
    assert not decision.passed
    assert decision.winner is None


def test_select_recall_winner_rejects_union_without_gain() -> None:
    baseline = _baseline_result()
    union_flat = _baseline_result(
        variant=RecallVariant("top1000", 1000, 50),
        dense_recall=0.35,
        union_recall=0.776,
        fingerprint="union-flat",
    )
    decision = select_recall_winner([baseline, union_flat], dense_lift=0.05)
    assert not decision.passed
    assert decision.winner is None


def test_ablate_candidates_cli_writes_evidence_and_refuses_overwrite(
    tmp_path, monkeypatch
) -> None:
    import recagent_eval.cli as cli
    from recagent_eval.cli import app

    movies = _tiny_movies()
    ratings = _tiny_ratings()
    monkeypatch.setattr(cli, "_load_dataset", lambda path: (movies, ratings))
    config = tmp_path / "config.yaml"
    config.write_text(
        "name: recall-sweep\nretrieval_top_k: 8\nsemantic_profile_history_cap: 2\n"
        "semantic:\n  kind: tfidf\n"
    )
    output = tmp_path / "recall.json"
    first = CliRunner().invoke(
        app,
        [
            "ablate-candidates",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--max-users",
            "6",
        ],
    )
    assert first.exit_code == 0, first.output
    evidence = json.loads(output.read_text())
    assert evidence["schema_version"] == "candidate-recall-sweep/v1"
    assert len(evidence["variants"]) == len(DEFAULT_VARIANTS)
    assert "gate" in evidence
    second = CliRunner().invoke(
        app,
        [
            "ablate-candidates",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--max-users",
            "6",
        ],
    )
    assert second.exit_code != 0
    assert "refusing to overwrite" in second.output

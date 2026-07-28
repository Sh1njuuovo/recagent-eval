from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from recagent_eval.cases import (
    EvaluationCase,
    generate_cases,
    load_cases,
    save_cases,
    select_stratified_cases,
)
from recagent_eval.config import load_experiment_config
from recagent_eval.data import (
    Movie,
    Rating,
    chronological_split,
    load_movielens_movies,
    load_movielens_ratings,
)
from recagent_eval.dataset import download_movielens_1m
from recagent_eval.models import PreferenceState
from recagent_eval.provider import OpenAICompatibleProvider, RuleBasedProvider
from recagent_eval.runner import ExperimentConfig, run_experiment
from recagent_eval.tuning import tune_on_validation

app = typer.Typer(no_args_is_help=True, help="Evaluate a conversational movie recommender.")


@app.command("download-data")
def download_data(
    output: Annotated[Path, typer.Option(help="Data directory")] = Path("data/raw"),
) -> None:
    path = download_movielens_1m(output)
    typer.echo(f"MovieLens 1M ready at {path}")


@app.command("prepare-cases")
def prepare_cases(
    data_dir: Annotated[Path, typer.Option(help="Directory containing movies.dat")] = Path(
        "data/raw/ml-1m"
    ),
    output: Annotated[Path, typer.Option(help="Case JSON path")] = Path("cases/fixed_cases.json"),
    single_turn_count: int = 40,
    multi_turn_count: int = 10,
    seed: int = 42,
) -> None:
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    cases = generate_cases(
        movies,
        split,
        ratings,
        single_turn_count=single_turn_count,
        multi_turn_count=multi_turn_count,
        seed=seed,
    )
    if len(cases) != single_turn_count + multi_turn_count:
        raise typer.BadParameter("not enough eligible users to build cases")
    save_cases(cases, output)
    typer.echo(f"Wrote {len(cases)} fixed cases to {output}")


@app.command("tune")
def tune(
    data_dir: Annotated[Path, typer.Option(help="MovieLens 1M directory")] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option(help="Tuned weight JSON")] = Path(
        "artifacts/tuned_weights.json"
    ),
    step: float = 0.1,
) -> None:
    movies, ratings = _load_dataset(data_dir)
    weights = tune_on_validation(movies, chronological_split(ratings), step=step)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"weights": weights}, indent=2) + "\n")
    typer.echo(f"Validation-selected weights: {weights}")


@app.command("subset-cases")
def subset_cases(
    source: Annotated[Path, typer.Option(help="Source case JSON")],
    output: Annotated[Path, typer.Option(help="Subset case JSON")],
    single_turn_count: int = 16,
    multi_turn_count: int = 4,
) -> None:
    try:
        selected = select_stratified_cases(
            load_cases(source),
            single_turn_count=single_turn_count,
            multi_turn_count=multi_turn_count,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    save_cases(selected, output)
    typer.echo(f"Wrote {len(selected)} stratified cases to {output}")


@app.command("evaluate")
def evaluate(
    config_path: Annotated[Path, typer.Option("--config")],
    cases_path: Annotated[Path, typer.Option("--cases")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/runs/latest"),
    provider_name: Annotated[
        str, typer.Option("--provider", help="rule-based, deepseek, or vllm")
    ] = "rule-based",
) -> None:
    config = _validated_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    provider = _provider(provider_name)
    metrics = run_experiment(
        movies=movies,
        ratings=list(split.train),
        cases=load_cases(cases_path),
        provider=provider,
        config=config,
        output_dir=output,
    )
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("show-config")
def show_config(config_path: Path) -> None:
    config = _validated_config(config_path)
    typer.echo(json.dumps(config.__dict__, indent=2))


@app.command("smoke")
def smoke(
    output: Annotated[Path, typer.Option()] = Path("artifacts/runs/smoke"),
) -> None:
    movies = {
        1: Movie(1, "Space One (2000)", ("Sci-Fi",), 2000),
        2: Movie(2, "Space Two (2001)", ("Sci-Fi",), 2001),
        3: Movie(3, "Quiet Drama (2002)", ("Drama",), 2002),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(2, 1, 5, 1),
        Rating(2, 2, 4, 2),
    ]
    cases = [
        EvaluationCase(
            case_id="smoke-001",
            user_id=1,
            turns=("Recommend a science fiction movie.",),
            relevant_movie_ids={2},
            initial_state=PreferenceState(
                liked_movie_ids={1},
                liked_genres={"Sci-Fi"},
            ),
        )
    ]
    run_experiment(
        movies=movies,
        ratings=ratings,
        cases=cases,
        provider=RuleBasedProvider(),
        config=ExperimentConfig(name="offline-smoke"),
        output_dir=output,
    )
    typer.echo(f"Offline smoke test passed; artifacts written to {output}")


def _validated_config(path: Path) -> ExperimentConfig:
    try:
        return load_experiment_config(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_dataset(data_dir: Path) -> tuple[dict[int, Movie], list[Rating]]:
    movies_path = data_dir / "movies.dat"
    ratings_path = data_dir / "ratings.dat"
    if not movies_path.exists() or not ratings_path.exists():
        raise typer.BadParameter(
            f"MovieLens files missing under {data_dir}; run download-data first"
        )
    return (
        load_movielens_movies(movies_path),
        load_movielens_ratings(ratings_path),
    )


def _provider(name: str):
    if name == "rule-based":
        return RuleBasedProvider()
    if name == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise typer.BadParameter("DEEPSEEK_API_KEY is required")
        return OpenAICompatibleProvider(
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=key,
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    if name == "vllm":
        return OpenAICompatibleProvider(
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "local"),
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
    raise typer.BadParameter("provider must be rule-based, deepseek, or vllm")


if __name__ == "__main__":
    app()

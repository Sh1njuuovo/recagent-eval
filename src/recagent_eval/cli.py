from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml

from recagent_eval.candidate_features import FEATURE_SCHEMA_FINGERPRINT
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
    leakage_safe_ranking_split,
    load_movielens_movies,
    load_movielens_ratings,
)
from recagent_eval.dataset import download_movielens_1m
from recagent_eval.lambdamart_pipeline import (
    candidate_policy_fingerprint,
    ranking_dataset_fingerprint,
    train_lambdamart_pipeline,
)
from recagent_eval.learned_ranking import (
    LearnedRanker,
    estimator_from_artifact,
    load_ranker_artifact,
)
from recagent_eval.models import PreferenceState
from recagent_eval.provider import OpenAICompatibleProvider, RuleBasedProvider
from recagent_eval.ranker_selection import (
    RankerSelectionEvidence,
    build_ranker_ablation,
    evaluate_frozen_cases,
    ranker_dataset_fingerprint,
    validate_test_gate,
)
from recagent_eval.ranker_selection import (
    select_ranker as select_ranker_evidence,
)
from recagent_eval.ranking import HybridRanker
from recagent_eval.retrieval import (
    DEFAULT_DENSE_MODEL,
    DenseSemanticRetriever,
    TfidfSemanticRetriever,
)
from recagent_eval.runner import ExperimentConfig, case_fingerprint, run_experiment
from recagent_eval.tuning import (
    build_retrieval_ablation,
    select_retrieval_parameters,
    tune_on_validation,
)
from recagent_eval.v2_selection import (
    LearnedValidationEvidence,
    validate_learned_gate,
)

app = typer.Typer(no_args_is_help=True, help="Evaluate a conversational movie recommender.")


@app.command("download-data")
def download_data(
    output: Annotated[Path, typer.Option(help="Data directory")] = Path("data/raw"),
) -> None:
    path = download_movielens_1m(output)
    typer.echo(f"MovieLens 1M ready at {path}")


@app.command("build-embeddings")
def build_embeddings(
    data_dir: Annotated[Path, typer.Option(help="MovieLens 1M directory")] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option(help="Dense embedding NPZ cache")] = Path(
        "artifacts/embeddings/movielens.npz"
    ),
    model_name: Annotated[str, typer.Option()] = DEFAULT_DENSE_MODEL,
    model_revision: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option(help="cpu or cuda")] = "cpu",
    force: Annotated[bool, typer.Option("--force", help="Rebuild a matching cache")] = False,
) -> None:
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("device must be cpu or cuda")
    movies_path = data_dir / "movies.dat"
    if not movies_path.exists():
        raise typer.BadParameter(
            f"MovieLens movies.dat missing under {data_dir}; run download-data first"
        )
    movies = load_movielens_movies(movies_path)
    try:
        if (output.exists() or Path(f"{output}.json").exists()) and not force:
            manifest = DenseSemanticRetriever.validate_cache(
                output,
                movies=movies,
                model_name=model_name,
                model_revision=model_revision,
                device=device,
            )
            typer.echo(
                f"Reused {len(movies)} embeddings from {output} "
                f"at revision {manifest['resolved_revision']}"
            )
            return
        retriever = DenseSemanticRetriever.fit(
            movies,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
        )
        retriever.save(output)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {len(movies)} embeddings to {output} at revision {retriever.model_revision}")


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
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Frozen retrieval config"),
    ] = None,
    config_output: Annotated[
        Path | None,
        typer.Option("--config-output", help="Config updated with tuned weights"),
    ] = None,
    step: float = 0.1,
) -> None:
    movies, ratings = _load_dataset(data_dir)
    config = _validated_config(config_path) if config_path is not None else None
    weights = tuple(
        round(value, 10)
        for value in tune_on_validation(
            movies,
            chronological_split(ratings),
            step=step,
            retrieval_top_k=config.retrieval_top_k if config else 100,
            semantic_profile_history_cap=(config.semantic_profile_history_cap if config else 20),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"weights": weights}, indent=2) + "\n")
    if config_output is not None:
        if config_path is None:
            raise typer.BadParameter("--config-output requires --config")
        payload = yaml.safe_load(config_path.read_text()) or {}
        payload["weights"] = list(weights)
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    typer.echo(f"Validation-selected weights: {weights}")


@app.command("select-retrieval")
def select_retrieval(
    data_dir: Annotated[Path, typer.Option(help="MovieLens 1M directory")] = Path("data/raw/ml-1m"),
    evidence_output: Annotated[
        Path,
        typer.Option(help="Validation ablation JSON"),
    ] = Path("artifacts/retrieval_ablation.json"),
    config_output: Annotated[
        Path,
        typer.Option(help="Frozen hybrid YAML"),
    ] = Path("configs/full_constraint_aware.yaml"),
) -> None:
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    rows = build_retrieval_ablation(movies, split)
    selection = select_retrieval_parameters(movies, split, rows=rows)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(
            {"rows": rows, "selection": selection},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_config = {
        "name": "structured-memory-hybrid-constraint-aware",
        "seed": 42,
        "retrieval_top_k": int(selection["retrieval_top_k"]),
        "semantic_profile_history_cap": int(selection["semantic_profile_history_cap"]),
        "enable_memory": True,
        "enable_semantic_retrieval": True,
        "structured_planning": True,
        "required_retrieval_tools": [
            "itemcf_retrieve",
            "semantic_retrieve",
        ],
        "weights": [0.7, 0.3, 0.0],
    }
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )
    typer.echo(
        "Selected retrieval_top_k="
        f"{selection['retrieval_top_k']}, semantic_profile_history_cap="
        f"{selection['semantic_profile_history_cap']}"
    )


@app.command("select-ranker")
def select_ranker_command(
    config_path: Annotated[Path, typer.Option("--config")],
    cases_path: Annotated[Path, typer.Option("--cases")] = Path("cases/fixed_cases.json"),
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    evidence_output: Annotated[Path, typer.Option()] = Path("artifacts/ranker_ablation.json"),
    config_output: Annotated[Path, typer.Option()] = Path("configs/full_ranker_selected.yaml"),
    max_users: int = 500,
) -> None:
    config = _validated_config(config_path)
    if not config.enable_semantic_retrieval or config.required_retrieval_tools != (
        "itemcf_retrieve",
        "semantic_retrieve",
    ):
        raise typer.BadParameter(
            "ranker selection requires enabled ItemCF and semantic retrieval routes"
        )
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    rows = build_ranker_ablation(
        movies,
        split,
        max_users=max_users,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
    )
    if config.learned_model_path is not None or config.learned_evidence_path is not None:
        if config.learned_model_path is None or config.learned_evidence_path is None:
            raise typer.BadParameter(
                "ranker.model_path and ranker.evidence_path must be configured together"
            )
        learned_split = leakage_safe_ranking_split(ratings)
        learned_dataset_fingerprint = ranking_dataset_fingerprint(
            movies, learned_split
        )
        try:
            artifact = load_ranker_artifact(
                Path(config.learned_model_path),
                expected_dataset_fingerprint=learned_dataset_fingerprint,
            )
            learned_evidence = LearnedValidationEvidence.model_validate_json(
                Path(config.learned_evidence_path).read_text()
            )
            validate_learned_gate(
                learned_evidence,
                dataset_fingerprint=learned_dataset_fingerprint,
                feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT,
                model_fingerprint=artifact.model_checksum,
                candidate_policy_fingerprint=candidate_policy_fingerprint(config),
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        rows.append(
            {
                "kind": "lambdamart",
                "parameters": artifact.selected_params,
                "recall_at_10": learned_evidence.aggregates["lambdamart_recall_at_10"],
                "ndcg_at_10": learned_evidence.mean_lambdamart_ndcg_at_10,
                "hit_rate_at_10": learned_evidence.aggregates["lambdamart_hit_at_10"],
                "itemcf_candidate_recall": learned_evidence.aggregates["itemcf_candidate_recall"],
                "semantic_candidate_recall": learned_evidence.aggregates["dense_candidate_recall"],
                "union_candidate_recall": learned_evidence.aggregates["union_candidate_recall"],
                "latency_ms_per_user": learned_evidence.aggregates["latency_ms"],
                "users": len(learned_evidence.per_user_rows),
                "validation_evidence_fingerprint": learned_evidence.evidence_fingerprint,
            }
        )
        rows = [
            {
                "kind": "itemcf",
                "parameters": {},
                "recall_at_10": learned_evidence.aggregates[
                    "itemcf_recall_at_10"
                ],
                "ndcg_at_10": learned_evidence.mean_itemcf_ndcg_at_10,
                "hit_rate_at_10": learned_evidence.aggregates[
                    "itemcf_hit_at_10"
                ],
                "itemcf_candidate_recall": learned_evidence.aggregates[
                    "itemcf_candidate_recall"
                ],
                "semantic_candidate_recall": learned_evidence.aggregates[
                    "dense_candidate_recall"
                ],
                "union_candidate_recall": learned_evidence.aggregates[
                    "union_candidate_recall"
                ],
                "latency_ms_per_user": 0.0,
                "users": len(learned_evidence.per_user_rows),
            },
            rows[-1],
        ]
    fingerprint = ranker_dataset_fingerprint(
        movies,
        split,
        max_users=max_users,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
    )
    fixed_case_fingerprint = case_fingerprint(load_cases(cases_path))
    evidence = select_ranker_evidence(
        rows,
        dataset_fingerprint=fingerprint,
        case_fingerprint=fixed_case_fingerprint,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        max_users=max_users,
    )
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not evidence.test_unlocked:
        typer.echo(
            "Frozen test remains locked: selected validation NDCG@10="
            f"{evidence.selected_ndcg_at_10:.6f}, ItemCF="
            f"{evidence.itemcf_ndcg_at_10:.6f}"
        )
        return

    payload = yaml.safe_load(config_path.read_text()) or {}
    selected_kind = str(evidence.selected["kind"])
    parameters = dict(evidence.selected.get("parameters", {}))
    ranker_payload: dict[str, object] = {"kind": selected_kind}
    if selected_kind == "rrf":
        ranker_payload["rrf_k"] = int(parameters["rrf_k"])
    elif selected_kind == "percentile_linear":
        ranker_payload["weights"] = list(parameters["weights"])
    elif selected_kind == "lambdamart":
        ranker_payload["model_path"] = config.learned_model_path
        ranker_payload["evidence_path"] = config.learned_evidence_path
    payload["ranker"] = ranker_payload
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    typer.echo(
        "Frozen test unlocked: selected "
        f"{selected_kind} with validation margin {evidence.margin:.6f}"
    )


@app.command("train-ranker")
def train_ranker(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option(help="LambdaMART artifact JSON")] = Path(
        "artifacts/lambdamart.json"
    ),
    evidence_output: Annotated[Path, typer.Option(help="Per-user validation evidence JSON")] = Path(
        "artifacts/lambdamart-validation.json"
    ),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
    seed: int = 42,
) -> None:
    config = _validated_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    if len(split.ranker_targets) < 3:
        raise typer.BadParameter("train-ranker requires at least three eligible users")
    try:
        if config.semantic_kind == "dense":
            if config.semantic_cache_path is None:
                raise ValueError("semantic.cache_path is required for offline LambdaMART training")
            semantic = DenseSemanticRetriever.load(
                Path(config.semantic_cache_path),
                movies=movies,
                model_name=config.semantic_model_name,
                model_revision=config.semantic_model_revision,
                device=config.semantic_device,
            )
        else:
            semantic = TfidfSemanticRetriever.fit(movies)
        summary = train_lambdamart_pipeline(
            movies,
            split,
            semantic,
            config,
            model_output=output,
            evidence_output=evidence_output,
            max_users=max_users,
            seed=seed,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("evaluate-ranker")
def evaluate_ranker(
    config_path: Annotated[Path, typer.Option("--config")],
    evidence_path: Annotated[Path, typer.Option("--evidence")],
    cases_path: Annotated[Path, typer.Option("--cases")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/runs/offline-ranker-test/metrics.json"
    ),
) -> None:
    config = _validated_config(config_path)
    if config.ranker_kind == "lambdamart":
        _evaluate_learned_ranker(
            config=config,
            evidence_path=evidence_path,
            cases_path=cases_path,
            data_dir=data_dir,
            output=output,
        )
        return
    try:
        evidence = RankerSelectionEvidence.model_validate_json(evidence_path.read_text())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    dataset_fingerprint = ranker_dataset_fingerprint(
        movies,
        split,
        max_users=evidence.max_users,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
    )
    parameters = _ranker_parameters(config)
    cases = load_cases(cases_path)
    fixed_case_fingerprint = case_fingerprint(cases)
    try:
        validate_test_gate(
            evidence,
            dataset_fingerprint=dataset_fingerprint,
            case_fingerprint=fixed_case_fingerprint,
            retrieval_top_k=config.retrieval_top_k,
            semantic_profile_history_cap=config.semantic_profile_history_cap,
            ranker_kind=config.ranker_kind,
            ranker_parameters=parameters,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    metrics = evaluate_frozen_cases(
        movies,
        split.train,
        cases,
        ranker=HybridRanker(
            config.weights,
            kind=config.ranker_kind,
            rrf_k=config.rrf_k,
        ),
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
    )
    metrics.update(
        {
            "selection_margin": evidence.margin,
            "selection_evidence_fingerprint": hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest(),
            "case_fingerprint": fixed_case_fingerprint,
            "retrieval_top_k": config.retrieval_top_k,
            "semantic_profile_history_cap": config.semantic_profile_history_cap,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Frozen ranker test metrics written to {output}")


def _evaluate_learned_ranker(
    *,
    config: ExperimentConfig,
    evidence_path: Path,
    cases_path: Path,
    data_dir: Path,
    output: Path,
) -> None:
    if config.learned_model_path is None:
        raise typer.BadParameter("ranker.model_path is required when ranker.kind is lambdamart")
    try:
        evidence = LearnedValidationEvidence.model_validate_json(evidence_path.read_text())
        movies, ratings = _load_dataset(data_dir)
        split = leakage_safe_ranking_split(ratings)
        dataset_fingerprint = ranking_dataset_fingerprint(movies, split)
        artifact = load_ranker_artifact(
            Path(config.learned_model_path),
            expected_dataset_fingerprint=dataset_fingerprint,
        )
        validate_learned_gate(
            evidence,
            dataset_fingerprint=dataset_fingerprint,
            feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT,
            model_fingerprint=artifact.model_checksum,
            candidate_policy_fingerprint=candidate_policy_fingerprint(config),
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    cases = load_cases(cases_path)
    ranker = LearnedRanker(
        estimator_from_artifact(artifact),
        legal_train_rows=split.legal_retrieval_train,
    )
    if config.semantic_kind == "dense":
        if config.semantic_cache_path is None:
            raise typer.BadParameter(
                "semantic.cache_path is required for frozen LambdaMART evaluation"
            )
        try:
            semantic = DenseSemanticRetriever.load(
                Path(config.semantic_cache_path),
                movies=movies,
                model_name=config.semantic_model_name,
                model_revision=config.semantic_model_revision,
                device=config.semantic_device,
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        semantic = TfidfSemanticRetriever.fit(movies)
    metrics = evaluate_frozen_cases(
        movies,
        split.legal_retrieval_train,
        cases,
        ranker=ranker,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        semantic_retriever=semantic,
    )
    metrics.update(
        {
            "selection_margin": evidence.mean_ndcg_delta,
            "selection_evidence_fingerprint": evidence.evidence_fingerprint,
            "case_fingerprint": case_fingerprint(cases),
            "dataset_fingerprint": dataset_fingerprint,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    typer.echo(f"Frozen LambdaMART test metrics written to {output}")


def _ranker_parameters(config: ExperimentConfig) -> dict[str, object]:
    if config.ranker_kind == "rrf":
        return {"rrf_k": config.rrf_k}
    if config.ranker_kind == "percentile_linear":
        return {"weights": [config.weights[0], config.weights[1]]}
    return {}


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

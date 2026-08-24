from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
import yaml

from recagent_eval.baseline_eval import BASELINE_SCORERS, metric_json
from recagent_eval.baseline_summary import summarize_baselines, summary_to_markdown
from recagent_eval.baselines import als_direct as _als_registration  # noqa: F401
from recagent_eval.baselines import bpr_mf as _bpr_registration  # noqa: F401
from recagent_eval.baselines import current_v2b as _v2b_registration  # noqa: F401
from recagent_eval.baselines import itemcf_direct as _itemcf_registration  # noqa: F401
from recagent_eval.baselines import lightgcn as _lightgcn_registration  # noqa: F401
from recagent_eval.baselines import popularity as _popularity_registration  # noqa: F401
from recagent_eval.bundle import load_ranker_bundle
from recagent_eval.candidate_features import (
    FEATURE_SCHEMA_FINGERPRINT,
    FEATURE_SCHEMA_FINGERPRINT_V2,
    FEATURE_SCHEMA_FINGERPRINT_V2B,
)
from recagent_eval.cases import (
    EvaluationCase,
    generate_cases,
    load_cases,
    save_cases,
    select_stratified_cases,
)
from recagent_eval.cohorts import build_cohort_ledger
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
from recagent_eval.evidence import (
    canonical_digest,
    runtime_dependency_versions,
    runtime_hardware,
)
from recagent_eval.evidence_replay import (
    build_compact_bundle,
    replay_compact_bundle,
    write_new_json,
)
from recagent_eval.lambdamart_pipeline import (
    build_validation_rows,
    candidate_policy_fingerprint,
    lambdamart_config_fingerprint,
    ranking_dataset_fingerprint,
    train_lambdamart_pipeline,
)
from recagent_eval.latent_diagnostics import (
    aggregate_latent_diagnostics,
    build_latent_diagnostic_queries,
    build_latent_user_rows,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.learned_ranking import (
    LearnedRanker,
    estimator_from_artifact,
    parse_ranker_artifact,
)
from recagent_eval.models import PreferenceState
from recagent_eval.promotion import (
    PromotionManifest,
    ReplayVerification,
    audit_one_shot,
    execute_one_shot,
    load_promotion_documents,
    load_source_inventory,
    preflight_promotion,
    publish_promotion_package,
    verify_git_identity,
)
from recagent_eval.provider import RuleBasedProvider, build_provider
from recagent_eval.ranker_diagnostics import (
    aggregate_diagnostics,
    build_diagnostic_queries,
    build_user_diagnostics,
)
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
from recagent_eval.recall_sweep import run_recall_sweep
from recagent_eval.retrieval import (
    DEFAULT_DENSE_MODEL,
    DenseSemanticRetriever,
    TfidfSemanticRetriever,
)
from recagent_eval.robustness import build_parameter_recovery_manifest
from recagent_eval.runner import ExperimentConfig, case_fingerprint, run_experiment
from recagent_eval.safe_io import ensure_distinct_files
from recagent_eval.tuning import (
    build_retrieval_ablation,
    select_retrieval_parameters,
    tune_on_validation,
)
from recagent_eval.v2_selection import (
    LearnedValidationEvidence,
    assert_frozen_authorization_available,
    consume_frozen_authorization,
    consumption_marker_path,
    validate_learned_gate,
)

app = typer.Typer(no_args_is_help=True, help="Evaluate a conversational movie recommender.")


def _canonical_validation_contract(rows: list[dict[str, object]]) -> str:
    stable_rows = [
        {key: value for key, value in row.items() if key != "latency_ms"}
        for row in rows
    ]
    return json.dumps(stable_rows, sort_keys=True, separators=(",", ":"))


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
    learned_paths = (
        config.learned_model_path,
        config.learned_evidence_path,
        config.learned_bundle_manifest_path,
    )
    if any(path is not None for path in learned_paths):
        if any(path is None for path in learned_paths):
            raise typer.BadParameter(
                "ranker model, evidence, and bundle manifest paths must be configured together"
            )
        assert config.learned_model_path is not None
        assert config.learned_evidence_path is not None
        assert config.learned_bundle_manifest_path is not None
        learned_split = leakage_safe_ranking_split(ratings)
        learned_dataset_fingerprint = ranking_dataset_fingerprint(
            movies, learned_split
        )
        registered_case_fingerprint = case_fingerprint(load_cases(cases_path))
        try:
            bundle = load_ranker_bundle(
                Path(config.learned_model_path),
                Path(config.learned_evidence_path),
                Path(config.learned_bundle_manifest_path),
            )
            model_bytes = bundle.model_bytes
            evidence_bytes = bundle.evidence_bytes
            artifact = parse_ranker_artifact(
                model_bytes,
                expected_dataset_fingerprint=learned_dataset_fingerprint,
                expected_candidate_policy_fingerprint=candidate_policy_fingerprint(
                    config
                ),
                expected_config_fingerprint=lambdamart_config_fingerprint(config),
                expected_case_fingerprint=registered_case_fingerprint,
            )
            learned_evidence = LearnedValidationEvidence.model_validate_json(
                evidence_bytes
            )
            validate_learned_gate(
                learned_evidence,
                dataset_fingerprint=learned_dataset_fingerprint,
                feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT,
                model_fingerprint=artifact.model_checksum,
                candidate_policy_fingerprint=candidate_policy_fingerprint(config),
                case_fingerprint=registered_case_fingerprint,
                config_fingerprint=lambdamart_config_fingerprint(config),
                artifact_provenance=artifact.model_dump(mode="python"),
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
        ranker_payload["bundle_manifest_path"] = config.learned_bundle_manifest_path
        ranker_payload["dataset_fingerprint"] = learned_evidence.dataset_fingerprint
        ranker_payload["candidate_policy_fingerprint"] = (
            learned_evidence.candidate_policy_fingerprint
        )
        ranker_payload["config_fingerprint"] = learned_evidence.config_fingerprint
        ranker_payload["case_fingerprint"] = learned_evidence.case_fingerprint
        ranker_payload["gate_fingerprint"] = learned_evidence.evidence_fingerprint
        ranker_payload["consumption_dir"] = (
            config.learned_consumption_dir or "artifacts/frozen-consumption"
        )
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
    cases_path: Annotated[Path, typer.Option("--cases")] = Path(
        "cases/fixed_cases.json"
    ),
    output: Annotated[Path, typer.Option(help="LambdaMART artifact JSON")] = Path(
        "artifacts/lambdamart.json"
    ),
    evidence_output: Annotated[Path, typer.Option(help="Per-user validation evidence JSON")] = Path(
        "artifacts/lambdamart-validation.json"
    ),
    bundle_manifest_output: Annotated[
        Path, typer.Option(help="Atomic model/evidence bundle manifest JSON")
    ] = Path("artifacts/lambdamart-bundle.json"),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
) -> None:
    config = _validated_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    if len(split.ranker_targets) < 3:
        raise typer.BadParameter("train-ranker requires at least three eligible users")
    if not cases_path.exists():
        raise typer.BadParameter(
            "train-ranker requires a registered --cases file for frozen-test provenance"
        )
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
            bundle_manifest_output=bundle_manifest_output,
            max_users=max_users,
            seed=config.seed,
            registered_case_fingerprint=case_fingerprint(load_cases(cases_path)),
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("ablate-candidates")
def ablate_candidates(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-recall-sweep/recall.json"
    ),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
) -> None:
    """Measure dense/ItemCF/union candidate recall across candidate variants."""
    config = _validated_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    if output.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing recall evidence: {output}"
        )
    try:
        if config.semantic_kind == "dense":
            if config.semantic_cache_path is None:
                raise ValueError(
                    "semantic.cache_path is required for dense recall sweeps"
                )
            semantic = DenseSemanticRetriever.load(
                Path(config.semantic_cache_path),
                movies=movies,
                model_name=config.semantic_model_name,
                model_revision=config.semantic_model_revision,
                device=config.semantic_device,
            )
        else:
            semantic = TfidfSemanticRetriever.fit(movies)
        dataset_fingerprint = ranker_dataset_fingerprint(
            movies,
            split,
            max_users=max_users,
            retrieval_top_k=config.retrieval_top_k,
            history_cap=config.semantic_profile_history_cap,
        )
        results, gate = run_recall_sweep(
            movies,
            split,
            semantic,
            retrieval_top_k=config.retrieval_top_k,
            max_users=max_users,
            dataset_fingerprint=dataset_fingerprint,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    evidence = {
        "schema_version": "candidate-recall-sweep/v1",
        "dataset_fingerprint": dataset_fingerprint,
        "retrieval_top_k": config.retrieval_top_k,
        "max_users": max_users,
        "variants": [
            {
                "variant": {
                    "name": result.variant.name,
                    "semantic_top_k": result.variant.semantic_top_k,
                    "history_cap": result.variant.history_cap,
                    "query_style": result.variant.query_style,
                },
                "user_count": result.user_count,
                "dense_recall": result.dense_recall,
                "itemcf_recall": result.itemcf_recall,
                "union_recall": result.union_recall,
                "fingerprint": result.fingerprint,
            }
            for result in results
        ],
        "gate": {
            "baseline": gate.baseline.variant.name,
            "winner": gate.winner.variant.name if gate.winner is not None else None,
            "passed": gate.passed,
            "reason": gate.reason,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(evidence["gate"], indent=2, sort_keys=True))


@app.command("diagnose-ranker")
def diagnose_ranker(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    cases_path: Annotated[Path, typer.Option("--cases")] = Path(
        "cases/fixed_cases.json"
    ),
    model_path: Annotated[Path, typer.Option("--model")] = Path(
        "artifacts/experiments/v2-recall-1500/model.json"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-ranker-diagnostics/diagnostics.json"
    ),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
) -> None:
    """Write read-only ranking diagnostics for the validation user set."""
    config = _validated_config(config_path)
    if output.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing diagnostics artifact: {output}"
        )
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    try:
        if config.semantic_kind == "dense":
            if config.semantic_cache_path is None:
                raise ValueError(
                    "semantic.cache_path is required for dense diagnostics"
                )
            semantic = DenseSemanticRetriever.load(
                Path(config.semantic_cache_path),
                movies=movies,
                model_name=config.semantic_model_name,
                model_revision=config.semantic_model_revision,
                device=config.semantic_device,
            )
        else:
            semantic = TfidfSemanticRetriever.fit(movies)
        artifact = parse_ranker_artifact(Path(model_path).read_bytes())
        learned = LearnedRanker(
            estimator_from_artifact(artifact),
            legal_train_rows=split.legal_retrieval_train,
            score_calibration=config.score_calibration,
        )
        queries = build_diagnostic_queries(
            movies,
            split,
            semantic,
            retrieval_top_k=config.retrieval_top_k,
            history_cap=config.semantic_profile_history_cap,
            semantic_top_k=config.semantic_top_k,
            score_calibration=config.score_calibration,
            max_users=max_users,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    rows = build_user_diagnostics(queries, movies, learned)
    dataset_fingerprint = ranker_dataset_fingerprint(
        movies,
        split,
        max_users=max_users,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
    )
    summary = aggregate_diagnostics(
        rows,
        fingerprints={
            "dataset": dataset_fingerprint,
            "candidate_policy": candidate_policy_fingerprint(config),
            "feature_schema": FEATURE_SCHEMA_FINGERPRINT,
            "model": artifact.model_checksum,
            "case": case_fingerprint(load_cases(cases_path)),
        },
    )
    evidence = {
        "schema_version": "ranker-diagnostics/v1",
        "config_fingerprint": lambdamart_config_fingerprint(config),
        "max_users": max_users,
        "summary": {
            "user_count": summary.user_count,
            "present_user_count": summary.present_user_count,
            "union_recall": summary.union_recall,
            "itemcf_recall": summary.itemcf_recall,
            "dense_recall": summary.dense_recall,
            "itemcf_top10_hit": summary.itemcf_top10_hit,
            "lambdamart_top10_hit": summary.lambdamart_top10_hit,
            "itemcf_top10_hit_present": summary.itemcf_top10_hit_present,
            "lambdamart_top10_hit_present": summary.lambdamart_top10_hit_present,
            "itemcf_ndcg_at_10": summary.itemcf_ndcg_at_10,
            "lambdamart_ndcg_at_10": summary.lambdamart_ndcg_at_10,
            "itemcf_ndcg_at_10_present": summary.itemcf_ndcg_at_10_present,
            "lambdamart_ndcg_at_10_present": summary.lambdamart_ndcg_at_10_present,
            "target_itemcf_rank_quantiles": summary.target_itemcf_rank_quantiles,
            "target_lambdamart_rank_quantiles": (
                summary.target_lambdamart_rank_quantiles
            ),
            "feature_separation": summary.feature_separation,
        },
        "fingerprints": summary.fingerprints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(evidence["summary"], indent=2, sort_keys=True))


@app.command("diagnose-latent")
def diagnose_latent(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    cases_path: Annotated[Path, typer.Option("--cases")] = Path(
        "cases/fixed_cases.json"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-latent-diagnostics/diagnostics.json"
    ),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
) -> None:
    """Write read-only latent-route candidate diagnostics for validation users."""
    config = _validated_config(config_path)
    if not config.latent_enabled:
        raise typer.BadParameter("diagnose-latent requires latent.enabled=true")
    if config.ranker_feature_version not in {"v2", "v2b"}:
        raise typer.BadParameter(
            "diagnose-latent requires ranker.feature_version v2 or v2b"
        )
    if output.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing diagnostics artifact: {output}"
        )
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    try:
        if config.semantic_kind == "dense":
            if config.semantic_cache_path is None:
                raise ValueError(
                    "semantic.cache_path is required for dense diagnostics"
                )
            semantic = DenseSemanticRetriever.load(
                Path(config.semantic_cache_path),
                movies=movies,
                model_name=config.semantic_model_name,
                model_revision=config.semantic_model_revision,
                device=config.semantic_device,
            )
        else:
            semantic = TfidfSemanticRetriever.fit(movies)
        started = time.perf_counter()
        latent = LatentFactorRetriever.fit(
            split.legal_retrieval_train,
            rank=config.latent_rank,
            iterations=config.latent_iterations,
            alpha=config.latent_alpha,
            lambda_reg=config.latent_lambda_reg,
            seed=config.latent_seed,
        )
        fit_seconds = time.perf_counter() - started
        queries = build_latent_diagnostic_queries(
            movies,
            split,
            semantic,
            latent=latent,
            retrieval_top_k=config.retrieval_top_k,
            history_cap=config.semantic_profile_history_cap,
            semantic_top_k=config.semantic_top_k,
            latent_top_k=config.latent_top_k,
            feature_version=config.ranker_feature_version,
            max_users=max_users,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    rows = build_latent_user_rows(queries)
    summary = aggregate_latent_diagnostics(
        rows,
        fingerprints={
            "dataset": ranking_dataset_fingerprint(movies, split),
            "diagnostic_dataset": ranker_dataset_fingerprint(
                movies,
                split,
                max_users=max_users,
                retrieval_top_k=config.retrieval_top_k,
                history_cap=config.semantic_profile_history_cap,
            ),
            "candidate_policy": candidate_policy_fingerprint(config),
            "feature_schema": FEATURE_SCHEMA_FINGERPRINT_V2,
            "case": case_fingerprint(load_cases(cases_path)),
        },
        fit_seconds=fit_seconds,
    )
    evidence = {
        "schema_version": "latent-diagnostics/v1",
        "config_fingerprint": lambdamart_config_fingerprint(config),
        "max_users": max_users,
        "summary": {
            "user_count": summary.user_count,
            "latent_present_user_count": summary.latent_present_user_count,
            "latent_recall_500_all": summary.latent_recall_500_all,
            "latent_recall_100_all": summary.latent_recall_100_all,
            "latent_recall_50_all": summary.latent_recall_50_all,
            "latent_recall_10_all": summary.latent_recall_10_all,
            "latent_recall_500_present": summary.latent_recall_500_present,
            "latent_recall_10_present": summary.latent_recall_10_present,
            "union_recall_three_route": summary.union_recall_three_route,
            "latent_only_coverage": summary.latent_only_coverage,
            "target_latent_rank_quantiles": summary.target_latent_rank_quantiles,
            "overlap_itemcf_latent": summary.overlap_itemcf_latent,
            "overlap_dense_latent": summary.overlap_dense_latent,
            "fit_seconds": summary.fit_seconds,
        },
        "fingerprints": summary.fingerprints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(evidence["summary"], indent=2, sort_keys=True))


@app.command("build-cohorts")
def build_cohorts(
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    cases_path: Annotated[Path, typer.Option("--cases")] = Path(
        "cases/fixed_cases.json"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/cohorts/cohort_ledger.json"
    ),
    seed: Annotated[int, typer.Option()] = 42,
    development_size: Annotated[int, typer.Option()] = 600,
    confirmation_a_size: Annotated[int, typer.Option()] = 1000,
    confirmation_b_size: Annotated[int, typer.Option()] = 1000,
) -> None:
    """Build the fixed, mutually exclusive development/confirmation cohorts."""
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing cohort ledger: {output}")
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    eligible = sorted(split.validation_targets)
    historical = set(eligible[:500])
    frozen_users = {case.user_id for case in load_cases(cases_path)}
    ledger = build_cohort_ledger(
        eligible,
        historical=historical,
        excluded=frozen_users,
        sizes={
            "development": development_size,
            "confirmation_a": confirmation_a_size,
            "confirmation_b": confirmation_b_size,
        },
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        json.dumps({"fingerprint": ledger["fingerprint"], "sizes": ledger["sizes"]})
    )


@app.command("evaluate-baselines")
def evaluate_baselines(
    ledger_path: Annotated[Path, typer.Option("--ledger")],
    cohort: Annotated[str, typer.Option()],
    method: Annotated[str, typer.Option()],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-baselines/result.json"
    ),
    max_users: Annotated[int, typer.Option(min=1)] = 1000,
    max_training_users: Annotated[int | None, typer.Option()] = None,
    locked_params_path: Annotated[Path | None, typer.Option("--locked-params")] = None,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Evaluate one registered baseline method on one cohort."""
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing baseline artifact: {output}")
    if cohort not in {"development", "confirmation_a", "confirmation_b"}:
        raise typer.BadParameter("cohort must be development, confirmation_a, or confirmation_b")
    if method not in BASELINE_SCORERS:
        raise typer.BadParameter(
            f"unknown baseline method {method!r}; registered: {sorted(BASELINE_SCORERS)}"
        )
    if seed != 42 and locked_params_path is None:
        raise typer.BadParameter("non-default seed requires --locked-params")
    try:
        ledger = json.loads(ledger_path.read_text())
        users = [int(user) for user in ledger["cohorts"][cohort][:max_users]]
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(f"invalid cohort ledger: {exc}") from exc
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    scorer = BASELINE_SCORERS[method]
    scorer_kwargs: dict[str, object] = {
        "ledger": ledger,
        "max_training_users": max_training_users,
    }
    selected_params_provenance = None
    parameter_grid_provenance = None
    if locked_params_path is not None:
        if method not in {"bpr_mf", "lightgcn"}:
            raise typer.BadParameter("locked params are supported only for BPR-MF and LightGCN")
        try:
            recovery_manifest = json.loads(locked_params_path.read_text())
            recorded = recovery_manifest.pop("fingerprint")
            if recorded != canonical_digest(recovery_manifest):
                raise ValueError("parameter recovery fingerprint drift")
            recovery_manifest["fingerprint"] = recorded
            method_recovery = recovery_manifest["cohorts"][cohort][method]
            selected_params_provenance = method_recovery["selected_params"]
            parameter_grid_provenance = method_recovery["parameter_grid"]
            scorer_kwargs.update(
                {
                    "selected_params": selected_params_provenance["value"],
                    "selection_fingerprint": method_recovery["selection_fingerprint"],
                    "seed": seed,
                }
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(f"invalid locked params: {exc}") from exc
    try:
        result = scorer(
            movies,
            split,
            users,
            **scorer_kwargs,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    artifact = metric_json(
        result["rows"],
        method=method,
        cohort=cohort,
        universe_size=len(movies),
        config_fingerprint=result["config_fingerprint"],
        dataset_fingerprint=result["dataset_fingerprint"],
        model_fingerprint=result["model_fingerprint"],
        cohort_ledger_fingerprint=ledger["fingerprint"],
        selected_params=result["selected_params"],
        parameter_grid=result["parameter_grid"],
        seed=result["seed"],
        dependency_versions=runtime_dependency_versions(),
        hardware=runtime_hardware(),
        training_seconds=result["training_seconds"],
        resource_usage=result["resource_usage"],
        model_size_bytes=result["model_size_bytes"],
        environment=result["environment"],
        bootstrap=result.get("bootstrap_vs_itemcf"),
        selected_params_provenance=selected_params_provenance,
        parameter_grid_provenance=parameter_grid_provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(artifact["aggregates"], indent=2, sort_keys=True))


@app.command("summarize-baselines")
def summarize_baselines_cli(
    cohort: Annotated[str, typer.Option()],
    ledger_path: Annotated[Path, typer.Option("--ledger")] = Path(
        "reports/audit/2026-08-23-cohort-ledger.json"
    ),
    artifact_dir: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-baselines"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "reports/experiments/v2-strong-baselines-confirmation-a.json"
    ),
) -> None:
    """Summarize all baseline artifacts for one cohort with paired bootstrap."""
    if cohort not in {"development", "confirmation_a", "confirmation_b"}:
        raise typer.BadParameter(
            "cohort must be development, confirmation_a, or confirmation_b"
        )
    if output.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing baseline summary: {output}"
        )
    methods = ["popularity", "itemcf_direct", "als_direct", "current_v2b", "bpr_mf", "lightgcn"]
    artifacts: dict[str, object] = {}
    file_cohort = cohort.replace("_", "-")
    for method in methods:
        file_method = method.replace("_", "-")
        path = artifact_dir / f"{file_method}-{file_cohort}.json"
        if not path.exists():
            raise typer.BadParameter(f"missing baseline artifact: {path}")
        try:
            artifacts[method] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(f"invalid baseline artifact {path}: {exc}") from exc
    try:
        ledger = json.loads(ledger_path.read_text())
        expected_user_ids = [int(user) for user in ledger["cohorts"][cohort]]
        ledger_fingerprint = str(ledger["fingerprint"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise typer.BadParameter(f"invalid cohort ledger: {exc}") from exc
    summary = summarize_baselines(
        artifacts,
        cohort=cohort,
        expected_user_ids=expected_user_ids,
        cohort_ledger_fingerprint=ledger_fingerprint,
    )
    md_path = output.with_suffix(".md")
    if md_path.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing baseline summary: {md_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(summary_to_markdown(summary), encoding="utf-8")
    typer.echo(json.dumps(summary["aggregates"], indent=2, sort_keys=True))


@app.command("build-evidence-bundle")
def build_evidence_bundle_cli(
    cohort: Annotated[str, typer.Option()],
    ledger_path: Annotated[Path, typer.Option("--ledger")],
    artifact_dir: Annotated[Path, typer.Option()],
    summary_path: Annotated[Path, typer.Option("--summary")],
    recovery_path: Annotated[Path, typer.Option("--recovery")],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Build a compact, provenance-bound baseline replay bundle."""
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing evidence: {output}")
    try:
        ledger_bytes = ledger_path.read_bytes()
        summary_bytes = summary_path.read_bytes()
        summary = json.loads(summary_bytes)
        recovery = json.loads(recovery_path.read_bytes())
        if recovery.get("schema_version") == "baseline-parameter-recovery/v1":
            recovery = recovery["cohorts"][cohort]
        methods = sorted(summary["aggregates"])
        file_cohort = cohort.replace("_", "-")
        source_artifacts = {
            method: (
                artifact_dir
                / f"{method.replace('_', '-')}-{file_cohort}.json"
            ).read_bytes()
            for method in methods
        }
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        bundle = build_compact_bundle(
            source_artifacts=source_artifacts,
            ledger_bytes=ledger_bytes,
            summary_bytes=summary_bytes,
            recovery=recovery,
            cohort=cohort,
            commit_sha=commit_sha,
        )
        write_new_json(output, bundle)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps({"fingerprint": bundle["fingerprint"], "output": str(output)}))


@app.command("recover-baseline-params")
def recover_baseline_params_cli(
    ledger_path: Annotated[Path, typer.Option("--ledger")],
    artifact_dir: Annotated[Path, typer.Option()],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path(
        "reports/evidence/baseline-parameter-recovery.json"
    ),
) -> None:
    """Deterministically recover v1 selected parameters once for evidence binding."""
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing evidence: {output}")
    try:
        ledger = json.loads(ledger_path.read_text())
        dev_users = [int(user) for user in ledger["cohorts"]["development"]]
        movies, ratings = _load_dataset(data_dir)
        split = leakage_safe_ranking_split(ratings)
        als_selection = _als_registration.select_als_params(movies, split, dev_users)
        bpr_selection = _bpr_registration.select_bpr_params(movies, split, dev_users)
        lightgcn_selection = _lightgcn_registration.select_lightgcn_params(
            movies, split, dev_users
        )
        lightgcn_selection["selected_params"] = {
            **lightgcn_selection["selected_params"],
            "epochs": int(lightgcn_selection["epochs"]),
        }
        lightgcn_selection["grid"] = [
            {**params, "epochs": int(lightgcn_selection["epochs"])}
            for params in lightgcn_selection["grid"]
        ]
        current_config = load_experiment_config(
            Path("configs/v2_dense_latent_bfeat.yaml")
        )
        selections = {
            "popularity": {
                "selected_params": {},
                "grid": [],
                "fingerprint": hashlib.sha256(b"popularity/v1").hexdigest(),
                "seed": "not_applicable",
            },
            "itemcf_direct": {
                "selected_params": {},
                "grid": [],
                "fingerprint": hashlib.sha256(b"itemcf_direct/v1").hexdigest(),
                "seed": "not_applicable",
            },
            "als_direct": als_selection,
            "bpr_mf": bpr_selection,
            "lightgcn": lightgcn_selection,
            "current_v2b": {
                "selected_params": asdict(current_config),
                "grid": [],
                "fingerprint": lambdamart_config_fingerprint(current_config),
                "seed": current_config.seed,
            },
        }
        source_artifacts = {
            cohort: {
                method: (
                    artifact_dir
                    / f"{method.replace('_', '-')}-{cohort.replace('_', '-')}.json"
                ).read_bytes()
                for method in selections
            }
            for cohort in ("confirmation_a", "confirmation_b")
        }
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        command = (
            ".venv/bin/recagent-eval recover-baseline-params "
            f"--ledger {ledger_path} --artifact-dir {artifact_dir} "
            f"--data-dir {data_dir} --output {output}"
        )
        manifest = build_parameter_recovery_manifest(
            selections=selections,
            source_artifacts=source_artifacts,
            command=command,
            commit_sha=commit_sha,
        )
        write_new_json(output, manifest)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps({"fingerprint": manifest["fingerprint"], "output": str(output)}))


@app.command("replay-evidence")
def replay_evidence_cli(
    bundle_path: Annotated[Path, typer.Option("--bundle")],
    ledger_path: Annotated[Path, typer.Option("--ledger")],
    summary_path: Annotated[Path, typer.Option("--summary")],
) -> None:
    """Replay aggregate metrics and every paired bootstrap from a compact bundle."""
    try:
        bundle = json.loads(bundle_path.read_bytes())
        replayed = replay_compact_bundle(
            bundle,
            ledger_bytes=ledger_path.read_bytes(),
            summary_bytes=summary_path.read_bytes(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "cohort": replayed["cohort"],
                "user_count": replayed["user_count"],
                "fingerprint": replayed["fingerprint"],
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-ranker")
def evaluate_ranker(
    config_path: Annotated[Path, typer.Option("--config")],
    evidence_path: Annotated[
        Path,
        typer.Option(
            "--evidence",
            help="Validation evidence; registered LambdaMART frozen identity is consumed once",
        ),
    ],
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


@app.command("preflight-frozen-promotion")
def preflight_frozen_promotion(
    promotion_path: Annotated[
        Path, typer.Option("--promotion", help="Promotion execution YAML")
    ] = Path("reports/promotion/current-v2b.yaml"),
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
) -> object:
    repo_root = Path.cwd().resolve()
    resolved_promotion = (
        promotion_path
        if promotion_path.is_absolute()
        else repo_root / promotion_path
    )
    context: dict[str, object] = {}

    def dataset_check(
        manifest: PromotionManifest, _member_paths: dict[str, Path]
    ) -> str:
        config = load_experiment_config(repo_root / manifest.training_config_path)
        if lambdamart_config_fingerprint(config) != manifest.training_config_fingerprint:
            raise ValueError("promotion training config fingerprint mismatch")
        if candidate_policy_fingerprint(config) != manifest.candidate_policy_fingerprint:
            raise ValueError("promotion candidate-policy fingerprint mismatch")
        contract = (
            config.ranker_feature_version,
            config.score_calibration,
            config.retrieval_top_k,
            config.semantic_top_k,
            config.latent_top_k,
        )
        expected = (
            manifest.feature_version,
            manifest.score_calibration,
            manifest.itemcf_top_k,
            manifest.semantic_top_k,
            manifest.latent_top_k,
        )
        if contract != expected:
            raise ValueError("promotion training/candidate/feature contract mismatch")
        movies, ratings = _load_dataset(data_dir)
        split = leakage_safe_ranking_split(ratings)
        fingerprint = ranking_dataset_fingerprint(movies, split)
        context.update(config=config, movies=movies, split=split)
        return fingerprint

    def validation_replay(
        manifest: PromotionManifest, member_paths: dict[str, Path]
    ) -> ReplayVerification:
        config = context["config"]
        movies = context["movies"]
        split = context["split"]
        bundle = load_ranker_bundle(
            member_paths["model.json"],
            member_paths["validation.json"],
            member_paths["bundle.json"],
            expected_metadata={
                "config_fingerprint": manifest.training_config_fingerprint,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "candidate_policy_fingerprint": manifest.candidate_policy_fingerprint,
                "feature_fingerprint": manifest.feature_fingerprint,
            },
            latent_path=member_paths["latent.npz"],
            latent_manifest_path=member_paths["latent.npz.json"],
        )
        artifact = parse_ranker_artifact(
            bundle.model_bytes,
            expected_dataset_fingerprint=manifest.dataset_fingerprint,
            expected_feature_fingerprint=manifest.feature_fingerprint,
            expected_candidate_policy_fingerprint=manifest.candidate_policy_fingerprint,
            expected_config_fingerprint=manifest.training_config_fingerprint,
            expected_case_fingerprint=manifest.case_fingerprint,
            expected_latent_artifact_checksum=manifest.members["latent.npz"].sha256,
        )
        if (
            artifact.feature_fingerprint != manifest.feature_fingerprint
            or artifact.model_checksum != manifest.model_checksum
        ):
            raise ValueError("promotion model feature/checksum identity mismatch")
        evidence = LearnedValidationEvidence.model_validate_json(bundle.evidence_bytes)
        ordered = tuple(int(row["user_id"]) for row in evidence.per_user_rows)
        if ordered != manifest.ordered_user_ids:
            raise ValueError("promotion evidence ordered users mismatch")
        semantic = DenseSemanticRetriever.load(
            member_paths["semantic.npz"],
            movies=movies,
            model_name=manifest.semantic.model_name,
            model_revision=manifest.semantic.immutable_revision,
            device="cpu",
        )
        latent = LatentFactorRetriever.load(
            member_paths["latent.npz"],
            expected_training_fingerprint=str(
                (artifact.latent_provenance or {})["training_fingerprint"]
            ),
        )
        ranker = LearnedRanker(
            estimator_from_artifact(artifact),
            legal_train_rows=split.legal_retrieval_train,
            score_calibration=manifest.score_calibration,
            feature_version=manifest.feature_version,
        )
        replay_rows = build_validation_rows(
            movies,
            split,
            semantic,
            config,
            ranker,
            max_users=len(ordered),
            latent=latent,
            ordered_user_ids=ordered,
        )
        if _canonical_validation_contract(replay_rows) != _canonical_validation_contract(
            evidence.per_user_rows
        ):
            raise ValueError("complete Confirmation-B validation replay mismatch")
        recorded_rows_fingerprint = hashlib.sha256(
            json.dumps(
                evidence.per_user_rows, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if recorded_rows_fingerprint != artifact.validation_rows_fingerprint:
            raise ValueError("recorded validation evidence fingerprint mismatch")
        validate_learned_gate(
            evidence,
            dataset_fingerprint=manifest.dataset_fingerprint,
            feature_fingerprint=manifest.feature_fingerprint,
            model_fingerprint=manifest.model_checksum,
            candidate_policy_fingerprint=manifest.candidate_policy_fingerprint,
            case_fingerprint=manifest.case_fingerprint,
            config_fingerprint=manifest.training_config_fingerprint,
            artifact_provenance=artifact.model_dump(mode="python"),
        )
        stable_fingerprint = hashlib.sha256(
            _canonical_validation_contract(replay_rows).encode()
        ).hexdigest()
        return ReplayVerification(
            ordered_user_ids=ordered,
            dataset_fingerprint=manifest.dataset_fingerprint,
            model_checksum=artifact.model_checksum,
            validation_rows_fingerprint=stable_fingerprint,
            validated_components=(
                "model",
                "evidence",
                "bundle",
                "latent",
                "semantic",
            ),
        )

    try:
        receipt = preflight_promotion(
            repo_root,
            resolved_promotion,
            dataset_fingerprint_check=dataset_check,
            validation_replay=validation_replay,
            git_identity_check=lambda manifest: verify_git_identity(
                repo_root, manifest
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(receipt.model_dump_json(indent=2))
    return receipt


@app.command("prepare-frozen-promotion")
def prepare_frozen_promotion(
    inventory_path: Annotated[
        Path, typer.Option("--inventory", help="Locked source inventory")
    ] = Path("reports/promotion/current-v2b-source-inventory.json"),
    confirmation_source: Annotated[
        Path,
        typer.Option("--confirmation-source", help="Original Confirmation-B bundle directory"),
    ] = Path("artifacts/promotion-source/current-v2b"),
    semantic_cache: Annotated[
        Path, typer.Option("--semantic-cache", help="Original dense NPZ cache")
    ] = Path("artifacts/embeddings/movielens-minilm.npz"),
) -> None:
    try:
        inventory = load_source_inventory(inventory_path)
        sources = {
            name: confirmation_source / name
            for name in (
                "model.json",
                "validation.json",
                "bundle.json",
                "latent.npz",
                "latent.npz.json",
            )
        }
        sources.update(
            {
                "semantic.npz": semantic_cache,
                "semantic.npz.json": Path(f"{semantic_cache}.json"),
            }
        )
        destination = publish_promotion_package(Path.cwd(), inventory, sources)
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Promotion package published atomically at {destination}")


def _load_frozen_execution_runtime(
    repo_root: Path,
    manifest: PromotionManifest,
    data_dir: Path,
) -> dict[str, object]:
    config = load_experiment_config(repo_root / manifest.training_config_path)
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    member_paths = {
        name: repo_root / identity.path for name, identity in manifest.members.items()
    }
    bundle = load_ranker_bundle(
        member_paths["model.json"],
        member_paths["validation.json"],
        member_paths["bundle.json"],
        expected_metadata={
            "config_fingerprint": manifest.training_config_fingerprint,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "candidate_policy_fingerprint": manifest.candidate_policy_fingerprint,
            "feature_fingerprint": manifest.feature_fingerprint,
        },
        latent_path=member_paths["latent.npz"],
        latent_manifest_path=member_paths["latent.npz.json"],
    )
    artifact = parse_ranker_artifact(
        bundle.model_bytes,
        expected_dataset_fingerprint=manifest.dataset_fingerprint,
        expected_feature_fingerprint=manifest.feature_fingerprint,
        expected_candidate_policy_fingerprint=manifest.candidate_policy_fingerprint,
        expected_config_fingerprint=manifest.training_config_fingerprint,
        expected_case_fingerprint=manifest.case_fingerprint,
        expected_latent_artifact_checksum=manifest.members["latent.npz"].sha256,
    )
    if (
        artifact.feature_fingerprint != manifest.feature_fingerprint
        or artifact.model_checksum != manifest.model_checksum
    ):
        raise ValueError("promotion model identity drift before execution")
    evidence = LearnedValidationEvidence.model_validate_json(bundle.evidence_bytes)
    semantic = DenseSemanticRetriever.load(
        member_paths["semantic.npz"],
        movies=movies,
        model_name=manifest.semantic.model_name,
        model_revision=manifest.semantic.immutable_revision,
        device="cpu",
    )
    latent = LatentFactorRetriever.load(
        member_paths["latent.npz"],
        expected_training_fingerprint=str(
            (artifact.latent_provenance or {})["training_fingerprint"]
        ),
    )
    ranker = LearnedRanker(
        estimator_from_artifact(artifact),
        legal_train_rows=split.legal_retrieval_train,
        score_calibration=manifest.score_calibration,
        feature_version=manifest.feature_version,
    )
    return {
        "config": config,
        "movies": movies,
        "split": split,
        "semantic": semantic,
        "latent": latent,
        "ranker": ranker,
        "evidence": evidence,
    }


@app.command("run-frozen-promotion")
def run_frozen_promotion(
    authorized_manifest_sha: Annotated[
        str,
        typer.Option(
            "--authorized-manifest-sha",
            help="Exact manifest SHA named by the user's one-time authorization",
        ),
    ],
    promotion_path: Annotated[
        Path, typer.Option("--promotion", help="Promotion execution YAML")
    ] = Path("reports/promotion/current-v2b.yaml"),
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
) -> None:
    repo_root = Path.cwd().resolve()
    resolved_promotion = (
        promotion_path if promotion_path.is_absolute() else repo_root / promotion_path
    )
    try:
        manifest, promotion = load_promotion_documents(repo_root, resolved_promotion)
        if authorized_manifest_sha != promotion.manifest_sha256:
            raise ValueError("real execution requires exact manifest SHA authorization")
        receipt = preflight_frozen_promotion(resolved_promotion, data_dir)
        runtime = _load_frozen_execution_runtime(repo_root, manifest, data_dir)

        def case_loader():
            cases = load_cases(repo_root / manifest.frozen_cases_path)
            if case_fingerprint(cases) != manifest.case_fingerprint:
                raise ValueError("registered frozen case fingerprint mismatch")
            return cases

        def evaluator(cases):
            config = runtime["config"]
            metrics = evaluate_frozen_cases(
                runtime["movies"],
                runtime["split"].legal_retrieval_train,
                cases,
                ranker=runtime["ranker"],
                retrieval_top_k=config.retrieval_top_k,
                semantic_top_k=config.semantic_top_k,
                history_cap=config.semantic_profile_history_cap,
                semantic_retriever=runtime["semantic"],
                latent_retriever=runtime["latent"],
                latent_top_k=config.latent_top_k,
                feature_version=config.ranker_feature_version,
            )
            metrics.update(
                {
                    "manifest_sha256": promotion.manifest_sha256,
                    "case_fingerprint": manifest.case_fingerprint,
                    "dataset_fingerprint": manifest.dataset_fingerprint,
                    "model_checksum": manifest.model_checksum,
                    "selection_evidence_fingerprint": runtime[
                        "evidence"
                    ].evidence_fingerprint,
                }
            )
            return metrics

        marker = execute_one_shot(
            repo_root,
            manifest,
            promotion,
            receipt,
            case_loader=case_loader,
            evaluator=evaluator,
            authorized_manifest_sha=authorized_manifest_sha,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(marker.model_dump_json(indent=2))


@app.command("audit-frozen-promotion")
def audit_frozen_promotion(
    promotion_path: Annotated[
        Path, typer.Option("--promotion", help="Promotion execution YAML")
    ] = Path("reports/promotion/current-v2b.yaml"),
) -> None:
    repo_root = Path.cwd().resolve()
    resolved_promotion = (
        promotion_path if promotion_path.is_absolute() else repo_root / promotion_path
    )
    try:
        manifest, promotion = load_promotion_documents(repo_root, resolved_promotion)
        audit = audit_one_shot(repo_root, manifest, promotion)
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(audit, indent=2, sort_keys=True))


def _evaluate_learned_ranker(
    *,
    config: ExperimentConfig,
    evidence_path: Path,
    cases_path: Path,
    data_dir: Path,
    output: Path,
) -> None:
    feature_fingerprint = {
        "v1": FEATURE_SCHEMA_FINGERPRINT,
        "v2": FEATURE_SCHEMA_FINGERPRINT_V2,
        "v2b": FEATURE_SCHEMA_FINGERPRINT_V2B,
    }[config.ranker_feature_version]
    if config.learned_model_path is None:
        raise typer.BadParameter("ranker.model_path is required when ranker.kind is lambdamart")
    if config.learned_bundle_manifest_path is None:
        raise typer.BadParameter(
            "ranker.bundle_manifest_path is required when ranker.kind is lambdamart"
        )
    if (
        config.learned_dataset_fingerprint is None
        or config.learned_config_fingerprint is None
        or config.learned_case_fingerprint is None
        or config.learned_candidate_policy_fingerprint is None
        or config.learned_gate_fingerprint is None
        or config.learned_consumption_dir is None
    ):
        raise typer.BadParameter(
            "LambdaMART frozen evaluation requires registered dataset/config/case "
            "fingerprints and ranker.consumption_dir"
        )
    try:
        marker_path = consumption_marker_path(
            Path(config.learned_consumption_dir),
            case_fingerprint=config.learned_case_fingerprint,
            dataset_fingerprint=config.learned_dataset_fingerprint,
            config_fingerprint=config.learned_config_fingerprint,
        )
        assert_frozen_authorization_available(
            marker_path, case_fingerprint=config.learned_case_fingerprint
        )
        frozen_paths = {
            "model": Path(config.learned_model_path),
            "evidence": evidence_path,
            "bundle manifest": Path(config.learned_bundle_manifest_path),
            "frozen cases": cases_path,
            "dataset directory": data_dir,
            "frozen output": output,
            "consumption marker": marker_path,
        }
        if config.semantic_cache_path is not None:
            frozen_paths["semantic cache"] = Path(config.semantic_cache_path)
        ensure_distinct_files(frozen_paths)
        latent_kwargs: dict[str, Path] = {}
        if config.latent_enabled and config.latent_artifact_path is not None:
            latent_kwargs = {
                "latent_path": Path(config.latent_artifact_path),
                "latent_manifest_path": Path(f"{config.latent_artifact_path}.json"),
            }
        bundle = load_ranker_bundle(
            Path(config.learned_model_path),
            evidence_path,
            Path(config.learned_bundle_manifest_path),
            expected_metadata={
                "run_fingerprint": config.learned_gate_fingerprint,
                "config_fingerprint": config.learned_config_fingerprint,
                "dataset_fingerprint": config.learned_dataset_fingerprint,
                "candidate_policy_fingerprint": config.learned_candidate_policy_fingerprint,
                "feature_fingerprint": feature_fingerprint,
            },
            **latent_kwargs,
        )
        model_bytes = bundle.model_bytes
        evidence_bytes = bundle.evidence_bytes
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        evidence = LearnedValidationEvidence.model_validate_json(evidence_bytes)
        if evidence.evidence_fingerprint != config.learned_gate_fingerprint:
            raise ValueError("LambdaMART validation evidence fingerprint mismatch")
        movies, ratings = _load_dataset(data_dir)
        split = leakage_safe_ranking_split(ratings)
        dataset_fingerprint = ranking_dataset_fingerprint(movies, split)
        fixed_case_fingerprint = config.learned_case_fingerprint
        artifact = parse_ranker_artifact(
            model_bytes,
            expected_dataset_fingerprint=dataset_fingerprint,
            expected_candidate_policy_fingerprint=candidate_policy_fingerprint(config),
            expected_config_fingerprint=lambdamart_config_fingerprint(config),
            expected_case_fingerprint=fixed_case_fingerprint,
            expected_feature_fingerprint=feature_fingerprint,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    ranker = LearnedRanker(
        estimator_from_artifact(artifact),
        legal_train_rows=split.legal_retrieval_train,
        score_calibration=config.score_calibration,
        feature_version=config.ranker_feature_version,
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
    latent = None
    if config.latent_enabled:
        if config.latent_artifact_path is None:
            raise typer.BadParameter(
                "latent.artifact_path is required for frozen LambdaMART evaluation"
            )
        try:
            latent = LatentFactorRetriever.load(
                Path(config.latent_artifact_path),
                expected_training_fingerprint=str(
                    (artifact.latent_provenance or {})["training_fingerprint"]
                ),
            )
        except (KeyError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    try:
        ordered_user_ids = tuple(
            int(row["user_id"]) for row in evidence.per_user_rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "LambdaMART validation evidence has invalid ordered user IDs"
        ) from exc
    if len(ordered_user_ids) != artifact.validation_user_count:
        raise typer.BadParameter(
            "LambdaMART validation evidence user count does not match artifact"
        )
    replay_rows = build_validation_rows(
        movies,
        split,
        semantic,
        config,
        ranker,
        max_users=artifact.validation_user_count,
        ordered_user_ids=ordered_user_ids,
    )
    canonical_replay = _canonical_validation_contract(replay_rows)
    canonical_evidence = _canonical_validation_contract(evidence.per_user_rows)
    recorded_evidence_fingerprint = hashlib.sha256(
        json.dumps(
            evidence.per_user_rows, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        canonical_replay != canonical_evidence
        or recorded_evidence_fingerprint != artifact.validation_rows_fingerprint
    ):
        raise typer.BadParameter(
            "LambdaMART validation replay does not match recorded per-user evidence"
        )
    try:
        validate_learned_gate(
            evidence,
            dataset_fingerprint=dataset_fingerprint,
            feature_fingerprint=feature_fingerprint,
            model_fingerprint=artifact.model_checksum,
            candidate_policy_fingerprint=candidate_policy_fingerprint(config),
            case_fingerprint=fixed_case_fingerprint,
            config_fingerprint=lambdamart_config_fingerprint(config),
            artifact_provenance=artifact.model_dump(mode="python"),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        consume_frozen_authorization(
            marker_path,
            evidence_hash=hashlib.sha256(evidence_bytes).hexdigest(),
            case_fingerprint=config.learned_case_fingerprint,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    cases = load_cases(cases_path)
    if case_fingerprint(cases) != fixed_case_fingerprint:
        raise typer.BadParameter("registered frozen case fingerprint mismatch")
    metrics = evaluate_frozen_cases(
        movies,
        split.legal_retrieval_train,
        cases,
        ranker=ranker,
        retrieval_top_k=config.retrieval_top_k,
        semantic_top_k=config.semantic_top_k,
        history_cap=config.semantic_profile_history_cap,
        semantic_retriever=semantic,
        latent_retriever=latent,
        latent_top_k=config.latent_top_k,
        feature_version=config.ranker_feature_version,
    )
    metrics.update(
        {
            "selection_margin": evidence.mean_ndcg_delta,
            "selection_evidence_fingerprint": evidence.evidence_fingerprint,
            "case_fingerprint": fixed_case_fingerprint,
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
        str, typer.Option("--provider", help="rule-based, deepseek, vllm, or qwen")
    ] = "rule-based",
) -> None:
    config = _validated_config(config_path)
    if config.ranker_kind == "lambdamart":
        raise typer.BadParameter(
            "generic evaluate cannot authorize LambdaMART; use evaluate-ranker "
            "with validation evidence and its one-time frozen-test marker"
        )
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


@app.command("demo")
def demo_command(
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    provider_name: Annotated[
        str | None,
        typer.Option("--provider", help="rule-based, deepseek, vllm, or qwen"),
    ] = None,
    semantic_config_path: Annotated[
        Path | None,
        typer.Option("--semantic-config", help="Semantic retrieval config YAML"),
    ] = None,
    ranker_config_path: Annotated[
        Path | None,
        typer.Option("--ranker-config", help="Demo ranker config YAML"),
    ] = None,
) -> None:
    from recagent_eval.demo import launch

    launch(
        data_dir,
        provider_name=provider_name,
        semantic_config_path=semantic_config_path,
        ranker_config_path=ranker_config_path,
    )


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
    try:
        return build_provider(name).provider
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()

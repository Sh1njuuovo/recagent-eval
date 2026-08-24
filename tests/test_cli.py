import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from recagent_eval.baseline_eval import MetricRow
from recagent_eval.cases import EvaluationCase
from recagent_eval.cli import app
from recagent_eval.data import Movie, Rating
from recagent_eval.evidence import canonical_digest
from recagent_eval.models import PreferenceState
from recagent_eval.v2_selection import (
    consume_frozen_authorization,
    consumption_marker_path,
)


def test_smoke_command_runs_offline_end_to_end(tmp_path) -> None:
    output = tmp_path / "smoke"
    result = CliRunner().invoke(app, ["smoke", "--output", str(output)])

    assert result.exit_code == 0, result.output
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["episodes"] == 1
    assert metrics["tool_success_rate"] == 1.0
    assert "Offline smoke test passed" in result.output


def test_show_config_rejects_invalid_weight_sum(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "name: bad\nweights: [0.8, 0.8, 0.8]\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["show-config", str(config)])

    assert result.exit_code != 0
    assert "sum to 1" in result.output


def test_show_config_includes_retrieval_policy(tmp_path) -> None:
    config = tmp_path / "hybrid.yaml"
    config.write_text(
        "\n".join(
            [
                "name: hybrid",
                "required_retrieval_tools: [itemcf_retrieve, semantic_retrieve]",
                "semantic_profile_history_cap: 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["show-config", str(config)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["required_retrieval_tools"] == [
        "itemcf_retrieve",
        "semantic_retrieve",
    ]
    assert payload["semantic_profile_history_cap"] == 20


def test_train_ranker_cli_smoke_uses_offline_semantic_retriever(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "ranker.yaml"
    config.write_text("seed: 73\nsemantic:\n  kind: tfidf\n")
    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id}", ("Drama",), 2000)
        for movie_id in range(1, 5)
    }
    ratings = [
        Rating(user, movie_id, 5, movie_id)
        for user in range(1, 4)
        for movie_id in range(1, 5)
    ]
    monkeypatch.setattr(
        "recagent_eval.cli._load_dataset", lambda _path: (movies, ratings)
    )
    seen = {}

    def fake_train(*args, **kwargs):
        seen.update(kwargs)
        return {"training_users": 3}

    monkeypatch.setattr("recagent_eval.cli.train_lambdamart_pipeline", fake_train)

    result = CliRunner().invoke(
        app,
        [
            "train-ranker",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--cases",
            str(cases),
            "--output",
            str(tmp_path / "model.json"),
            "--evidence-output",
            str(tmp_path / "evidence.json"),
            "--max-users",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["training_users"] == 3
    assert seen["max_users"] == 3
    assert seen["seed"] == 73


def test_train_ranker_has_no_ambiguous_cli_seed_override(tmp_path) -> None:
    config = tmp_path / "ranker.yaml"
    config.write_text("seed: 73\n")
    result = CliRunner().invoke(
        app,
        ["train-ranker", "--config", str(config), "--seed", "42"],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_generic_evaluate_rejects_lambdamart_before_loading_artifact(tmp_path) -> None:
    config = tmp_path / "learned.yaml"
    config.write_text("ranker:\n  kind: lambdamart\n  model_path: missing.json\n")
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config),
            "--cases",
            str(tmp_path / "cases.json"),
        ],
    )

    assert result.exit_code != 0
    assert "generic evaluate cannot authorize LambdaMART" in result.output


def test_formal_evaluate_fails_actionably_for_unconfigured_vllm(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: remote\n")
    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config),
            "--cases",
            str(cases),
            "--provider",
            "vllm",
        ],
    )

    assert result.exit_code != 0
    assert "VLLM_BASE_URL" in result.output
    assert "rule-based" not in result.output.lower()


def test_demo_command_forwards_provider_and_config_paths(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_launch(data_dir, **kwargs):
        seen["data_dir"] = data_dir
        seen.update(kwargs)

    monkeypatch.setattr("recagent_eval.demo.launch", fake_launch)
    semantic = tmp_path / "semantic.yaml"
    ranker = tmp_path / "ranker.yaml"

    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--data-dir",
            str(tmp_path),
            "--provider",
            "qwen",
            "--semantic-config",
            str(semantic),
            "--ranker-config",
            str(ranker),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "data_dir": tmp_path,
        "provider_name": "qwen",
        "semantic_config_path": semantic,
        "ranker_config_path": ranker,
    }


def test_consumed_frozen_identity_rejects_before_any_dataset_or_model_load(
    tmp_path, monkeypatch
) -> None:
    consumption_dir = tmp_path / "consumed"
    evidence = tmp_path / "copied-evidence.json"
    evidence.write_text("{}")
    config = tmp_path / "selected.yaml"
    config.write_text(
        f"""ranker:
  kind: lambdamart
  model_path: {tmp_path / 'missing-model.json'}
  evidence_path: {evidence}
  bundle_manifest_path: {tmp_path / 'missing-bundle.json'}
  dataset_fingerprint: dataset
  candidate_policy_fingerprint: policy
  config_fingerprint: config
  case_fingerprint: cases
  gate_fingerprint: gate
  consumption_dir: {consumption_dir}
"""
    )
    marker = consumption_marker_path(
        consumption_dir,
        case_fingerprint="cases",
        dataset_fingerprint="dataset",
        config_fingerprint="config",
    )
    consume_frozen_authorization(
        marker,
        evidence_hash=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        case_fingerprint="cases",
    )

    def forbidden_loader(_path):
        raise AssertionError("dataset loader must not run after consumed claim")

    monkeypatch.setattr("recagent_eval.cli._load_dataset", forbidden_loader)
    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(evidence),
            "--cases",
            str(tmp_path / "missing-cases.json"),
        ],
    )
    assert result.exit_code != 0
    assert "already consumed" in result.output


def _tiny_dataset():
    movies = {
        movie_id: Movie(movie_id, f"Movie {movie_id}", ("Drama",), 2000)
        for movie_id in range(1, 6)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id)
        for user_id in range(1, 4)
        for movie_id in range(1, 6)
    ]
    return movies, ratings


def test_build_embeddings_rejects_device_and_missing_dataset(tmp_path) -> None:
    invalid = CliRunner().invoke(
        app, ["build-embeddings", "--data-dir", str(tmp_path), "--device", "metal"]
    )
    missing = CliRunner().invoke(
        app, ["build-embeddings", "--data-dir", str(tmp_path)]
    )

    assert invalid.exit_code != 0
    assert "device must be cpu or cuda" in invalid.output
    assert missing.exit_code != 0
    assert "movies.dat missing" in missing.output


def test_build_embeddings_reuses_cache_and_force_rebuilds(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "movies.dat").write_text("catalog")
    output = tmp_path / "dense.npz"
    output.write_bytes(b"cache")
    movies, _ = _tiny_dataset()
    monkeypatch.setattr("recagent_eval.cli.load_movielens_movies", lambda path: movies)
    validated = []
    monkeypatch.setattr(
        "recagent_eval.cli.DenseSemanticRetriever.validate_cache",
        lambda path, **kwargs: validated.append((path, kwargs))
        or {"resolved_revision": "commit-123"},
    )

    reused = CliRunner().invoke(
        app,
        ["build-embeddings", "--data-dir", str(data_dir), "--output", str(output)],
    )

    saved = []
    retriever = SimpleNamespace(
        model_revision="commit-456", save=lambda path: saved.append(path)
    )
    monkeypatch.setattr(
        "recagent_eval.cli.DenseSemanticRetriever.fit",
        lambda *args, **kwargs: retriever,
    )
    rebuilt = CliRunner().invoke(
        app,
        [
            "build-embeddings",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--force",
        ],
    )

    assert reused.exit_code == 0, reused.output
    assert "Reused 5 embeddings" in reused.output
    assert validated[0][0] == output
    assert rebuilt.exit_code == 0, rebuilt.output
    assert saved == [output]
    assert "commit-456" in rebuilt.output


def test_prepare_cases_writes_generated_cases_and_rejects_shortfall(
    tmp_path, monkeypatch
) -> None:
    movies, ratings = _tiny_dataset()
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: (movies, ratings))
    case = EvaluationCase(
        case_id="generated",
        user_id=1,
        turns=("recommend",),
        relevant_movie_ids={5},
        initial_state=PreferenceState(liked_movie_ids={1}),
    )
    monkeypatch.setattr("recagent_eval.cli.generate_cases", lambda *args, **kwargs: [case])
    output = tmp_path / "cases.json"

    valid = CliRunner().invoke(
        app,
        [
            "prepare-cases",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--single-turn-count",
            "1",
            "--multi-turn-count",
            "0",
        ],
    )
    short = CliRunner().invoke(
        app,
        [
            "prepare-cases",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--single-turn-count",
            "2",
            "--multi-turn-count",
            "0",
        ],
    )

    assert valid.exit_code == 0, valid.output
    assert json.loads(output.read_text())[0]["case_id"] == "generated"
    assert short.exit_code != 0
    assert "not enough eligible users" in short.output


def test_tune_and_select_retrieval_write_frozen_artifacts(tmp_path, monkeypatch) -> None:
    # Render CLI errors through a fixed-width console so Typer/Rich does not wrap
    # or ellipsize the message on narrow CI runners; the message assertion below
    # must be independent of the host terminal width.
    monkeypatch.setattr(
        "typer.rich_utils._get_rich_console",
        lambda stderr=False: Console(stderr=stderr, width=200, highlight=False),
    )
    movies, ratings = _tiny_dataset()
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: (movies, ratings))
    monkeypatch.setattr(
        "recagent_eval.cli.tune_on_validation",
        lambda *args, **kwargs: (0.7, 0.2, 0.1),
    )
    config = tmp_path / "config.yaml"
    config.write_text("name: tuned\nretrieval_top_k: 9\n")
    tuned = tmp_path / "weights.json"
    tuned_config = tmp_path / "tuned.yaml"

    tune_result = CliRunner().invoke(
        app,
        [
            "tune",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tuned),
            "--config",
            str(config),
            "--config-output",
            str(tuned_config),
        ],
    )
    missing_config = CliRunner().invoke(
        app,
        [
            "tune",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "other.json"),
            "--config-output",
            str(tmp_path / "invalid.yaml"),
        ],
    )
    monkeypatch.setattr(
        "recagent_eval.cli.build_retrieval_ablation",
        lambda *args, **kwargs: [{"retrieval_top_k": 20, "ndcg_at_10": 0.4}],
    )
    monkeypatch.setattr(
        "recagent_eval.cli.select_retrieval_parameters",
        lambda *args, **kwargs: {
            "retrieval_top_k": 20,
            "semantic_profile_history_cap": 7,
        },
    )
    evidence = tmp_path / "retrieval.json"
    selected_config = tmp_path / "selected.yaml"
    select_result = CliRunner().invoke(
        app,
        [
            "select-retrieval",
            "--data-dir",
            str(tmp_path),
            "--evidence-output",
            str(evidence),
            "--config-output",
            str(selected_config),
        ],
    )

    assert tune_result.exit_code == 0, tune_result.output
    assert json.loads(tuned.read_text())["weights"] == [0.7, 0.2, 0.1]
    assert yaml.safe_load(tuned_config.read_text())["weights"] == [0.7, 0.2, 0.1]
    assert missing_config.exit_code != 0
    assert "requires --config" in missing_config.output
    assert not (tmp_path / "invalid.yaml").exists()
    assert select_result.exit_code == 0, select_result.output
    assert json.loads(evidence.read_text())["selection"]["retrieval_top_k"] == 20
    assert yaml.safe_load(selected_config.read_text())["semantic_profile_history_cap"] == 7


@pytest.mark.parametrize(
    ("candidate_row", "expected_ranker"),
    [
        (
            {
                "kind": "rrf",
                "parameters": {"rrf_k": 30},
                "ndcg_at_10": 0.4,
                "recall_at_10": 0.5,
                "hit_rate_at_10": 0.5,
                "users": 3,
            },
            {"kind": "rrf", "rrf_k": 30},
        ),
        (
            {
                "kind": "percentile_linear",
                "parameters": {"weights": [0.8, 0.2]},
                "ndcg_at_10": 0.4,
                "recall_at_10": 0.5,
                "hit_rate_at_10": 0.5,
                "users": 3,
            },
            {"kind": "percentile_linear", "weights": [0.8, 0.2]},
        ),
    ],
)
def test_select_ranker_unlocks_and_writes_exact_selected_parameters(
    tmp_path, monkeypatch, candidate_row, expected_ranker
) -> None:
    movies, ratings = _tiny_dataset()
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: (movies, ratings))
    itemcf = {
        "kind": "itemcf",
        "parameters": {},
        "ndcg_at_10": 0.2,
        "recall_at_10": 0.3,
        "hit_rate_at_10": 0.3,
        "users": 3,
    }
    monkeypatch.setattr(
        "recagent_eval.cli.build_ranker_ablation",
        lambda *args, **kwargs: [itemcf, candidate_row],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "name: selection\nrequired_retrieval_tools: [itemcf_retrieve, semantic_retrieve]\n"
    )
    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    evidence = tmp_path / "evidence.json"
    selected = tmp_path / "selected.yaml"

    result = CliRunner().invoke(
        app,
        [
            "select-ranker",
            "--config",
            str(config),
            "--cases",
            str(cases),
            "--data-dir",
            str(tmp_path),
            "--evidence-output",
            str(evidence),
            "--config-output",
            str(selected),
            "--max-users",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Frozen test unlocked" in result.output
    assert yaml.safe_load(selected.read_text())["ranker"] == expected_ranker
    assert json.loads(evidence.read_text())["test_unlocked"] is True


def test_evaluate_and_subset_commands_forward_validated_inputs(tmp_path, monkeypatch) -> None:
    movies, ratings = _tiny_dataset()
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: (movies, ratings))
    config = tmp_path / "config.yaml"
    config.write_text("name: evaluate\n")
    source = tmp_path / "cases.json"
    cases = [
        EvaluationCase(
            case_id="single-1",
            user_id=1,
            turns=("recommend",),
            relevant_movie_ids={5},
            tags=("single-turn",),
        )
    ]
    source.write_text(json.dumps([case.model_dump(mode="json") for case in cases]))
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"episodes": len(kwargs["cases"]), "ndcg_at_10": 0.5}

    monkeypatch.setattr("recagent_eval.cli.run_experiment", fake_run)
    output = tmp_path / "run"
    evaluated = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config),
            "--cases",
            str(source),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    subset = tmp_path / "subset.json"
    subset_result = CliRunner().invoke(
        app,
        [
            "subset-cases",
            "--source",
            str(source),
            "--output",
            str(subset),
            "--single-turn-count",
            "1",
            "--multi-turn-count",
            "0",
        ],
    )

    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.output)["episodes"] == 1
    assert seen["output_dir"] == output
    assert subset_result.exit_code == 0, subset_result.output
    assert json.loads(subset.read_text())[0]["case_id"] == "single-1"


def _learned_frozen_config(tmp_path) -> tuple[object, object]:
    consumption_dir = tmp_path / "consumed"
    config = tmp_path / "selected.yaml"
    config.write_text(
        f"""ranker:
  kind: lambdamart
  model_path: {tmp_path / "model.json"}
  evidence_path: {tmp_path / "evidence.json"}
  bundle_manifest_path: {tmp_path / "bundle.json"}
  dataset_fingerprint: dataset
  candidate_policy_fingerprint: policy
  config_fingerprint: config
  case_fingerprint: cases
  gate_fingerprint: gate
  consumption_dir: {consumption_dir}
"""
    )
    marker = consumption_marker_path(
        consumption_dir,
        case_fingerprint="cases",
        dataset_fingerprint="dataset",
        config_fingerprint="config",
    )
    return config, marker


def _stub_learned_preclaim(monkeypatch, *, validation_fingerprint: str) -> None:
    evidence = SimpleNamespace(
        evidence_fingerprint="gate", per_user_rows=[], mean_ndcg_delta=0.1
    )
    monkeypatch.setattr(
        "recagent_eval.cli.load_ranker_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            model_bytes=b"model",
            evidence_bytes=b"evidence",
            latent_bytes=None,
            manifest=SimpleNamespace(schema_version="lambdamart-bundle/v1"),
        ),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LearnedValidationEvidence",
        SimpleNamespace(model_validate_json=lambda _raw: evidence),
    )
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))
    monkeypatch.setattr("recagent_eval.cli.ranking_dataset_fingerprint", lambda *args: "dataset")
    artifact = SimpleNamespace(
        validation_user_count=0,
        validation_rows_fingerprint=validation_fingerprint,
        model_checksum="model",
        model_dump=lambda **kwargs: {},
    )
    monkeypatch.setattr("recagent_eval.cli.parse_ranker_artifact", lambda *args, **kwargs: artifact)
    monkeypatch.setattr("recagent_eval.cli.estimator_from_artifact", lambda _artifact: object())
    monkeypatch.setattr("recagent_eval.cli.TfidfSemanticRetriever.fit", lambda _movies: object())
    monkeypatch.setattr("recagent_eval.cli.build_validation_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr("recagent_eval.cli.candidate_policy_fingerprint", lambda _config: "policy")
    monkeypatch.setattr("recagent_eval.cli.lambdamart_config_fingerprint", lambda _config: "config")
    monkeypatch.setattr("recagent_eval.cli.validate_learned_gate", lambda *args, **kwargs: None)


def test_validation_replay_failure_does_not_consume_marker(tmp_path, monkeypatch) -> None:
    config, marker = _learned_frozen_config(tmp_path)
    _stub_learned_preclaim(monkeypatch, validation_fingerprint="wrong")
    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--cases",
            str(tmp_path / "cases.json"),
        ],
    )
    assert result.exit_code != 0
    assert "replay" in result.output
    assert not marker.exists()


def test_learned_preflight_uses_artifact_v2b_contract_and_evidence_user_order(
    tmp_path, monkeypatch
) -> None:
    from recagent_eval.cli import _evaluate_learned_ranker
    from recagent_eval.runner import ExperimentConfig

    rows = [{"user_id": 9}, {"user_id": 3}]
    row_fingerprint = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evidence = SimpleNamespace(
        evidence_fingerprint="gate", per_user_rows=rows, mean_ndcg_delta=0.1
    )
    artifact = SimpleNamespace(
        validation_user_count=2,
        validation_rows_fingerprint=row_fingerprint,
        model_checksum="e" * 64,
        model_dump=lambda **kwargs: {},
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "recagent_eval.cli.load_ranker_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            model_bytes=b"model",
            evidence_bytes=b"evidence",
            latent_bytes=b"latent",
            manifest=SimpleNamespace(schema_version="lambdamart-bundle/v2"),
        ),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LearnedValidationEvidence",
        SimpleNamespace(model_validate_json=lambda _raw: evidence),
    )
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))
    split = SimpleNamespace(legal_retrieval_train=())
    monkeypatch.setattr("recagent_eval.cli.leakage_safe_ranking_split", lambda _rows: split)
    monkeypatch.setattr(
        "recagent_eval.cli.ranking_dataset_fingerprint", lambda *args: "dataset"
    )
    monkeypatch.setattr("recagent_eval.cli.parse_ranker_artifact", lambda *a, **k: artifact)
    monkeypatch.setattr("recagent_eval.cli.estimator_from_artifact", lambda _artifact: object())

    class CapturingRanker:
        def __init__(self, estimator, **kwargs):
            observed["ranker"] = (estimator, kwargs)

    monkeypatch.setattr("recagent_eval.cli.LearnedRanker", CapturingRanker)
    monkeypatch.setattr("recagent_eval.cli.TfidfSemanticRetriever.fit", lambda _movies: object())

    def capture_replay(*args, **kwargs):
        observed["ordered_user_ids"] = kwargs["ordered_user_ids"]
        return rows

    monkeypatch.setattr("recagent_eval.cli.build_validation_rows", capture_replay)
    monkeypatch.setattr("recagent_eval.cli.candidate_policy_fingerprint", lambda _c: "policy")
    monkeypatch.setattr("recagent_eval.cli.lambdamart_config_fingerprint", lambda _c: "config")
    monkeypatch.setattr("recagent_eval.cli.validate_learned_gate", lambda *a, **k: None)
    monkeypatch.setattr(
        "recagent_eval.cli.consume_frozen_authorization",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("stop after replay")),
    )
    config = ExperimentConfig(
        name="promotion-contract",
        ranker_feature_version="v2b",
        score_calibration="raw",
        learned_model_path=str(tmp_path / "model.json"),
        learned_bundle_manifest_path=str(tmp_path / "bundle.json"),
        learned_dataset_fingerprint="dataset",
        learned_config_fingerprint="config",
        learned_case_fingerprint="cases",
        learned_candidate_policy_fingerprint="policy",
        learned_gate_fingerprint="gate",
        learned_consumption_dir=str(tmp_path / "consumption"),
    )

    with pytest.raises(Exception, match="stop after replay"):
        _evaluate_learned_ranker(
            config=config,
            evidence_path=tmp_path / "evidence.json",
            cases_path=tmp_path / "synthetic-cases.json",
            data_dir=tmp_path / "data",
            output=tmp_path / "output.json",
        )

    assert observed["ordered_user_ids"] == (9, 3)
    _, ranker_kwargs = observed["ranker"]
    assert ranker_kwargs["score_calibration"] == "raw"
    assert ranker_kwargs["feature_version"] == "v2b"


def test_run_frozen_promotion_rejects_wrong_manifest_authorization_before_preflight(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "recagent_eval.cli.load_promotion_documents",
        lambda *_args: (
            object(),
            SimpleNamespace(canonical_manifest_identity="a" * 64),
        ),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.preflight_frozen_promotion",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("wrong authorization reached preflight")
        ),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.load_cases",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("wrong authorization read cases")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "run-frozen-promotion",
            "--promotion",
            str(tmp_path / "promotion.yaml"),
            "--authorized-canonical-manifest-identity",
            "b" * 64,
        ],
    )

    assert result.exit_code != 0
    assert "exact canonical manifest identity" in result.output


def test_preflight_frozen_promotion_wires_complete_v2b_replay_without_cases(
    tmp_path, monkeypatch
) -> None:
    rows = [{"user_id": 9}]
    rows_fingerprint = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = SimpleNamespace(
        training_config_path="configs/v2_dense_latent_bfeat.yaml",
        training_config_fingerprint="config",
        candidate_policy_fingerprint="policy",
        dataset_fingerprint="d" * 64,
        feature_version="v2b",
        feature_fingerprint="feature",
        score_calibration="raw",
        itemcf_top_k=500,
        semantic_top_k=1500,
        latent_top_k=500,
        case_fingerprint="cases",
        model_checksum="e" * 64,
        ordered_user_ids=(9,),
        members={"latent.npz": SimpleNamespace(sha256="latent-sha")},
        semantic=SimpleNamespace(model_name="dense-model", immutable_revision="revision"),
    )
    config = SimpleNamespace(
        ranker_feature_version="v2b",
        score_calibration="raw",
        retrieval_top_k=500,
        semantic_top_k=1500,
        latent_top_k=500,
    )
    split = SimpleNamespace(legal_retrieval_train=())
    artifact = SimpleNamespace(
        feature_fingerprint="feature",
        model_checksum="e" * 64,
        latent_provenance={"training_fingerprint": "latent-training"},
        validation_rows_fingerprint=rows_fingerprint,
        model_dump=lambda **_kwargs: {},
    )
    evidence = SimpleNamespace(per_user_rows=rows)
    observed: dict[str, object] = {}

    monkeypatch.setattr("recagent_eval.cli.load_experiment_config", lambda _path: config)
    monkeypatch.setattr(
        "recagent_eval.cli.lambdamart_config_fingerprint", lambda _config: "config"
    )
    monkeypatch.setattr(
        "recagent_eval.cli.candidate_policy_fingerprint", lambda _config: "policy"
    )
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.leakage_safe_ranking_split", lambda _ratings: split
    )
    monkeypatch.setattr(
        "recagent_eval.cli.ranking_dataset_fingerprint",
        lambda _movies, _split: "d" * 64,
    )
    monkeypatch.setattr(
        "recagent_eval.cli.load_ranker_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            model_bytes=b"model", evidence_bytes=b"evidence"
        ),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.parse_ranker_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LearnedValidationEvidence",
        SimpleNamespace(model_validate_json=lambda _raw: evidence),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.DenseSemanticRetriever.load_read_only",
        lambda *args, **kwargs: "semantic",
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LatentFactorRetriever.load", lambda *args, **kwargs: "latent"
    )
    monkeypatch.setattr(
        "recagent_eval.cli.estimator_from_artifact", lambda _artifact: "estimator"
    )

    class CapturingRanker:
        def __init__(self, estimator, **kwargs):
            observed["ranker"] = (estimator, kwargs)

    monkeypatch.setattr("recagent_eval.cli.LearnedRanker", CapturingRanker)
    monkeypatch.setattr(
        "recagent_eval.cli.build_validation_rows", lambda *args, **kwargs: rows
    )
    monkeypatch.setattr(
        "recagent_eval.cli.validate_learned_gate", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "recagent_eval.cli.verify_git_identity", lambda *_args, **_kwargs: None
    )

    def run_callbacks(_root, _promotion, **kwargs):
        members = {
            name: tmp_path / name
            for name in (
                "model.json",
                "validation.json",
                "bundle.json",
                "latent.npz",
                "latent.npz.json",
                "semantic.npz",
                "semantic.npz.json",
            )
        }
        assert kwargs["dataset_fingerprint_check"](manifest, members) == "d" * 64
        replay = kwargs["validation_replay"](manifest, members)
        assert replay.validation_rows_fingerprint == rows_fingerprint
        kwargs["git_identity_check"](manifest)
        return SimpleNamespace(model_dump_json=lambda **_kwargs: '{"label_free":true}')

    monkeypatch.setattr("recagent_eval.cli.preflight_promotion", run_callbacks)
    monkeypatch.setattr(
        "recagent_eval.cli.load_cases",
        lambda *_args: (_ for _ in ()).throw(AssertionError("preflight read cases")),
    )

    result = CliRunner().invoke(
        app,
        [
            "preflight-frozen-promotion",
            "--promotion",
            str(tmp_path / "reports/promotion/synthetic.yaml"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"label_free":true' in result.output
    estimator, ranker_kwargs = observed["ranker"]
    assert estimator == "estimator"
    assert ranker_kwargs["feature_version"] == "v2b"
    assert ranker_kwargs["score_calibration"] == "raw"


def test_frozen_execution_runtime_loads_only_manifest_bound_package_members(
    tmp_path, monkeypatch
) -> None:
    from recagent_eval.cli import _load_frozen_execution_runtime

    members = {
        name: SimpleNamespace(path=f"artifacts/promotion/current-v2b/{name}")
        for name in (
            "model.json",
            "validation.json",
            "bundle.json",
            "latent.npz",
            "latent.npz.json",
            "semantic.npz",
            "semantic.npz.json",
        )
    }
    members["latent.npz"].sha256 = "latent-sha"
    manifest = SimpleNamespace(
        training_config_path="configs/v2_dense_latent_bfeat.yaml",
        members=members,
        training_config_fingerprint="config",
        dataset_fingerprint="dataset",
        candidate_policy_fingerprint="policy",
        feature_fingerprint="feature",
        case_fingerprint="cases",
        model_checksum="model-checksum",
        semantic=SimpleNamespace(model_name="dense-model", immutable_revision="revision"),
        score_calibration="raw",
        feature_version="v2b",
    )
    split = SimpleNamespace(legal_retrieval_train=("legal-row",))
    artifact = SimpleNamespace(
        feature_fingerprint="feature",
        model_checksum="model-checksum",
        latent_provenance={"training_fingerprint": "latent-training"},
    )
    evidence = SimpleNamespace(evidence_fingerprint="evidence")
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "recagent_eval.cli.load_experiment_config", lambda _path: "config-object"
    )
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.leakage_safe_ranking_split", lambda _ratings: split
    )

    def load_bundle(*args, **kwargs):
        observed["bundle_args"] = args
        observed["bundle_kwargs"] = kwargs
        return SimpleNamespace(model_bytes=b"model", evidence_bytes=b"evidence")

    monkeypatch.setattr("recagent_eval.cli.load_ranker_bundle", load_bundle)
    monkeypatch.setattr(
        "recagent_eval.cli.parse_ranker_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LearnedValidationEvidence",
        SimpleNamespace(model_validate_json=lambda _raw: evidence),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.DenseSemanticRetriever.load_read_only",
        lambda *args, **kwargs: "semantic",
    )
    monkeypatch.setattr(
        "recagent_eval.cli.LatentFactorRetriever.load", lambda *args, **kwargs: "latent"
    )
    monkeypatch.setattr(
        "recagent_eval.cli.estimator_from_artifact", lambda _artifact: "estimator"
    )

    class CapturingRanker:
        def __init__(self, estimator, **kwargs):
            observed["ranker"] = (estimator, kwargs)

    monkeypatch.setattr("recagent_eval.cli.LearnedRanker", CapturingRanker)

    runtime = _load_frozen_execution_runtime(tmp_path, manifest, tmp_path / "data")

    assert runtime["config"] == "config-object"
    assert runtime["semantic"] == "semantic"
    assert runtime["latent"] == "latent"
    assert runtime["evidence"] is evidence
    assert observed["bundle_args"][0] == (
        tmp_path / "artifacts/promotion/current-v2b/model.json"
    )
    assert observed["ranker"][1]["feature_version"] == "v2b"
    artifact.model_checksum = "drifted-model"
    with pytest.raises(ValueError, match="model identity drift"):
        _load_frozen_execution_runtime(tmp_path, manifest, tmp_path / "data")


def test_prepare_frozen_promotion_wires_locked_sources_without_regeneration(
    tmp_path, monkeypatch
) -> None:
    inventory = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "recagent_eval.cli.load_source_inventory", lambda path: inventory
    )

    def publish(root, active_inventory, sources):
        observed.update(root=root, inventory=active_inventory, sources=sources)
        return tmp_path / "artifacts/promotion/current-v2b"

    monkeypatch.setattr("recagent_eval.cli.publish_promotion_package", publish)
    confirmation = tmp_path / "confirmation"
    semantic = tmp_path / "semantic.npz"
    result = CliRunner().invoke(
        app,
        [
            "prepare-frozen-promotion",
            "--inventory",
            str(tmp_path / "inventory.json"),
            "--confirmation-source",
            str(confirmation),
            "--semantic-cache",
            str(semantic),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["inventory"] is inventory
    assert observed["sources"]["model.json"] == confirmation / "model.json"
    assert observed["sources"]["semantic.npz"] == semantic
    assert observed["sources"]["semantic.npz.json"] == Path(f"{semantic}.json")


def test_prepare_and_audit_promotion_cli_wrap_failures_and_emit_read_only_result(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "recagent_eval.cli.load_source_inventory",
        lambda _path: (_ for _ in ()).throw(ValueError("locked source missing")),
    )
    failed = CliRunner().invoke(
        app,
        ["prepare-frozen-promotion", "--inventory", str(tmp_path / "missing.json")],
    )
    assert failed.exit_code != 0
    assert "locked source missing" in failed.output

    monkeypatch.setattr(
        "recagent_eval.cli.load_promotion_documents", lambda *_args: ("manifest", "yaml")
    )
    monkeypatch.setattr(
        "recagent_eval.cli.audit_one_shot",
        lambda *_args: {"state": "started", "output_exists": True},
    )
    audited = CliRunner().invoke(
        app,
        ["audit-frozen-promotion", "--promotion", str(tmp_path / "promotion.yaml")],
    )
    assert audited.exit_code == 0, audited.output
    assert json.loads(audited.output) == {"output_exists": True, "state": "started"}


@pytest.mark.parametrize(
    ("config_fingerprint", "policy_fingerprint", "feature_version", "message"),
    [
        ("wrong", "policy", "v2b", "training config fingerprint"),
        ("config", "wrong", "v2b", "candidate-policy fingerprint"),
        ("config", "policy", "v1", "training/candidate/feature contract"),
    ],
)
def test_preflight_cli_fails_closed_on_training_identity_drift(
    tmp_path,
    monkeypatch,
    config_fingerprint,
    policy_fingerprint,
    feature_version,
    message,
) -> None:
    from recagent_eval.cli import preflight_frozen_promotion

    manifest = SimpleNamespace(
        training_config_path="configs/v2_dense_latent_bfeat.yaml",
        training_config_fingerprint="config",
        candidate_policy_fingerprint="policy",
        feature_version="v2b",
        score_calibration="raw",
        itemcf_top_k=500,
        semantic_top_k=1500,
        latent_top_k=500,
    )
    config = SimpleNamespace(
        ranker_feature_version=feature_version,
        score_calibration="raw",
        retrieval_top_k=500,
        semantic_top_k=1500,
        latent_top_k=500,
    )
    monkeypatch.setattr("recagent_eval.cli.load_experiment_config", lambda _path: config)
    monkeypatch.setattr(
        "recagent_eval.cli.lambdamart_config_fingerprint",
        lambda _config: config_fingerprint,
    )
    monkeypatch.setattr(
        "recagent_eval.cli.candidate_policy_fingerprint",
        lambda _config: policy_fingerprint,
    )

    def invoke_dataset(_root, _promotion, **kwargs):
        kwargs["dataset_fingerprint_check"](manifest, {})
        raise AssertionError("identity drift did not fail closed")

    monkeypatch.setattr("recagent_eval.cli.preflight_promotion", invoke_dataset)
    with pytest.raises(Exception, match=message):
        preflight_frozen_promotion(tmp_path / "promotion.yaml", tmp_path / "data")


def test_run_frozen_promotion_wires_authorized_synthetic_services_without_real_io(
    tmp_path, monkeypatch
) -> None:
    manifest_sha = "a" * 64
    manifest = SimpleNamespace(
        frozen_cases_path="cases/fixed_cases.json",
        case_fingerprint="case-fingerprint",
        dataset_fingerprint="dataset-fingerprint",
        model_checksum="model-checksum",
    )
    promotion = SimpleNamespace(
        canonical_manifest_identity=manifest_sha,
        manifest_file_sha256="f" * 64,
    )
    receipt = object()
    config = SimpleNamespace(
        retrieval_top_k=500,
        semantic_top_k=1500,
        semantic_profile_history_cap=50,
        latent_top_k=500,
        ranker_feature_version="v2b",
    )
    cases = ["synthetic-case"]
    runtime = {
        "config": config,
        "movies": {1: "movie"},
        "split": SimpleNamespace(legal_retrieval_train=("legal",)),
        "ranker": "ranker",
        "semantic": "semantic",
        "latent": "latent",
        "evidence": SimpleNamespace(evidence_fingerprint="evidence-fingerprint"),
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "recagent_eval.cli.load_promotion_documents",
        lambda *_args: (manifest, promotion),
    )
    monkeypatch.setattr(
        "recagent_eval.cli.preflight_frozen_promotion", lambda *_args: receipt
    )
    monkeypatch.setattr(
        "recagent_eval.cli._load_frozen_execution_runtime", lambda *_args: runtime
    )
    monkeypatch.setattr("recagent_eval.cli.load_cases", lambda _path: cases)
    monkeypatch.setattr(
        "recagent_eval.cli.case_fingerprint", lambda active_cases: "case-fingerprint"
    )

    def evaluate(*args, **kwargs):
        observed["evaluate_args"] = args
        observed["evaluate_kwargs"] = kwargs
        return {"cases": 1}

    monkeypatch.setattr("recagent_eval.cli.evaluate_frozen_cases", evaluate)

    def execute(root, active_manifest, active_promotion, active_receipt, **kwargs):
        observed["execute"] = (
            root,
            active_manifest,
            active_promotion,
            active_receipt,
            kwargs["authorized_canonical_manifest_identity"],
        )
        loaded_cases = kwargs["case_loader"]()
        observed["metrics"] = kwargs["evaluator"](loaded_cases)
        return SimpleNamespace(model_dump_json=lambda **_kwargs: '{"state":"completed"}')

    monkeypatch.setattr("recagent_eval.cli.execute_one_shot", execute)
    result = CliRunner().invoke(
        app,
        [
            "run-frozen-promotion",
            "--promotion",
            str(tmp_path / "reports/promotion/synthetic.yaml"),
            "--authorized-canonical-manifest-identity",
            manifest_sha,
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"state":"completed"' in result.output
    assert observed["execute"][4] == manifest_sha
    assert observed["evaluate_args"][2] == cases
    assert observed["evaluate_kwargs"]["semantic_top_k"] == 1500
    assert observed["evaluate_kwargs"]["latent_top_k"] == 500
    assert observed["metrics"]["canonical_manifest_identity"] == manifest_sha
    assert observed["metrics"]["manifest_file_sha256"] == "f" * 64
    assert observed["metrics"]["selection_evidence_fingerprint"] == (
        "evidence-fingerprint"
    )


def test_missing_bundle_members_do_not_consume_marker(tmp_path) -> None:
    config, marker = _learned_frozen_config(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--cases",
            str(tmp_path / "cases.json"),
        ],
    )
    assert result.exit_code != 0
    assert "bundle manifest" in result.output
    assert not marker.exists()


def test_case_loader_failure_after_claim_keeps_marker_consumed(tmp_path, monkeypatch) -> None:
    config, marker = _learned_frozen_config(tmp_path)
    fingerprint = hashlib.sha256(b"[]").hexdigest()
    _stub_learned_preclaim(monkeypatch, validation_fingerprint=fingerprint)
    monkeypatch.setattr(
        "recagent_eval.cli.load_cases", lambda _path: (_ for _ in ()).throw(ValueError("bad cases"))
    )
    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--cases",
            str(tmp_path / "cases.json"),
        ],
    )
    assert result.exit_code != 0
    assert marker.exists()


def test_build_embeddings_uses_injected_sentence_encoder(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "ml-1m"
    data_dir.mkdir()
    (data_dir / "movies.dat").write_text(
        "1::Space One (2000)::Sci-Fi\n2::Comedy (2001)::Comedy\n",
        encoding="latin-1",
    )
    class FakeSentenceEncoder:
        model_revision = "a" * 40
        initializations = 0
        encodes = 0

        def __init__(self, model_name: str, *, revision: str | None, device: str) -> None:
            type(self).initializations += 1
            assert model_name == "fake/model"
            assert revision == "main"
            assert device == "cpu"

        def encode(self, texts: list[str]) -> np.ndarray:
            type(self).encodes += 1
            return np.eye(len(texts), dtype=np.float32)

    monkeypatch.setattr("recagent_eval.retrieval.SentenceTransformerEncoder", FakeSentenceEncoder)
    output = tmp_path / "embeddings.npz"

    result = CliRunner().invoke(
        app,
        [
            "build-embeddings",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--model-name",
            "fake/model",
            "--model-revision",
            "main",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert FakeSentenceEncoder.initializations == 1
    assert FakeSentenceEncoder.encodes == 1
    with np.load(output, allow_pickle=False) as payload:
        assert payload["embeddings"].shape == (2, 2)

    reused = CliRunner().invoke(
        app,
        [
            "build-embeddings",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--model-name",
            "fake/model",
            "--model-revision",
            "main",
        ],
    )
    assert reused.exit_code == 0, reused.output
    assert "Reused" in reused.output
    assert FakeSentenceEncoder.initializations == 1
    assert FakeSentenceEncoder.encodes == 1

    mismatch = CliRunner().invoke(
        app,
        [
            "build-embeddings",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--model-name",
            "different/model",
            "--model-revision",
            "main",
        ],
    )
    assert mismatch.exit_code != 0
    assert "model_name mismatch" in mismatch.output
    assert FakeSentenceEncoder.initializations == 1

    rebuilt = CliRunner().invoke(
        app,
        [
            "build-embeddings",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--model-name",
            "fake/model",
            "--model-revision",
            "main",
            "--force",
        ],
    )
    assert rebuilt.exit_code == 0, rebuilt.output
    assert FakeSentenceEncoder.initializations == 2
    assert FakeSentenceEncoder.encodes == 2


def test_select_retrieval_writes_evidence_and_frozen_config(
    tmp_path,
    monkeypatch,
) -> None:
    rows = [
        {
            "retrieval_top_k": 200,
            "semantic_profile_history_cap": 20,
            "itemcf_candidate_recall": 0.4,
            "semantic_candidate_recall": 0.2,
            "union_candidate_recall": 0.5,
            "ndcg_at_10": 0.1,
            "latency_ms_per_user": 2.0,
            "users": 10,
        }
    ]
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.build_retrieval_ablation",
        lambda movies, split: rows,
    )
    evidence = tmp_path / "ablation.json"
    config = tmp_path / "frozen.yaml"

    result = CliRunner().invoke(
        app,
        [
            "select-retrieval",
            "--evidence-output",
            str(evidence),
            "--config-output",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(evidence.read_text())["rows"] == rows
    frozen = yaml.safe_load(config.read_text())
    assert frozen["retrieval_top_k"] == 200
    assert frozen["semantic_profile_history_cap"] == 20
    assert frozen["required_retrieval_tools"] == [
        "itemcf_retrieve",
        "semantic_retrieve",
    ]


def test_tune_updates_a_frozen_config_with_validation_weights(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.tune_on_validation",
        lambda *args, **kwargs: (0.7000000000000001, 0.30000000000000004, 0.0),
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        "\n".join(
            [
                "name: frozen",
                "retrieval_top_k: 200",
                "semantic_profile_history_cap: 20",
                "weights: [0.7, 0.3, 0.0]",
            ]
        )
        + "\n"
    )
    weights_output = tmp_path / "weights.json"
    config_output = tmp_path / "updated.yaml"

    result = CliRunner().invoke(
        app,
        [
            "tune",
            "--config",
            str(source),
            "--config-output",
            str(config_output),
            "--output",
            str(weights_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(weights_output.read_text())["weights"] == [0.7, 0.3, 0.0]
    assert yaml.safe_load(config_output.read_text())["weights"] == [0.7, 0.3, 0.0]


def _ranker_source_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "name: full",
                "retrieval_top_k: 500",
                "semantic_profile_history_cap: 50",
                "enable_semantic_retrieval: true",
                "required_retrieval_tools: [itemcf_retrieve, semantic_retrieve]",
                "weights: [0.7, 0.3, 0.0]",
            ]
        )
        + "\n"
    )


def test_select_ranker_writes_unlocked_evidence_and_config(
    tmp_path,
    monkeypatch,
) -> None:
    rows = [
        {
            "kind": "itemcf",
            "parameters": {},
            "ndcg_at_10": 0.1,
            "recall_at_10": 0.1,
            "hit_rate_at_10": 0.1,
            "users": 10,
        },
        {
            "kind": "rrf",
            "parameters": {"rrf_k": 30},
            "ndcg_at_10": 0.2,
            "recall_at_10": 0.2,
            "hit_rate_at_10": 0.2,
            "users": 10,
        },
    ]
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.build_ranker_ablation",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        "recagent_eval.cli.ranker_dataset_fingerprint",
        lambda *args, **kwargs: "abc",
    )
    source = tmp_path / "source.yaml"
    _ranker_source_config(source)
    evidence = tmp_path / "ablation.json"
    config = tmp_path / "selected.yaml"

    result = CliRunner().invoke(
        app,
        [
            "select-ranker",
            "--config",
            str(source),
            "--evidence-output",
            str(evidence),
            "--config-output",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(evidence.read_text())["test_unlocked"] is True
    selected = yaml.safe_load(config.read_text())
    assert selected["ranker"] == {"kind": "rrf", "rrf_k": 30}
    assert selected["retrieval_top_k"] == 500
    assert "Frozen test unlocked" in result.output


def test_select_ranker_keeps_test_locked_without_improvement(
    tmp_path,
    monkeypatch,
) -> None:
    rows = [
        {
            "kind": "itemcf",
            "parameters": {},
            "ndcg_at_10": 0.2,
            "recall_at_10": 0.2,
            "hit_rate_at_10": 0.2,
            "users": 10,
        },
        {
            "kind": "rrf",
            "parameters": {"rrf_k": 30},
            "ndcg_at_10": 0.1,
            "recall_at_10": 0.1,
            "hit_rate_at_10": 0.1,
            "users": 10,
        },
    ]
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.build_ranker_ablation",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        "recagent_eval.cli.ranker_dataset_fingerprint",
        lambda *args, **kwargs: "abc",
    )
    source = tmp_path / "source.yaml"
    _ranker_source_config(source)
    evidence = tmp_path / "ablation.json"
    config = tmp_path / "selected.yaml"

    result = CliRunner().invoke(
        app,
        [
            "select-ranker",
            "--config",
            str(source),
            "--evidence-output",
            str(evidence),
            "--config-output",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(evidence.read_text())["test_unlocked"] is False
    assert not config.exists()
    assert "Frozen test remains locked" in result.output


def _selection_evidence(path, *, unlocked: bool) -> None:
    selected_ndcg = 0.21 if unlocked else 0.19
    rrf = {
        "kind": "rrf",
        "parameters": {"rrf_k": 30},
        "ndcg_at_10": selected_ndcg,
        "recall_at_10": 0.2,
        "hit_rate_at_10": 0.2,
        "users": 10,
    }
    itemcf = {
        "kind": "itemcf",
        "parameters": {},
        "ndcg_at_10": 0.2,
        "recall_at_10": 0.2,
        "hit_rate_at_10": 0.2,
        "users": 10,
    }
    selected = rrf if unlocked else itemcf
    payload = {
        "rows": [itemcf, rrf],
        "selected": selected,
        "itemcf_ndcg_at_10": 0.2,
        "selected_ndcg_at_10": selected["ndcg_at_10"],
        "margin": selected["ndcg_at_10"] - 0.2,
        "test_unlocked": unlocked,
        "dataset_fingerprint": "abc",
        "case_fingerprint": hashlib.sha256(
            json.dumps([], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "retrieval_top_k": 500,
        "semantic_profile_history_cap": 50,
        "max_users": 10,
    }
    path.write_text(json.dumps(payload))


def test_evaluate_ranker_refuses_locked_evidence_before_writing(
    tmp_path,
    monkeypatch,
) -> None:
    config = tmp_path / "selected.yaml"
    _ranker_source_config(config)
    with config.open("a") as stream:
        stream.write("ranker:\n  kind: rrf\n  rrf_k: 30\n")
    evidence = tmp_path / "locked.json"
    _selection_evidence(evidence, unlocked=False)
    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    output = tmp_path / "result" / "metrics.json"
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.ranker_dataset_fingerprint",
        lambda *args, **kwargs: "abc",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(evidence),
            "--cases",
            str(cases),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "frozen test is locked" in result.output.lower()
    assert not output.exists()


def test_evaluate_ranker_writes_unlocked_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    config = tmp_path / "selected.yaml"
    _ranker_source_config(config)
    with config.open("a") as stream:
        stream.write("ranker:\n  kind: rrf\n  rrf_k: 30\n")
    evidence = tmp_path / "unlocked.json"
    _selection_evidence(evidence, unlocked=True)
    cases = tmp_path / "cases.json"
    cases.write_text("[]")
    output = tmp_path / "result" / "metrics.json"
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.ranker_dataset_fingerprint",
        lambda *args, **kwargs: "abc",
    )
    monkeypatch.setattr(
        "recagent_eval.cli.evaluate_frozen_cases",
        lambda *args, **kwargs: {"cases": 0, "ranker_kind": "rrf"},
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(evidence),
            "--cases",
            str(cases),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads(output.read_text())
    assert metrics["ranker_kind"] == "rrf"
    assert metrics["selection_evidence_fingerprint"]


def test_evaluate_ranker_rejects_unregistered_case_fingerprint(
    tmp_path,
    monkeypatch,
) -> None:
    config = tmp_path / "selected.yaml"
    _ranker_source_config(config)
    with config.open("a") as stream:
        stream.write("ranker:\n  kind: rrf\n  rrf_k: 30\n")
    evidence = tmp_path / "unlocked.json"
    _selection_evidence(evidence, unlocked=True)
    cases = tmp_path / "changed-cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "changed",
                    "user_id": 1,
                    "turns": ["recommend"],
                    "relevant_movie_ids": [1],
                }
            ]
        )
    )
    output = tmp_path / "result" / "metrics.json"
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda path: ({}, []))
    monkeypatch.setattr(
        "recagent_eval.cli.ranker_dataset_fingerprint",
        lambda *args, **kwargs: "abc",
    )
    monkeypatch.setattr(
        "recagent_eval.cli.evaluate_frozen_cases",
        lambda *args, **kwargs: {"cases": 1, "ranker_kind": "rrf"},
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-ranker",
            "--config",
            str(config),
            "--evidence",
            str(evidence),
            "--cases",
            str(cases),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "case_fingerprint" in result.output
    assert not output.exists()


def test_diagnose_latent_refuses_overwrite_and_requires_latent(
    tmp_path, monkeypatch
) -> None:
    from recagent_eval.data import Movie, Rating

    movies = {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 8)
        for movie_id in range(1, 9)
    ]
    monkeypatch.setattr(
        "recagent_eval.cli._load_dataset", lambda _path: (movies, ratings)
    )
    config_path = tmp_path / "latent.yaml"
    config_path.write_text(
        "semantic:\n"
        "  kind: tfidf\n"
        "latent:\n"
        "  enabled: true\n"
        "  artifact_path: artifacts/experiments/x/latent.npz\n"
        "ranker:\n"
        "  feature_version: v2\n"
    )
    output = tmp_path / "diagnostics.json"
    result = CliRunner().invoke(
        app,
        [
            "diagnose-latent",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    result = CliRunner().invoke(
        app,
        [
            "diagnose-latent",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )
    assert "refusing to overwrite" in result.output
    disabled = tmp_path / "disabled.yaml"
    disabled.write_text("semantic:\n  kind: tfidf\n")
    result = CliRunner().invoke(
        app,
        [
            "diagnose-latent",
            "--config",
            str(disabled),
            "--output",
            str(tmp_path / "d2.json"),
        ],
    )
    assert "latent" in result.output.lower()


def test_evaluate_baselines_refuses_overwrite_before_method_check(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cohorts": {"confirmation_a": [1, 2]}}))
    output = tmp_path / "out.json"
    output.write_text("{}")
    result = CliRunner().invoke(
        app,
        [
            "evaluate-baselines",
            "--ledger",
            str(ledger),
            "--cohort",
            "confirmation_a",
            "--method",
            "popularity",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output


def test_evaluate_baselines_rejects_bad_cohort_and_unknown_method(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cohorts": {"confirmation_a": [1, 2]}}))
    result = CliRunner().invoke(
        app,
        [
            "evaluate-baselines",
            "--ledger",
            str(ledger),
            "--cohort",
            "bogus",
            "--method",
            "popularity",
            "--output",
            str(tmp_path / "o.json"),
        ],
    )
    assert result.exit_code != 0
    assert "cohort must be" in result.output
    result = CliRunner().invoke(
        app,
        [
            "evaluate-baselines",
            "--ledger",
            str(ledger),
            "--cohort",
            "confirmation_a",
            "--method",
            "nope",
            "--output",
            str(tmp_path / "o.json"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown baseline method" in result.output


def test_evaluate_baselines_rejects_nondefault_seed_without_locked_params(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cohorts": {"confirmation_b": [1]}}))
    result = CliRunner().invoke(
        app,
        [
            "evaluate-baselines",
            "--ledger",
            str(ledger),
            "--cohort",
            "confirmation_b",
            "--method",
            "bpr_mf",
            "--seed",
            "7",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
    assert "locked-params" in result.output


def test_evaluate_baselines_writes_strict_v2_with_locked_recovery(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"fingerprint": "l" * 64, "cohorts": {"confirmation_b": [7]}})
    )
    recovery_binding = {
        "status": "recovered_after_run",
        "command": "recover",
        "source_artifact_sha256": "s" * 64,
        "input_fingerprint": "i" * 64,
        "output_fingerprint": "o" * 64,
        "commit_sha": "c" * 40,
    }
    manifest = {
        "schema_version": "baseline-parameter-recovery/v1",
        "cohorts": {
            "confirmation_b": {
                "bpr_mf": {
                    "selection_fingerprint": "f" * 64,
                    "selected_params": {
                        "source": "recovered",
                        "value": {"rank": 16},
                        "recovery": recovery_binding,
                    },
                    "parameter_grid": {
                        "source": "recovered",
                        "value": [{"rank": 16}],
                        "recovery": recovery_binding,
                    },
                }
            }
        },
    }
    manifest["fingerprint"] = canonical_digest(manifest)
    locked = tmp_path / "recovery.json"
    locked.write_text(json.dumps(manifest))
    movies = {1: Movie(1, "Movie", ("Drama",), 2000)}
    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: (movies, []))
    monkeypatch.setattr(
        "recagent_eval.cli.leakage_safe_ranking_split", lambda _ratings: object()
    )
    seen = {}

    def fake_scorer(_movies, _split, users, **kwargs):
        seen.update(kwargs)
        assert users == [7]
        return {
            "rows": [MetricRow(7, 1.0, 1.0, 1.0, 1.0, True, 1.0, (1,))],
            "config_fingerprint": "f" * 64,
            "dataset_fingerprint": "d" * 64,
            "model_fingerprint": "m" * 64,
            "selected_params": {"rank": 16},
            "parameter_grid": [{"rank": 16}],
            "seed": 7,
            "training_seconds": 1.0,
            "resource_usage": {
                "metric_name": "process_peak_rss_mib",
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
                "raw_value": 1024,
                "raw_unit": "bytes",
                "normalized_mib": 1.0,
                "platform": "Darwin",
                "measurement_scope": "single_process_lifetime_peak",
                "process_id": 1,
            },
            "model_size_bytes": 10,
            "environment": {"python": "test", "numpy": "test"},
        }

    monkeypatch.setitem(
        __import__("recagent_eval.cli", fromlist=["BASELINE_SCORERS"]).BASELINE_SCORERS,
        "bpr_mf",
        fake_scorer,
    )
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-baselines",
            "--ledger",
            str(ledger),
            "--cohort",
            "confirmation_b",
            "--method",
            "bpr_mf",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--locked-params",
            str(locked),
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = json.loads(output.read_text())
    assert artifact["schema_version"] == "baseline-evaluation/v2"
    assert artifact["seed"] == {"source": "observed", "value": 7}
    assert artifact["selected_params"]["source"] == "recovered"
    assert seen["selected_params"] == {"rank": 16}
    assert seen["seed"] == 7


def test_summarize_baselines_rejects_bad_cohort_and_existing_output(tmp_path) -> None:
    output = tmp_path / "summary.json"
    result = CliRunner().invoke(
        app,
        ["summarize-baselines", "--cohort", "bogus", "--output", str(output)],
    )
    assert result.exit_code != 0
    assert "cohort must be" in result.output

    output.write_text("{}")
    result = CliRunner().invoke(
        app,
        [
            "summarize-baselines",
            "--cohort",
            "confirmation_a",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output


def test_summarize_baselines_rejects_missing_and_corrupt_artifacts(tmp_path) -> None:
    output = tmp_path / "summary.json"
    result = CliRunner().invoke(
        app,
        [
            "summarize-baselines",
            "--cohort",
            "confirmation_a",
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "missing baseline artifact" in result.output

    (tmp_path / "popularity-confirmation-a.json").write_text("not-json")
    result = CliRunner().invoke(
        app,
        [
            "summarize-baselines",
            "--cohort",
            "confirmation_a",
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "invalid baseline artifact" in result.output


def test_build_evidence_bundle_refuses_existing_output_before_inputs(tmp_path) -> None:
    output = tmp_path / "bundle.json"
    output.write_text("{}")
    result = CliRunner().invoke(
        app,
        [
            "build-evidence-bundle",
            "--cohort",
            "confirmation_b",
            "--ledger",
            str(tmp_path / "missing-ledger.json"),
            "--artifact-dir",
            str(tmp_path / "missing-artifacts"),
            "--summary",
            str(tmp_path / "missing-summary.json"),
            "--recovery",
            str(tmp_path / "missing-recovery.json"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output


def test_replay_evidence_rejects_invalid_bundle(tmp_path) -> None:
    bundle = tmp_path / "bundle.json"
    ledger = tmp_path / "ledger.json"
    summary = tmp_path / "summary.json"
    bundle.write_text("{}")
    ledger.write_text("{}")
    summary.write_text("{}")
    result = CliRunner().invoke(
        app,
        [
            "replay-evidence",
            "--bundle",
            str(bundle),
            "--ledger",
            str(ledger),
            "--summary",
            str(summary),
        ],
    )
    assert result.exit_code != 0
    assert "unknown bundle schema" in result.output


def test_build_evidence_bundle_writes_new_file_with_bound_inputs(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"fingerprint":"ledger"}')
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"aggregates": {"bpr_mf": {}}}))
    recovery = tmp_path / "recovery.json"
    recovery.write_text(
        json.dumps(
            {
                "schema_version": "baseline-parameter-recovery/v1",
                "cohorts": {"confirmation_b": {"bpr_mf": {"selected_params": {}}}},
            }
        )
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "bpr-mf-confirmation-b.json").write_text('{"source":true}')
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {"fingerprint": "f" * 64}

    monkeypatch.setattr("recagent_eval.cli.build_compact_bundle", fake_build)
    monkeypatch.setattr(
        "recagent_eval.cli.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="a" * 40 + "\n"),
    )
    output = tmp_path / "bundle.json"
    result = CliRunner().invoke(
        app,
        [
            "build-evidence-bundle",
            "--cohort",
            "confirmation_b",
            "--ledger",
            str(ledger),
            "--artifact-dir",
            str(artifact_dir),
            "--summary",
            str(summary),
            "--recovery",
            str(recovery),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["fingerprint"] == "f" * 64
    assert seen["cohort"] == "confirmation_b"
    assert seen["commit_sha"] == "a" * 40
    assert seen["recovery"] == {"bpr_mf": {"selected_params": {}}}


def test_recover_baseline_params_refuses_existing_output_before_data(tmp_path) -> None:
    output = tmp_path / "recovery.json"
    output.write_text("{}")
    result = CliRunner().invoke(
        app,
        [
            "recover-baseline-params",
            "--ledger",
            str(tmp_path / "missing-ledger.json"),
            "--artifact-dir",
            str(tmp_path / "missing-artifacts"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output


def test_recover_baseline_params_writes_bound_manifest(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cohorts": {"development": [1, 2]}}))
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    def selection(fingerprint: str, *, lightgcn: bool = False) -> dict[str, object]:
        selected = {"rank": 16}
        if lightgcn:
            selected["epochs"] = 20
        return {
            "selected_params": selected,
            "grid": [selected],
            "fingerprint": fingerprint,
            "seed": 42,
            **({"epochs": 20} if lightgcn else {}),
        }

    fingerprints = {
        "popularity": hashlib.sha256(b"popularity/v1").hexdigest(),
        "itemcf_direct": hashlib.sha256(b"itemcf_direct/v1").hexdigest(),
        "als_direct": "a" * 64,
        "bpr_mf": "b" * 64,
        "lightgcn": "l" * 64,
        "current_v2b": "c" * 64,
    }
    for cohort in ("confirmation-a", "confirmation-b"):
        for method, fingerprint in fingerprints.items():
            path = artifact_dir / f"{method.replace('_', '-')}-{cohort}.json"
            path.write_text(
                json.dumps({"config_fingerprint": fingerprint, "fingerprint": method})
            )

    @dataclass
    class FakeConfig:
        seed: int = 42

    monkeypatch.setattr("recagent_eval.cli._load_dataset", lambda _path: ({}, []))
    monkeypatch.setattr("recagent_eval.cli.leakage_safe_ranking_split", lambda _ratings: object())
    monkeypatch.setattr(
        "recagent_eval.cli._als_registration.select_als_params",
        lambda *_args: selection("a" * 64),
    )
    monkeypatch.setattr(
        "recagent_eval.cli._bpr_registration.select_bpr_params",
        lambda *_args: selection("b" * 64),
    )
    monkeypatch.setattr(
        "recagent_eval.cli._lightgcn_registration.select_lightgcn_params",
        lambda *_args: selection("l" * 64, lightgcn=True),
    )
    monkeypatch.setattr("recagent_eval.cli.load_experiment_config", lambda _path: FakeConfig())
    monkeypatch.setattr(
        "recagent_eval.cli.lambdamart_config_fingerprint", lambda _config: "c" * 64
    )
    output = tmp_path / "recovery.json"
    result = CliRunner().invoke(
        app,
        [
            "recover-baseline-params",
            "--ledger",
            str(ledger),
            "--artifact-dir",
            str(artifact_dir),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "recovered_after_run"
    assert set(manifest["cohorts"]) == {"confirmation_a", "confirmation_b"}
    assert manifest["cohorts"]["confirmation_b"]["bpr_mf"]["selected_params"][
        "source"
    ] == "recovered"

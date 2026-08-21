import hashlib
import json
from types import SimpleNamespace

import numpy as np
import yaml
from typer.testing import CliRunner

from recagent_eval.cli import app
from recagent_eval.data import Movie, Rating
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
        "recagent_eval.cli.load_ranker_bundle", lambda *args, **kwargs: (b"model", b"evidence")
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

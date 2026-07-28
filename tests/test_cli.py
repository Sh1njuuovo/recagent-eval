import json

import yaml
from typer.testing import CliRunner

from recagent_eval.cli import app


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

import json

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

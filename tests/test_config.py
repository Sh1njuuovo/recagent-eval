from pathlib import Path

import pytest

from recagent_eval.config import load_experiment_config


def test_legacy_weights_keep_minmax_linear_behavior(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text("name: legacy\nweights: [0.7, 0.3, 0.0]\n")

    config = load_experiment_config(path)

    assert config.ranker_kind == "minmax_linear"
    assert config.weights == (0.7, 0.3, 0.0)
    assert config.rrf_k == 60


def test_nested_rrf_config_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "rrf.yaml"
    path.write_text("name: rrf\nranker:\n  kind: rrf\n  rrf_k: 30\n")

    config = load_experiment_config(path)

    assert config.ranker_kind == "rrf"
    assert config.rrf_k == 30


@pytest.mark.parametrize(
    "document, message",
    [
        ("ranker:\n  kind: unknown\n", "ranker.kind"),
        ("ranker:\n  kind: rrf\n  rrf_k: 0\n", "rrf_k"),
        (
            "ranker:\n  kind: percentile_linear\n  weights: [0.2, 0.2]\n",
            "sum to 1",
        ),
        (
            "ranker:\n  kind: percentile_linear\n  weights: [2.0, -1.0]\n",
            "non-negative",
        ),
    ],
)
def test_invalid_nested_ranker_is_rejected(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\n" + document)

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)

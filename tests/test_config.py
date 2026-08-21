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
    assert config.semantic_kind == "tfidf"
    assert config.semantic_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.semantic_model_revision is None
    assert config.semantic_cache_path is None
    assert config.semantic_device == "cpu"


def test_dense_semantic_config_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "dense.yaml"
    path.write_text(
        """semantic:
  kind: dense
  model_name: local/model
  model_revision: abc123
  cache_path: artifacts/dense.npz
  device: cuda
"""
    )

    config = load_experiment_config(path)

    assert config.semantic_kind == "dense"
    assert config.semantic_model_name == "local/model"
    assert config.semantic_model_revision == "abc123"
    assert config.semantic_cache_path == "artifacts/dense.npz"
    assert config.semantic_device == "cuda"


def test_learned_ranker_paths_are_loaded(tmp_path: Path) -> None:
    path = tmp_path / "learned.yaml"
    path.write_text(
        "ranker:\n  kind: lambdamart\n  model_path: artifacts/ranker.json\n"
        "  evidence_path: artifacts/validation.json\n"
    )

    config = load_experiment_config(path)

    assert config.ranker_kind == "lambdamart"
    assert config.learned_model_path == "artifacts/ranker.json"
    assert config.learned_evidence_path == "artifacts/validation.json"


def test_complete_learned_gate_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "selected.yaml"
    path.write_text(
        """ranker:
  kind: lambdamart
  model_path: model.json
  evidence_path: evidence.json
  bundle_manifest_path: bundle.json
  dataset_fingerprint: dataset
  candidate_policy_fingerprint: policy
  config_fingerprint: config
  case_fingerprint: cases
  gate_fingerprint: gate
  consumption_dir: artifacts/consumed
"""
    )
    config = load_experiment_config(path)
    assert config.learned_dataset_fingerprint == "dataset"
    assert config.learned_bundle_manifest_path == "bundle.json"
    assert config.learned_candidate_policy_fingerprint == "policy"
    assert config.learned_config_fingerprint == "config"
    assert config.learned_case_fingerprint == "cases"
    assert config.learned_gate_fingerprint == "gate"
    assert config.learned_consumption_dir == "artifacts/consumed"


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
        ("semantic:\n  kind: bm25\n", "semantic.kind"),
        ("semantic:\n  kind: dense\n  device: mps\n", "semantic.device"),
        ("semantic:\n  kind: dense\n  model_name: ''\n", "model_name"),
        ("semantic: false\n", "semantic must be a mapping"),
        ("semantic: []\n", "semantic must be a mapping"),
        ("semantic: ''\n", "semantic must be a mapping"),
        ("semantic: null\n", "semantic must be a mapping"),
        ("semantic:\n  kind: dense\n  cache_path: ''\n", "cache_path"),
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

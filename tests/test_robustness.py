from __future__ import annotations

import hashlib
import json

import pytest

from recagent_eval.evidence import artifact_fingerprint, canonical_digest
from recagent_eval.robustness import (
    POSTHOC_SEEDS,
    build_parameter_recovery_manifest,
    build_posthoc_robustness_input,
    summarize_posthoc_robustness,
    summarize_posthoc_robustness_input,
)


def _artifact(method: str, cohort: str, config_fingerprint: str) -> bytes:
    rows = [
        {
            "user_id": 1,
            "recall_at_10": 1.0,
            "ndcg_at_10": 0.5,
            "mrr_at_10": 0.5,
            "candidate_recall": 1.0,
            "constraint_satisfied": True,
            "latency_ms": 1.0,
        }
    ]
    value: dict[str, object] = {
        "schema_version": "baseline-evaluation/v1",
        "method": method,
        "cohort": cohort,
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": "d" * 64,
        "model_fingerprint": "m" * 64,
        "user_count": 1,
        "per_user_rows": rows,
        "aggregates": {
            "recall_at_10": 1.0,
            "ndcg_at_10": 0.5,
            "mrr_at_10": 0.5,
            "candidate_recall": 1.0,
            "constraint_satisfaction_rate": 1.0,
        },
    }
    value["fingerprint"] = artifact_fingerprint(value)
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_parameter_recovery_binds_each_cohort_source_artifact() -> None:
    selection = {
        "bpr_mf": {
            "selected_params": {"rank": 16},
            "grid": [{"rank": 16}, {"rank": 32}],
            "fingerprint": "f" * 64,
            "seed": 42,
        }
    }
    sources = {
        "confirmation_a": {"bpr_mf": _artifact("bpr_mf", "confirmation_a", "f" * 64)},
        "confirmation_b": {"bpr_mf": _artifact("bpr_mf", "confirmation_b", "f" * 64)},
    }
    manifest = build_parameter_recovery_manifest(
        selections=selection,
        source_artifacts=sources,
        command=".venv/bin/recagent-eval recover-baseline-params",
        commit_sha="a" * 40,
    )
    assert manifest["status"] == "recovered_after_run"
    assert manifest["fingerprint"] == canonical_digest(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    for cohort, artifacts in sources.items():
        record = manifest["cohorts"][cohort]["bpr_mf"]["selected_params"]
        assert record["source"] == "recovered"
        assert record["recovery"]["source_artifact_sha256"] == hashlib.sha256(
            artifacts["bpr_mf"]
        ).hexdigest()


def test_parameter_recovery_rejects_selection_fingerprint_drift() -> None:
    with pytest.raises(ValueError, match="selection fingerprint"):
        build_parameter_recovery_manifest(
            selections={
                "bpr_mf": {
                    "selected_params": {"rank": 16},
                    "grid": [],
                    "fingerprint": "f" * 64,
                    "seed": 42,
                }
            },
            source_artifacts={
                "confirmation_b": {
                    "bpr_mf": _artifact("bpr_mf", "confirmation_b", "x" * 64)
                }
            },
            command="recover",
            commit_sha="a" * 40,
        )


def test_posthoc_summary_reports_every_seed_mean_sample_std_and_worst() -> None:
    assert POSTHOC_SEEDS == (42, 7, 2026)
    seed_metrics = {
        "bpr_mf": {
            42: {"recall_at_10": 0.04, "ndcg_at_10": 0.02, "mrr_at_10": 0.01,
                 "candidate_recall": 1.0, "constraint_satisfaction_rate": 1.0},
            7: {"recall_at_10": 0.05, "ndcg_at_10": 0.03, "mrr_at_10": 0.02,
                "candidate_recall": 1.0, "constraint_satisfaction_rate": 1.0},
            2026: {"recall_at_10": 0.03, "ndcg_at_10": 0.01, "mrr_at_10": 0.01,
                   "candidate_recall": 1.0, "constraint_satisfaction_rate": 1.0},
        }
    }
    summary = summarize_posthoc_robustness(seed_metrics)
    row = summary["methods"]["bpr_mf"]
    assert [seed_row["seed"] for seed_row in row["seeds"]] == [42, 7, 2026]
    assert row["summary"]["ndcg_at_10"]["mean"] == pytest.approx(0.02)
    assert row["summary"]["ndcg_at_10"]["sample_std"] == pytest.approx(0.01)
    assert row["summary"]["ndcg_at_10"]["worst_seed"] == 2026


def test_posthoc_summary_rejects_missing_extra_or_duplicate_seed_identity() -> None:
    with pytest.raises(ValueError, match="exact seeds"):
        summarize_posthoc_robustness(
            {"bpr_mf": {42: {"ndcg_at_10": 0.1}, 7: {"ndcg_at_10": 0.1}}}
        )


def test_posthoc_input_normalizes_mixed_source_schemas_without_hiding_them() -> None:
    v1 = _artifact("bpr_mf", "confirmation_b", "f" * 64)
    v2_value = json.loads(v1)
    v2_value.update(
        schema_version="baseline-evaluation/v2",
        ordered_user_ids=[1],
        cohort_ledger_fingerprint="l" * 64,
        aggregates={
            "recall_at_10": 1.0,
            "ndcg_at_10": 0.5,
            "mrr_at_10": 0.5,
            "candidate_recall": 1.0,
            "constraint_satisfaction_rate": 1.0,
        },
        seed={"source": "observed", "value": 7},
    )
    def encoded_v2(seed: int) -> bytes:
        value = {**v2_value, "seed": {"source": "observed", "value": seed}}
        value["fingerprint"] = artifact_fingerprint(value)
        return (json.dumps(value, sort_keys=True) + "\n").encode()

    normalized = build_posthoc_robustness_input(
        source_artifacts={"bpr_mf": {42: v1, 7: encoded_v2(7), 2026: encoded_v2(2026)}},
        cohort="confirmation_b",
    )
    assert normalized["schema_version"] == "posthoc-robustness-input/v1"
    assert [row["source_schema"] for row in normalized["methods"]["bpr_mf"]] == [
        "baseline-evaluation/v1",
        "baseline-evaluation/v2",
        "baseline-evaluation/v2",
    ]
    assert normalized["methods"]["bpr_mf"][0]["metrics"]["source"] == "derived"
    summary = summarize_posthoc_robustness_input(normalized)
    assert summary["source_input_fingerprint"] == normalized["fingerprint"]


def test_posthoc_input_rejects_method_cohort_seed_or_fingerprint_drift() -> None:
    source = _artifact("bpr_mf", "confirmation_b", "f" * 64)
    with pytest.raises(ValueError, match="source method"):
        build_posthoc_robustness_input(
            source_artifacts={"lightgcn": {42: source}}, cohort="confirmation_b"
        )


def test_posthoc_normalization_and_summary_fail_closed_error_paths() -> None:
    source = _artifact("bpr_mf", "confirmation_b", "f" * 64)
    with pytest.raises(ValueError, match="invalid source artifact"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: b"{"}}, cohort="confirmation_b"
        )
    with pytest.raises(ValueError, match="source cohort"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: source}}, cohort="confirmation_a"
        )
    unknown = json.loads(source)
    unknown["schema_version"] = "unknown/v9"
    with pytest.raises(ValueError, match="unknown source schema"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: json.dumps(unknown).encode()}},
            cohort="confirmation_b",
        )
    drift = json.loads(source)
    drift["fingerprint"] = "drift"
    with pytest.raises(ValueError, match="source fingerprint drift"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: json.dumps(drift).encode()}},
            cohort="confirmation_b",
        )
    with pytest.raises(ValueError, match="legacy v1 source"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {7: source}}, cohort="confirmation_b"
        )
    missing = json.loads(source)
    missing.pop("aggregates")
    with pytest.raises(ValueError, match="aggregates missing"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: json.dumps(missing).encode()}},
            cohort="confirmation_b",
        )
    invalid = json.loads(source)
    invalid["aggregates"]["ndcg_at_10"] = float("nan")
    with pytest.raises(ValueError, match="invalid ndcg_at_10"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: json.dumps(invalid).encode()}},
            cohort="confirmation_b",
        )
    with pytest.raises(ValueError, match="exact seeds"):
        build_posthoc_robustness_input(
            source_artifacts={"bpr_mf": {42: source}}, cohort="confirmation_b"
        )

    with pytest.raises(ValueError, match="source methods"):
        build_parameter_recovery_manifest(
            selections={},
            source_artifacts={"confirmation_b": {"bpr_mf": source}},
            command="recover",
            commit_sha="a" * 40,
        )
    with pytest.raises(ValueError, match="invalid source artifact"):
        build_parameter_recovery_manifest(
            selections={"bpr_mf": {}},
            source_artifacts={"confirmation_b": {"bpr_mf": b"{"}},
            command="recover",
            commit_sha="a" * 40,
        )


def test_posthoc_summary_input_dispatch_rejects_malformed_layers() -> None:
    with pytest.raises(ValueError, match="unknown post-hoc"):
        summarize_posthoc_robustness_input({})

    def bound(methods: object) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "posthoc-robustness-input/v1",
            "methods": methods,
        }
        value["fingerprint"] = canonical_digest(value)
        return value

    with pytest.raises(ValueError, match="fingerprint drift"):
        summarize_posthoc_robustness_input(
            {"schema_version": "posthoc-robustness-input/v1", "fingerprint": "bad"}
        )
    with pytest.raises(ValueError, match="methods are missing"):
        summarize_posthoc_robustness_input(bound([]))
    with pytest.raises(ValueError, match="method is malformed"):
        summarize_posthoc_robustness_input(bound({1: []}))
    with pytest.raises(ValueError, match="row is malformed"):
        summarize_posthoc_robustness_input(bound({"bpr_mf": [1]}))
    with pytest.raises(ValueError, match="seed row is malformed"):
        summarize_posthoc_robustness_input(
            bound({"bpr_mf": [{"seed": True, "metrics": {}}]})
        )

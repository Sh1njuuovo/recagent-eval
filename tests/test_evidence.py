from __future__ import annotations

import copy

import pytest

from recagent_eval.evidence import (
    BASELINE_SCHEMA_V1,
    BASELINE_SCHEMA_V2,
    EvidenceValidationError,
    artifact_fingerprint,
    canonical_digest,
    provenance_value,
    validate_evidence_set,
)


def _rows(users: list[int]) -> list[dict[str, object]]:
    return [
        {
            "user_id": user,
            "recall_at_10": float(user % 2),
            "ndcg_at_10": 0.5 if user % 2 else 0.0,
            "mrr_at_10": 0.5 if user % 2 else 0.0,
            "candidate_recall": 1.0,
            "constraint_satisfied": True,
            "latency_ms": 1.0,
        }
        for user in users
    ]


def _v1(method: str, users: list[int], cohort: str = "confirmation_b") -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": BASELINE_SCHEMA_V1,
        "method": method,
        "cohort": cohort,
        "config_fingerprint": "c" * 64,
        "dataset_fingerprint": "d" * 64,
        "model_fingerprint": "m" * 64,
        "user_count": len(users),
        "per_user_rows": _rows(users),
    }
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    return artifact


def _v2(method: str, users: list[int], cohort: str = "confirmation_b") -> dict[str, object]:
    artifact = _v1(method, users, cohort)
    artifact.update(
        {
            "schema_version": BASELINE_SCHEMA_V2,
            "ordered_user_ids": users,
            "cohort_ledger_fingerprint": "l" * 64,
            "selected_params": provenance_value({}, source="observed"),
            "parameter_grid": provenance_value([], source="observed"),
            "seed": provenance_value(42, source="observed"),
            "dependency_versions": provenance_value(
                {"python": "3.13", "torch": "not_applicable"}, source="observed"
            ),
            "hardware": provenance_value(
                {"platform": "Darwin", "processor": "arm"}, source="observed"
            ),
            "resource_usage": {
                "metric_name": "process_peak_rss_mib",
                "raw_value": 1024,
                "raw_unit": "bytes",
                "normalized_mib": 1024 / 1024 / 1024,
                "platform": "Darwin",
            },
        }
    )
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    return artifact


def test_canonical_digest_is_order_independent_for_mapping_keys() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})


def test_recovered_provenance_requires_recovery_binding() -> None:
    with pytest.raises(EvidenceValidationError, match="recovery binding"):
        provenance_value({"rank": 20}, source="recovered")
    recovered = provenance_value(
        {"rank": 20},
        source="recovered",
        recovery={
            "status": "recovered_after_run",
            "command": ".venv/bin/recagent-eval recover-baseline-params",
            "source_artifact_sha256": "a" * 64,
            "input_fingerprint": "b" * 64,
            "output_fingerprint": "c" * 64,
            "commit_sha": "d" * 40,
        },
    )
    assert recovered["source"] == "recovered"


def test_validate_evidence_set_accepts_aligned_v1_and_v2_separately() -> None:
    users = [1, 2]
    v1 = validate_evidence_set(
        {"itemcf_direct": _v1("itemcf_direct", users)},
        cohort="confirmation_b",
        expected_user_ids=users,
        expected_ledger_fingerprint="l" * 64,
    )
    assert v1.schema_version == BASELINE_SCHEMA_V1
    v2 = validate_evidence_set(
        {"itemcf_direct": _v2("itemcf_direct", users)},
        cohort="confirmation_b",
        expected_user_ids=users,
        expected_ledger_fingerprint="l" * 64,
    )
    assert v2.schema_version == BASELINE_SCHEMA_V2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact.update(method="als_direct"), "method"),
        (lambda artifact: artifact.update(cohort="confirmation_a"), "cohort"),
        (lambda artifact: artifact.update(config_fingerprint=""), "fingerprint"),
        (lambda artifact: artifact.update(user_count=3), "user_count"),
        (
            lambda artifact: artifact["per_user_rows"].append(
                copy.deepcopy(artifact["per_user_rows"][0])
            ),
            "duplicate",
        ),
        (lambda artifact: artifact["per_user_rows"][0].update(ndcg_at_10=float("nan")), "finite"),
        (
            lambda artifact: artifact.update(schema_version="baseline-evaluation/v999"),
            "unknown schema",
        ),
    ],
)
def test_validate_evidence_set_rejects_invalid_identity_and_rows(
    mutation, message: str
) -> None:
    artifact = _v1("itemcf_direct", [1, 2])
    mutation(artifact)
    with pytest.raises(EvidenceValidationError, match=message):
        validate_evidence_set(
            {"itemcf_direct": artifact},
            cohort="confirmation_b",
            expected_user_ids=[1, 2],
            expected_ledger_fingerprint="l" * 64,
        )


def test_validate_evidence_set_rejects_mixed_schema_and_fingerprint_drift() -> None:
    with pytest.raises(EvidenceValidationError, match="mixed schema"):
        validate_evidence_set(
            {
                "itemcf_direct": _v1("itemcf_direct", [1, 2]),
                "als_direct": _v2("als_direct", [1, 2]),
            },
            cohort="confirmation_b",
            expected_user_ids=[1, 2],
            expected_ledger_fingerprint="l" * 64,
        )
    artifact = _v2("itemcf_direct", [1, 2])
    artifact["model_fingerprint"] = "x" * 64
    with pytest.raises(EvidenceValidationError, match="artifact fingerprint"):
        validate_evidence_set(
            {"itemcf_direct": artifact},
            cohort="confirmation_b",
            expected_user_ids=[1, 2],
            expected_ledger_fingerprint="l" * 64,
        )


def test_validate_v2_rejects_ledger_and_order_drift() -> None:
    artifact = _v2("itemcf_direct", [1, 2])
    artifact["cohort_ledger_fingerprint"] = "x" * 64
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    with pytest.raises(EvidenceValidationError, match="ledger"):
        validate_evidence_set(
            {"itemcf_direct": artifact},
            cohort="confirmation_b",
            expected_user_ids=[1, 2],
            expected_ledger_fingerprint="l" * 64,
        )

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

BASELINE_SCHEMA_V1 = "baseline-evaluation/v1"
BASELINE_SCHEMA_V2 = "baseline-evaluation/v2"
KNOWN_BASELINE_SCHEMAS = frozenset({BASELINE_SCHEMA_V1, BASELINE_SCHEMA_V2})
ProvenanceSource = Literal["observed", "derived", "recovered"]

_RECOVERY_FIELDS = {
    "status",
    "command",
    "source_artifact_sha256",
    "input_fingerprint",
    "output_fingerprint",
    "commit_sha",
}
_ROW_FLOAT_FIELDS = (
    "recall_at_10",
    "ndcg_at_10",
    "mrr_at_10",
    "candidate_recall",
    "latency_ms",
)


class EvidenceValidationError(ValueError):
    """Raised when an evidence identity or value fails closed validation."""


@dataclass(frozen=True)
class ValidatedEvidenceSet:
    schema_version: str
    artifacts: Mapping[str, Mapping[str, object]]
    ordered_user_ids: tuple[int, ...]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def artifact_fingerprint(artifact: Mapping[str, object]) -> str:
    schema = artifact.get("schema_version")
    if schema == BASELINE_SCHEMA_V1:
        rows = artifact.get("per_user_rows")
        if not isinstance(rows, list):
            raise EvidenceValidationError("v1 artifact has no per-user rows")
        payload = {
            "rows": [
                [
                    row["user_id"],
                    row["recall_at_10"],
                    row["ndcg_at_10"],
                    row["mrr_at_10"],
                ]
                for row in rows
            ],
            "method": artifact.get("method"),
            "cohort": artifact.get("cohort"),
        }
        return canonical_digest(payload)
    if schema == BASELINE_SCHEMA_V2:
        return canonical_digest(
            {key: value for key, value in artifact.items() if key != "fingerprint"}
        )
    raise EvidenceValidationError(f"unknown schema: {schema!r}")


def provenance_value(
    value: object,
    *,
    source: ProvenanceSource,
    recovery: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if source not in {"observed", "derived", "recovered"}:
        raise EvidenceValidationError(f"unknown provenance source: {source}")
    result: dict[str, object] = {"source": source, "value": value}
    if source == "recovered":
        if recovery is None or set(recovery) != _RECOVERY_FIELDS:
            raise EvidenceValidationError("recovered provenance requires complete recovery binding")
        if recovery.get("status") != "recovered_after_run" or any(
            not isinstance(recovery.get(field), str) or not recovery[field]
            for field in _RECOVERY_FIELDS - {"status"}
        ):
            raise EvidenceValidationError("recovered provenance has invalid recovery binding")
        result["recovery"] = dict(recovery)
    elif recovery is not None:
        raise EvidenceValidationError("only recovered provenance accepts a recovery binding")
    return result


def runtime_dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution, field in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("lightgbm", "lightgbm"),
        ("scikit-learn", "scikit_learn"),
    ):
        try:
            versions[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field] = "not_applicable"
    return versions


def runtime_hardware() -> dict[str, object]:
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or "unknown",
    }


def validate_evidence_set(
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    cohort: str,
    expected_user_ids: Sequence[int],
    expected_ledger_fingerprint: str,
) -> ValidatedEvidenceSet:
    if not artifacts:
        raise EvidenceValidationError("evidence set is empty")
    schemas = {artifact.get("schema_version") for artifact in artifacts.values()}
    unknown = schemas - KNOWN_BASELINE_SCHEMAS
    if unknown:
        raise EvidenceValidationError(f"unknown schema: {sorted(map(str, unknown))}")
    if len(schemas) != 1:
        raise EvidenceValidationError("mixed schema versions are not allowed")
    schema = str(next(iter(schemas)))
    expected = tuple(expected_user_ids)
    if len(expected) != len(set(expected)):
        raise EvidenceValidationError("ledger contains duplicate users")
    for slot_method, artifact in artifacts.items():
        _validate_artifact(
            artifact,
            slot_method=slot_method,
            cohort=cohort,
            expected_user_ids=expected,
            expected_ledger_fingerprint=expected_ledger_fingerprint,
            schema=schema,
        )
    return ValidatedEvidenceSet(schema, dict(artifacts), expected)


def _validate_artifact(
    artifact: Mapping[str, object],
    *,
    slot_method: str,
    cohort: str,
    expected_user_ids: tuple[int, ...],
    expected_ledger_fingerprint: str,
    schema: str,
) -> None:
    if artifact.get("method") != slot_method:
        raise EvidenceValidationError(
            f"artifact method does not match input slot {slot_method!r}"
        )
    if artifact.get("cohort") != cohort:
        raise EvidenceValidationError("artifact cohort does not match requested cohort")
    rows = artifact.get("per_user_rows")
    if not isinstance(rows, list) or not rows:
        raise EvidenceValidationError("baseline artifact has no per_user_rows")
    row_users = _validate_rows(rows)
    if len(row_users) != len(set(row_users)):
        raise EvidenceValidationError("artifact contains duplicate users")
    if tuple(row_users) != expected_user_ids:
        raise EvidenceValidationError("artifact users are missing, extra, or out of ledger order")
    if artifact.get("user_count") != len(rows):
        raise EvidenceValidationError("artifact user_count does not match rows")
    for field in ("config_fingerprint", "dataset_fingerprint", "model_fingerprint"):
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            raise EvidenceValidationError(f"artifact {field} is empty")
    if schema == BASELINE_SCHEMA_V2:
        if artifact.get("ordered_user_ids") != list(expected_user_ids):
            raise EvidenceValidationError("artifact ordered user IDs do not match ledger")
        if artifact.get("cohort_ledger_fingerprint") != expected_ledger_fingerprint:
            raise EvidenceValidationError("artifact cohort ledger fingerprint mismatch")
        for field in (
            "selected_params",
            "parameter_grid",
            "seed",
            "dependency_versions",
            "hardware",
        ):
            _validate_provenance_record(artifact.get(field), field=field)
        _validate_resource_usage(artifact.get("resource_usage"))
    recorded = artifact.get("fingerprint")
    if not isinstance(recorded, str) or recorded != artifact_fingerprint(artifact):
        raise EvidenceValidationError("artifact fingerprint drift")


def _validate_rows(rows: Sequence[object]) -> list[int]:
    users: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise EvidenceValidationError("per-user row must be an object")
        user = row.get("user_id")
        if not isinstance(user, int) or isinstance(user, bool):
            raise EvidenceValidationError("per-user row has invalid user_id")
        users.append(user)
        for field in _ROW_FLOAT_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EvidenceValidationError(f"per-user {field} must be numeric")
            if not math.isfinite(float(value)):
                raise EvidenceValidationError(f"per-user {field} must be finite")
        if type(row.get("constraint_satisfied")) is not bool:
            raise EvidenceValidationError("per-user constraint_satisfied must be boolean")
    return users


def _validate_provenance_record(value: object, *, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) - {"source", "value", "recovery"}:
        raise EvidenceValidationError(f"{field} provenance is malformed")
    source = value.get("source")
    try:
        provenance_value(
            value.get("value"),
            source=source,  # type: ignore[arg-type]
            recovery=value.get("recovery") if isinstance(value.get("recovery"), Mapping) else None,
        )
    except EvidenceValidationError as exc:
        raise EvidenceValidationError(f"{field} provenance is invalid: {exc}") from exc


def _validate_resource_usage(value: object) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceValidationError("resource_usage is missing")
    if value.get("metric_name") != "process_peak_rss_mib":
        raise EvidenceValidationError("resource_usage metric name is invalid")
    normalized = value.get("normalized_mib")
    if not isinstance(normalized, (int, float)) or not math.isfinite(float(normalized)):
        raise EvidenceValidationError("resource_usage normalized MiB must be finite")

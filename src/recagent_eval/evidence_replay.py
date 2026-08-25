from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from recagent_eval.baseline_eval import paired_bootstrap_deltas
from recagent_eval.evidence import (
    BASELINE_SCHEMA_V1,
    BASELINE_SCHEMA_V2,
    EvidenceValidationError,
    canonical_digest,
    provenance_value,
    validate_evidence_set,
)

BUNDLE_SCHEMA_VERSION = "baseline-evidence-bundle/v1"
GENERATOR_SCHEMA_VERSION = "evidence-bundle-generator/v1"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000

_COMPACT_FLOAT_FIELDS = (
    "recall_at_10",
    "ndcg_at_10",
    "mrr_at_10",
    "candidate_recall",
)


class EvidenceReplayError(ValueError):
    """Raised when a compact evidence chain cannot be verified."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_compact_bundle(
    *,
    source_artifacts: Mapping[str, bytes],
    ledger_bytes: bytes,
    summary_bytes: bytes,
    recovery: Mapping[str, object],
    cohort: str,
    commit_sha: str,
) -> dict[str, object]:
    try:
        artifacts = {method: json.loads(value) for method, value in source_artifacts.items()}
        ledger = json.loads(ledger_bytes)
        summary = json.loads(summary_bytes)
        expected_users = [int(user) for user in ledger["cohorts"][cohort]]
        ledger_fingerprint = str(ledger["fingerprint"])
        validated = validate_evidence_set(
            artifacts,
            cohort=cohort,
            expected_user_ids=expected_users,
            expected_ledger_fingerprint=ledger_fingerprint,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceReplayError(f"invalid bundle source evidence: {exc}") from exc
    _validate_summary_identity(summary, cohort=cohort, expected_users=expected_users)
    methods: dict[str, object] = {}
    for method, artifact in artifacts.items():
        source_sha = sha256_bytes(source_artifacts[method])
        provenance = _method_provenance(
            artifact,
            recovery.get(method),
            method=method,
            source_sha=source_sha,
        )
        rows = [_compact_row(row) for row in artifact["per_user_rows"]]
        methods[method] = {
            "method": method,
            "cohort": cohort,
            "source_schema_version": validated.schema_version,
            "source_artifact_sha256": source_sha,
            "source_artifact_fingerprint": artifact["fingerprint"],
            "config_fingerprint": artifact["config_fingerprint"],
            "dataset_fingerprint": artifact["dataset_fingerprint"],
            "model_fingerprint": artifact["model_fingerprint"],
            "ordered_user_ids": expected_users,
            "per_user_rows": rows,
            "canonical_per_user_digest": canonical_digest(rows),
            **provenance,
        }
    payload: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generator": {
            "name": "recagent_eval.evidence_replay",
            "schema_version": GENERATOR_SCHEMA_VERSION,
            "commit_sha": commit_sha,
        },
        "cohort": cohort,
        "cohort_ledger_fingerprint": ledger_fingerprint,
        "cohort_ledger_sha256": sha256_bytes(ledger_bytes),
        "summary_schema_version": summary["schema_version"],
        "summary_sha256": sha256_bytes(summary_bytes),
        "summary_digest": canonical_digest(summary),
        "summary_fingerprint": summary["fingerprint"],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "ordered_user_ids": expected_users,
        "methods": methods,
    }
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def replay_compact_bundle(
    bundle: Mapping[str, object],
    *,
    ledger_bytes: bytes,
    summary_bytes: bytes,
) -> dict[str, object]:
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise EvidenceReplayError(f"unknown bundle schema: {bundle.get('schema_version')!r}")
    generator = bundle.get("generator")
    if (
        not isinstance(generator, Mapping)
        or generator.get("schema_version") != GENERATOR_SCHEMA_VERSION
    ):
        raise EvidenceReplayError("unknown bundle generator schema")
    methods = bundle.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise EvidenceReplayError("bundle methods are missing")
    ordered = _ordered_users(bundle.get("ordered_user_ids"), context="bundle")
    compact_rows: dict[str, list[dict[str, object]]] = {}
    for slot_method, method_value in methods.items():
        if not isinstance(slot_method, str) or not isinstance(method_value, Mapping):
            raise EvidenceReplayError("bundle method entry is malformed")
        if method_value.get("method") != slot_method:
            raise EvidenceReplayError("bundle method identity mismatch")
        rows = _validate_compact_rows(method_value.get("per_user_rows"), ordered)
        if method_value.get("canonical_per_user_digest") != canonical_digest(rows):
            raise EvidenceReplayError(f"{slot_method} canonical per-user digest drift")
        compact_rows[slot_method] = rows
    recorded_fingerprint = bundle.get("fingerprint")
    computed_fingerprint = canonical_digest(
        {key: value for key, value in bundle.items() if key != "fingerprint"}
    )
    if recorded_fingerprint != computed_fingerprint:
        raise EvidenceReplayError("bundle fingerprint drift")
    if bundle.get("cohort_ledger_sha256") != sha256_bytes(ledger_bytes):
        raise EvidenceReplayError("cohort ledger SHA drift")
    if bundle.get("summary_sha256") != sha256_bytes(summary_bytes):
        raise EvidenceReplayError("committed summary SHA drift")
    try:
        ledger = json.loads(ledger_bytes)
        summary = json.loads(summary_bytes)
        cohort = str(bundle["cohort"])
        ledger_users = [int(user) for user in ledger["cohorts"][cohort]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceReplayError(f"invalid replay sidecar: {exc}") from exc
    if ledger.get("fingerprint") != bundle.get("cohort_ledger_fingerprint"):
        raise EvidenceReplayError("cohort ledger fingerprint drift")
    if ledger_users != ordered:
        raise EvidenceReplayError("bundle users do not match cohort ledger")
    if bundle.get("summary_digest") != canonical_digest(summary):
        raise EvidenceReplayError("summary digest drift")
    _validate_summary_identity(summary, cohort=cohort, expected_users=ordered)
    replayed = _replay_summary(summary, compact_rows, ordered)
    if replayed["fingerprint"] != bundle.get("summary_fingerprint"):
        raise EvidenceReplayError("summary fingerprint drift")
    if replayed != summary:
        raise EvidenceReplayError("replayed aggregate or bootstrap differs from summary")
    return replayed


def _method_provenance(
    artifact: Mapping[str, object],
    recovery_value: object,
    *,
    method: str,
    source_sha: str,
) -> dict[str, object]:
    if artifact.get("schema_version") == BASELINE_SCHEMA_V2:
        return {
            field: artifact[field]
            for field in ("selected_params", "parameter_grid", "seed")
        }
    if artifact.get("schema_version") != BASELINE_SCHEMA_V1:
        raise EvidenceReplayError(f"unknown artifact schema for {method}")
    if not isinstance(recovery_value, Mapping):
        raise EvidenceReplayError(f"v1 method {method} requires recovered provenance")
    result: dict[str, object] = {}
    for field in ("selected_params", "parameter_grid", "seed"):
        record = recovery_value.get(field)
        if not isinstance(record, Mapping):
            raise EvidenceReplayError(f"v1 method {method} recovery missing {field}")
        try:
            validated = provenance_value(
                record.get("value"),
                source=record.get("source"),  # type: ignore[arg-type]
                recovery=record.get("recovery")
                if isinstance(record.get("recovery"), Mapping)
                else None,
            )
        except EvidenceValidationError as exc:
            raise EvidenceReplayError(f"invalid recovered {field} for {method}: {exc}") from exc
        recovery_binding = validated.get("recovery")
        if (
            not isinstance(recovery_binding, Mapping)
            or recovery_binding.get("source_artifact_sha256") != source_sha
        ):
            raise EvidenceReplayError(f"recovered {field} source SHA drift for {method}")
        result[field] = validated
    return result


def _compact_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "user_id": row["user_id"],
        "recall_at_10": row["recall_at_10"],
        "ndcg_at_10": row["ndcg_at_10"],
        "mrr_at_10": row["mrr_at_10"],
        "candidate_recall": row["candidate_recall"],
        "constraint_satisfied": row["constraint_satisfied"],
    }


def _ordered_users(value: object, *, context: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(user, int) or isinstance(user, bool) for user in value
    ):
        raise EvidenceReplayError(f"{context} ordered user IDs are malformed")
    if len(value) != len(set(value)):
        raise EvidenceReplayError(f"{context} contains duplicate users")
    return list(value)


def _validate_compact_rows(value: object, ordered: Sequence[int]) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise EvidenceReplayError("bundle per-user rows are missing")
    rows = [dict(row) for row in value if isinstance(row, Mapping)]
    if len(rows) != len(value):
        raise EvidenceReplayError("bundle per-user row is malformed")
    users = [row.get("user_id") for row in rows]
    if len(users) != len(set(users)):
        raise EvidenceReplayError("bundle contains duplicate users")
    if users != list(ordered):
        raise EvidenceReplayError("bundle users are missing, extra, or out of order")
    for row in rows:
        for field in _COMPACT_FLOAT_FIELDS:
            number = row.get(field)
            if not isinstance(number, (int, float)) or isinstance(number, bool):
                raise EvidenceReplayError(f"bundle row {field} must be numeric")
            if not math.isfinite(float(number)):
                raise EvidenceReplayError(f"bundle row {field} must be finite")
        if type(row.get("constraint_satisfied")) is not bool:
            raise EvidenceReplayError("bundle row constraint result must be boolean")
    return rows


def _validate_summary_identity(
    summary: Mapping[str, object], *, cohort: str, expected_users: Sequence[int]
) -> None:
    if summary.get("schema_version") not in {"baseline-summary/v1", "baseline-summary/v2"}:
        raise EvidenceReplayError("unknown summary schema")
    if summary.get("cohort") != cohort:
        raise EvidenceReplayError("summary cohort drift")
    if summary.get("ordered_user_ids") != list(expected_users):
        raise EvidenceReplayError("summary ordered users drift")
    if summary.get("user_count") != len(expected_users):
        raise EvidenceReplayError("summary user count drift")
    recorded = summary.get("fingerprint")
    calculated = canonical_digest(
        {key: value for key, value in summary.items() if key != "fingerprint"}
    )
    if recorded != calculated:
        raise EvidenceReplayError("committed summary fingerprint drift")


def _replay_summary(
    summary: Mapping[str, object],
    rows_by_method: Mapping[str, Sequence[Mapping[str, object]]],
    ordered: Sequence[int],
) -> dict[str, object]:
    aggregates = {
        method: {
            "recall_at_10": _mean(rows, "recall_at_10"),
            "ndcg_at_10": _mean(rows, "ndcg_at_10"),
            "mrr_at_10": _mean(rows, "mrr_at_10"),
            "candidate_recall": _mean(rows, "candidate_recall"),
            "constraint_satisfaction_rate": sum(
                float(row["constraint_satisfied"]) for row in rows
            )
            / len(rows),
        }
        for method, rows in rows_by_method.items()
    }
    by_user = {
        method: {int(row["user_id"]): row for row in rows}
        for method, rows in rows_by_method.items()
    }
    pairwise: dict[str, object] = {}
    for left in sorted(rows_by_method):
        for right in sorted(rows_by_method):
            if left >= right:
                continue
            pairwise[f"{right}_vs_{left}"] = paired_bootstrap_deltas(
                [float(by_user[left][user]["ndcg_at_10"]) for user in ordered],
                [float(by_user[right][user]["ndcg_at_10"]) for user in ordered],
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {"aggregates", "pairwise_ndcg_bootstrap", "fingerprint"}
    }
    payload["aggregates"] = aggregates
    payload["pairwise_ndcg_bootstrap"] = pairwise
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)

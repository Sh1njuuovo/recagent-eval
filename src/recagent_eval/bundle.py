from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from recagent_eval.safe_io import ensure_distinct_files, read_regular_file

MAX_BUNDLE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_MANIFEST_BYTES = 64 * 1024
BUNDLE_SCHEMA_VERSION = "lambdamart-bundle/v1"


class RankerBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_sha256: str
    evidence_sha256: str
    run_fingerprint: str
    config_fingerprint: str
    dataset_fingerprint: str
    candidate_policy_fingerprint: str
    feature_fingerprint: str

    @model_validator(mode="after")
    def validate_contract(self) -> RankerBundleManifest:
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported ranker bundle schema")
        hashes = (self.model_sha256, self.evidence_sha256)
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("ranker bundle hashes must be lowercase SHA256 values")
        provenance = (
            self.run_fingerprint,
            self.config_fingerprint,
            self.dataset_fingerprint,
            self.candidate_policy_fingerprint,
            self.feature_fingerprint,
        )
        if any(not value.strip() for value in provenance):
            raise ValueError("ranker bundle provenance must be nonempty")
        return self


def publish_ranker_bundle(
    model_bytes: bytes,
    evidence_bytes: bytes,
    model_path: Path,
    evidence_path: Path,
    manifest_path: Path,
    metadata: Mapping[str, str],
) -> RankerBundleManifest:
    paths = {
        "model": model_path,
        "evidence": evidence_path,
        "bundle manifest": manifest_path,
    }
    ensure_distinct_files(paths)
    if len(model_bytes) > MAX_BUNDLE_MEMBER_BYTES or len(evidence_bytes) > MAX_BUNDLE_MEMBER_BYTES:
        raise ValueError("ranker bundle member exceeds maximum size")
    required = {
        "run_fingerprint",
        "config_fingerprint",
        "dataset_fingerprint",
        "candidate_policy_fingerprint",
        "feature_fingerprint",
    }
    if set(metadata) != required or any(not metadata[key] for key in required):
        raise ValueError("ranker bundle metadata is incomplete")
    manifest = RankerBundleManifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        **metadata,
    )
    manifest_bytes = (manifest.model_dump_json(indent=2) + "\n").encode()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    lock_paths = sorted(
        {
            path.with_name(f".{path.name}.lambdamart.lock")
            for path in paths.values()
        },
        key=lambda path: str(path.resolve(strict=False)),
    )
    ensure_distinct_files(
        {
            **paths,
            **{
                f"advisory lock {index}": path
                for index, path in enumerate(lock_paths)
            },
        }
    )
    lock_fds: list[int] = []
    try:
        for lock_path in lock_paths:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("unsafe ranker bundle advisory lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            lock_fds.append(descriptor)
        ensure_distinct_files(paths)
        existing = [label for label, path in paths.items() if os.path.lexists(path)]
        if existing:
            raise ValueError(
                "ranker bundle publication refuses to overwrite existing paths: "
                + ", ".join(existing)
            )
        temporary: list[str] = []
        try:
            model_temp = _write_temp(model_path, model_bytes)
            temporary.append(model_temp)
            evidence_temp = _write_temp(evidence_path, evidence_bytes)
            temporary.append(evidence_temp)
            manifest_temp = _write_temp(manifest_path, manifest_bytes)
            temporary.append(manifest_temp)
            os.replace(model_temp, model_path)
            temporary.remove(model_temp)
            os.replace(evidence_temp, evidence_path)
            temporary.remove(evidence_temp)
            os.replace(manifest_temp, manifest_path)
            temporary.remove(manifest_temp)
            _fsync_directories(paths.values())
        finally:
            for path in temporary:
                with suppress(FileNotFoundError):
                    os.unlink(path)
    finally:
        for descriptor in reversed(lock_fds):
            os.close(descriptor)
    return manifest


def load_ranker_bundle(
    model_path: Path,
    evidence_path: Path,
    manifest_path: Path,
    *,
    expected_metadata: Mapping[str, str] | None = None,
) -> tuple[bytes, bytes]:
    ensure_distinct_files(
        {"model": model_path, "evidence": evidence_path, "bundle manifest": manifest_path}
    )
    try:
        raw_manifest = read_regular_file(
            manifest_path, max_bytes=MAX_BUNDLE_MANIFEST_BYTES
        )
        manifest = RankerBundleManifest.model_validate_json(raw_manifest)
        if manifest.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported schema")
        model = read_regular_file(model_path, max_bytes=MAX_BUNDLE_MEMBER_BYTES)
        evidence = read_regular_file(evidence_path, max_bytes=MAX_BUNDLE_MEMBER_BYTES)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid ranker bundle manifest: {exc}") from exc
    if hashlib.sha256(model).hexdigest() != manifest.model_sha256:
        raise ValueError("ranker bundle model hash mismatch")
    if hashlib.sha256(evidence).hexdigest() != manifest.evidence_sha256:
        raise ValueError("ranker bundle evidence hash mismatch")
    if expected_metadata is not None:
        for field, expected in expected_metadata.items():
            if field not in RankerBundleManifest.model_fields:
                raise ValueError(f"unknown ranker bundle metadata field: {field}")
            if getattr(manifest, field) != expected:
                raise ValueError(f"ranker bundle {field} mismatch")
    return model, evidence


def _write_temp(destination: Path, payload: bytes) -> str:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _fsync_directories(paths) -> None:
    for directory in sorted({path.parent for path in paths}, key=str):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

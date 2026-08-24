from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PACKAGE_MEMBER_NAMES = (
    "model.json",
    "validation.json",
    "bundle.json",
    "latent.npz",
    "latent.npz.json",
    "semantic.npz",
    "semantic.npz.json",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def canonical_payload_sha256(payload: BaseModel | dict[str, object]) -> str:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    value.pop("fingerprint", None)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a normalized repository-relative path")
    if "\\" in value or "//" in value or value.startswith("./"):
        raise ValueError("path must be a normalized repository-relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("path must be a normalized repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("path must be a normalized repository-relative path")
    return value


def _validate_sha256(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase SHA-256")
    return value


class FileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    size_bytes: int = Field(gt=0)

    _path = field_validator("path")(validate_relative_path)
    _sha = field_validator("sha256")(_validate_sha256)


class SemanticCacheIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    immutable_revision: str = Field(min_length=1)
    dataset_fingerprint: str
    dimension: int = Field(gt=0)
    dtype: Literal["float32"]
    normalization: Literal["l2_unit"]
    cache_manifest_fingerprint: str

    _dataset_sha = field_validator("dataset_fingerprint")(_validate_sha256)
    _manifest_sha = field_validator("cache_manifest_fingerprint")(_validate_sha256)


class SourceInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-promotion-source-inventory/v1"]
    members: dict[str, FileIdentity]
    semantic: SemanticCacheIdentity
    provenance: Literal["observed_existing_bytes"]
    fingerprint: str

    _fingerprint_sha = field_validator("fingerprint")(_validate_sha256)

    @model_validator(mode="after")
    def validate_contract(self) -> SourceInventory:
        if set(self.members) != set(PACKAGE_MEMBER_NAMES):
            raise ValueError("source inventory package members are incomplete")
        if self.fingerprint != canonical_payload_sha256(self):
            raise ValueError("source inventory fingerprint mismatch")
        return self


class PromotionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-promotion/v1"]
    implementation_commit: str
    training_config_path: str
    training_config_fingerprint: str
    dataset_fingerprint: str
    case_fingerprint: str
    candidate_policy_fingerprint: str
    model_checksum: str
    feature_version: Literal["v2b"]
    feature_fingerprint: str
    score_calibration: Literal["raw", "percentile"]
    itemcf_top_k: int = Field(gt=0)
    semantic_top_k: int = Field(gt=0)
    latent_top_k: int = Field(gt=0)
    ordered_user_ids: tuple[int, ...]
    members: dict[str, FileIdentity]
    semantic: SemanticCacheIdentity

    _training_path = field_validator("training_config_path")(validate_relative_path)
    _training_sha = field_validator("training_config_fingerprint")(_validate_sha256)
    _dataset_sha = field_validator("dataset_fingerprint")(_validate_sha256)
    _case_sha = field_validator("case_fingerprint")(_validate_sha256)
    _policy_sha = field_validator("candidate_policy_fingerprint")(_validate_sha256)
    _model_sha = field_validator("model_checksum")(_validate_sha256)
    _feature_sha = field_validator("feature_fingerprint")(_validate_sha256)

    @field_validator("implementation_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("implementation_commit must be a lowercase full Git SHA")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> PromotionManifest:
        if set(self.members) != set(PACKAGE_MEMBER_NAMES):
            raise ValueError("promotion manifest package members are incomplete")
        if not self.ordered_user_ids:
            raise ValueError("ordered_user_ids must not be empty")
        if len(self.ordered_user_ids) != len(set(self.ordered_user_ids)):
            raise ValueError("ordered_user_ids contains duplicate users")
        return self


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["learned_frozen"]


@dataclass(frozen=True)
class ExecutionPaths:
    marker: str
    output: str
    command_log: str
    failure_log: str


def derive_execution_paths(
    *,
    manifest_sha256: str,
    case_fingerprint: str,
    dataset_fingerprint: str,
    model_checksum: str,
) -> ExecutionPaths:
    values = {
        "manifest_sha256": _validate_sha256(manifest_sha256),
        "case_fingerprint": _validate_sha256(case_fingerprint),
        "dataset_fingerprint": _validate_sha256(dataset_fingerprint),
        "model_checksum": _validate_sha256(model_checksum),
    }
    identity = canonical_payload_sha256(values)
    root = f"artifacts/frozen/{manifest_sha256[:16]}-{identity[:16]}"
    return ExecutionPaths(
        marker=f"{root}/marker.json",
        output=f"{root}/metrics.json",
        command_log=f"{root}/command.log",
        failure_log=f"{root}/failure.log",
    )


class PromotionYaml(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-promotion-execution/v1"]
    manifest_path: str
    manifest_sha256: str
    execution: ExecutionSettings
    training_config_fingerprint: str
    dataset_fingerprint: str
    case_fingerprint: str
    model_checksum: str
    marker_path: str
    output_path: str

    _manifest_path = field_validator("manifest_path")(validate_relative_path)
    _manifest_sha = field_validator("manifest_sha256")(_validate_sha256)
    _training_sha = field_validator("training_config_fingerprint")(_validate_sha256)
    _dataset_sha = field_validator("dataset_fingerprint")(_validate_sha256)
    _case_sha = field_validator("case_fingerprint")(_validate_sha256)
    _model_sha = field_validator("model_checksum")(_validate_sha256)
    _marker_path = field_validator("marker_path")(validate_relative_path)
    _output_path = field_validator("output_path")(validate_relative_path)

    def cross_check(self, manifest: PromotionManifest) -> None:
        if self.manifest_sha256 != canonical_payload_sha256(manifest):
            raise ValueError("promotion YAML manifest SHA mismatch")
        comparisons = {
            "training config fingerprint": (
                self.training_config_fingerprint,
                manifest.training_config_fingerprint,
            ),
            "dataset fingerprint": (self.dataset_fingerprint, manifest.dataset_fingerprint),
            "case fingerprint": (self.case_fingerprint, manifest.case_fingerprint),
            "model checksum": (self.model_checksum, manifest.model_checksum),
        }
        for label, (actual, expected) in comparisons.items():
            if actual != expected:
                raise ValueError(f"promotion YAML {label} mismatch")
        paths = derive_execution_paths(
            manifest_sha256=self.manifest_sha256,
            case_fingerprint=manifest.case_fingerprint,
            dataset_fingerprint=manifest.dataset_fingerprint,
            model_checksum=manifest.model_checksum,
        )
        if self.marker_path != paths.marker:
            raise ValueError("promotion YAML derived marker path mismatch")
        if self.output_path != paths.output:
            raise ValueError("promotion YAML derived output path mismatch")

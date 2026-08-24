from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from recagent_eval.safe_io import read_regular_file

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


def load_source_inventory(path: str | os.PathLike[str]) -> SourceInventory:
    try:
        raw = read_regular_file(Path(path), max_bytes=1024 * 1024)
        return SourceInventory.model_validate_json(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid promotion source inventory: {exc}") from exc


def verify_source_files(
    inventory: SourceInventory,
    source_paths: dict[str, os.PathLike[str]],
) -> None:
    if set(source_paths) != set(PACKAGE_MEMBER_NAMES):
        raise ValueError("source mapping package members are incomplete")
    for name in PACKAGE_MEMBER_NAMES:
        path = os.fspath(source_paths[name])
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError as exc:
            raise ValueError(f"source member {name} is missing") from exc
        except OSError as exc:
            raise ValueError(f"source member {name} is unsafe: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"source member {name} is not a unique regular file")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        expected = inventory.members[name]
        if metadata.st_size != expected.size_bytes or digest.hexdigest() != expected.sha256:
            raise ValueError(f"source member {name} identity mismatch")


def validate_semantic_source(
    inventory: SourceInventory | PromotionManifest,
    manifest_path: os.PathLike[str],
) -> None:
    try:
        raw = read_regular_file(Path(manifest_path), max_bytes=1024 * 1024)
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid semantic cache manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid semantic cache manifest: expected an object")
    semantic = inventory.semantic
    comparisons = {
        "model_name": semantic.model_name,
        "resolved_revision": semantic.immutable_revision,
        "dataset_fingerprint": semantic.dataset_fingerprint,
        "dimension": semantic.dimension,
        "embedding_dtype": semantic.dtype,
        "normalized": True,
    }
    for field, expected in comparisons.items():
        if payload.get(field) != expected:
            raise ValueError(f"semantic cache manifest {field} mismatch")
    if canonical_payload_sha256(payload) != semantic.cache_manifest_fingerprint:
        raise ValueError("semantic cache manifest fingerprint mismatch")


def _repo_destination(repo_root: Path, relative: str) -> Path:
    normalized = validate_relative_path(relative)
    root = repo_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("promotion destination escapes repository root")
    current = root
    for part in PurePosixPath(normalized).parts[:-1]:
        current /= part
        if os.path.lexists(current):
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("promotion destination contains an unsafe component")
    return candidate


def publish_promotion_package(
    repo_root: Path,
    inventory: SourceInventory,
    source_paths: dict[str, os.PathLike[str]],
) -> Path:
    verify_source_files(inventory, source_paths)
    validate_semantic_source(inventory, source_paths["semantic.npz.json"])
    expected_paths = {
        name: f"artifacts/promotion/current-v2b/{name}"
        for name in PACKAGE_MEMBER_NAMES
    }
    if any(inventory.members[name].path != expected for name, expected in expected_paths.items()):
        raise ValueError("source inventory destination paths are not canonical")
    final_path = _repo_destination(repo_root, "artifacts/promotion/current-v2b")
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _repo_destination(repo_root, "artifacts/promotion/current-v2b")
    if os.path.lexists(final_path):
        raise ValueError("promotion package publication refuses to overwrite final directory")
    build_path = Path(
        tempfile.mkdtemp(prefix=".current-v2b.build-", dir=parent)
    )
    published = False
    try:
        for name in PACKAGE_MEMBER_NAMES:
            source = os.fspath(source_paths[name])
            destination = build_path / name
            source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            destination_fd = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(source_fd, "rb") as source_stream, os.fdopen(
                    destination_fd, "wb"
                ) as destination_stream:
                    source_fd = -1
                    destination_fd = -1
                    shutil.copyfileobj(source_stream, destination_stream, 1024 * 1024)
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
            finally:
                if source_fd >= 0:
                    os.close(source_fd)
                if destination_fd >= 0:
                    os.close(destination_fd)
        verify_source_files(
            inventory,
            {name: build_path / name for name in PACKAGE_MEMBER_NAMES},
        )
        validate_semantic_source(inventory, build_path / "semantic.npz.json")
        directory_fd = os.open(build_path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if os.path.lexists(final_path):
            raise ValueError(
                "promotion package publication refuses to overwrite final directory"
            )
        os.rename(build_path, final_path)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and build_path.exists():
            shutil.rmtree(build_path)
    return final_path


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


class ReplayVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ordered_user_ids: tuple[int, ...]
    dataset_fingerprint: str
    model_checksum: str
    validation_rows_fingerprint: str
    validated_components: tuple[str, ...]

    _dataset_sha = field_validator("dataset_fingerprint")(_validate_sha256)
    _model_sha = field_validator("model_checksum")(_validate_sha256)
    _rows_sha = field_validator("validation_rows_fingerprint")(_validate_sha256)


class PreflightReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-promotion-preflight/v1"]
    manifest_sha256: str
    dataset_fingerprint: str
    model_checksum: str
    validation_rows_fingerprint: str
    validation_user_count: int = Field(gt=0)
    ordered_user_digest: str
    verified_member_sha256: dict[str, str]
    label_free: Literal[True]
    fingerprint: str

    _manifest_sha = field_validator("manifest_sha256")(_validate_sha256)
    _dataset_sha = field_validator("dataset_fingerprint")(_validate_sha256)
    _model_sha = field_validator("model_checksum")(_validate_sha256)
    _rows_sha = field_validator("validation_rows_fingerprint")(_validate_sha256)
    _users_sha = field_validator("ordered_user_digest")(_validate_sha256)
    _fingerprint_sha = field_validator("fingerprint")(_validate_sha256)

    @model_validator(mode="after")
    def validate_fingerprint(self) -> PreflightReceipt:
        if self.fingerprint != canonical_payload_sha256(self):
            raise ValueError("preflight receipt fingerprint mismatch")
        return self


def _regular_repo_file(repo_root: Path, relative: str, *, max_bytes: int) -> Path:
    path = _repo_destination(repo_root, relative)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"promotion file is missing: {relative}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"promotion file is unsafe: {relative}")
    if metadata.st_size > max_bytes:
        raise ValueError(f"promotion file is too large: {relative}")
    return path


def _file_sha_size(path: Path) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def preflight_promotion(
    repo_root: Path,
    promotion_yaml_path: Path,
    *,
    dataset_fingerprint_check,
    validation_replay,
    git_identity_check,
) -> PreflightReceipt:
    root = repo_root.resolve(strict=True)
    promotion_resolved = promotion_yaml_path.resolve(strict=True)
    if not promotion_resolved.is_relative_to(root / "reports/promotion"):
        raise ValueError("promotion YAML must be under reports/promotion")
    relative_promotion = promotion_resolved.relative_to(root).as_posix()
    promotion_file = _regular_repo_file(
        root, relative_promotion, max_bytes=1024 * 1024
    )
    try:
        promotion_payload = yaml.safe_load(
            read_regular_file(promotion_file, max_bytes=1024 * 1024)
        )
        promotion = PromotionYaml.model_validate(promotion_payload)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid promotion YAML: {exc}") from exc
    manifest_file = _regular_repo_file(
        root, promotion.manifest_path, max_bytes=4 * 1024 * 1024
    )
    try:
        manifest_payload = json.loads(
            read_regular_file(manifest_file, max_bytes=4 * 1024 * 1024)
        )
        manifest = PromotionManifest.model_validate(manifest_payload)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid promotion manifest: {exc}") from exc
    if canonical_payload_sha256(manifest_payload) != promotion.manifest_sha256:
        raise ValueError("promotion manifest canonical SHA mismatch")
    promotion.cross_check(manifest)
    member_paths: dict[str, Path] = {}
    verified_hashes: dict[str, str] = {}
    for name in PACKAGE_MEMBER_NAMES:
        identity = manifest.members[name]
        member_path = _regular_repo_file(
            root, identity.path, max_bytes=512 * 1024 * 1024
        )
        actual_sha, actual_size = _file_sha_size(member_path)
        if actual_sha != identity.sha256 or actual_size != identity.size_bytes:
            raise ValueError(f"promotion member {name} identity mismatch")
        member_paths[name] = member_path
        verified_hashes[name] = actual_sha
    validate_semantic_source(manifest, member_paths["semantic.npz.json"])
    git_identity_check(manifest)
    dataset_fingerprint = dataset_fingerprint_check(manifest, member_paths)
    if dataset_fingerprint != manifest.dataset_fingerprint:
        raise ValueError("complete dataset fingerprint mismatch")
    replay = ReplayVerification.model_validate(
        validation_replay(manifest, member_paths)
    )
    if replay.ordered_user_ids != manifest.ordered_user_ids:
        raise ValueError("validation replay ordered users mismatch")
    if replay.dataset_fingerprint != manifest.dataset_fingerprint:
        raise ValueError("validation replay dataset fingerprint mismatch")
    if replay.model_checksum != manifest.model_checksum:
        raise ValueError("validation replay model checksum mismatch")
    required_components = {"model", "evidence", "bundle", "latent", "semantic"}
    if set(replay.validated_components) != required_components:
        raise ValueError("validation replay component verification is incomplete")
    paths = derive_execution_paths(
        manifest_sha256=promotion.manifest_sha256,
        case_fingerprint=manifest.case_fingerprint,
        dataset_fingerprint=manifest.dataset_fingerprint,
        model_checksum=manifest.model_checksum,
    )
    for label, relative in {"marker": paths.marker, "output": paths.output}.items():
        destination = _repo_destination(root, relative)
        if os.path.lexists(destination):
            raise ValueError(f"real frozen {label} already exists")
    receipt_payload = {
        "schema_version": "frozen-promotion-preflight/v1",
        "manifest_sha256": promotion.manifest_sha256,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "model_checksum": manifest.model_checksum,
        "validation_rows_fingerprint": replay.validation_rows_fingerprint,
        "validation_user_count": len(replay.ordered_user_ids),
        "ordered_user_digest": canonical_payload_sha256(
            {"ordered_user_ids": list(replay.ordered_user_ids)}
        ),
        "verified_member_sha256": verified_hashes,
        "label_free": True,
    }
    receipt_payload["fingerprint"] = canonical_payload_sha256(receipt_payload)
    return PreflightReceipt.model_validate(receipt_payload)


def verify_git_identity(repo_root: Path, manifest: PromotionManifest) -> None:
    root = repo_root.resolve(strict=True)
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            manifest.implementation_commit,
            "HEAD",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("hardening implementation commit is not an ancestor of HEAD")
    protected = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"{manifest.implementation_commit}..HEAD",
            "--",
            "src",
            "configs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line for line in protected.stdout.splitlines() if line.strip()]
    if changed:
        raise ValueError(
            "protected production/config/dependency paths changed after implementation commit: "
            + ", ".join(changed)
        )

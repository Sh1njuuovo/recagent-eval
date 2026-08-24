from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recagent_eval.promotion import (
    PACKAGE_MEMBER_NAMES,
    FileIdentity,
    PromotionManifest,
    PromotionYaml,
    SemanticCacheIdentity,
    SourceInventory,
    canonical_payload_sha256,
    derive_execution_paths,
    load_source_inventory,
    publish_promotion_package,
    validate_relative_path,
    validate_semantic_source,
    verify_source_files,
)


def _sha(character: str) -> str:
    return character * 64


def _members() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": f"artifacts/promotion/current-v2b/{name}",
            "sha256": _sha(format(index, "x")),
            "size_bytes": index,
        }
        for index, name in enumerate(PACKAGE_MEMBER_NAMES, start=1)
    }


def _semantic() -> dict[str, object]:
    return {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "immutable_revision": "1" * 40,
        "dataset_fingerprint": _sha("a"),
        "dimension": 384,
        "dtype": "float32",
        "normalization": "l2_unit",
        "cache_manifest_fingerprint": _sha("b"),
    }


def test_source_inventory_is_strict_and_binds_all_seven_members() -> None:
    payload = {
        "schema_version": "frozen-promotion-source-inventory/v1",
        "members": _members(),
        "semantic": _semantic(),
        "provenance": "observed_existing_bytes",
    }
    payload["fingerprint"] = canonical_payload_sha256(payload)
    inventory = SourceInventory.model_validate(payload)
    assert set(inventory.members) == set(PACKAGE_MEMBER_NAMES)

    missing = copy.deepcopy(payload)
    missing["members"].pop("semantic.npz")
    missing["fingerprint"] = canonical_payload_sha256(missing)
    with pytest.raises(ValidationError, match="package members"):
        SourceInventory.model_validate(missing)

    drifted = copy.deepcopy(payload)
    drifted["members"]["model.json"]["size_bytes"] += 1
    with pytest.raises(ValidationError, match="fingerprint"):
        SourceInventory.model_validate(drifted)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/file",
        "../escape",
        "reports/../escape",
        "./reports/file",
        "reports//file",
        "reports\\file",
        "",
    ],
)
def test_promotion_paths_reject_non_normalized_or_escaping_values(path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        validate_relative_path(path)


def test_marker_and_output_paths_are_derived_from_bound_identity() -> None:
    paths = derive_execution_paths(
        manifest_sha256=_sha("1"),
        case_fingerprint=_sha("2"),
        dataset_fingerprint=_sha("3"),
        model_checksum=_sha("4"),
    )
    assert paths.marker.endswith("/marker.json")
    assert paths.output.endswith("/metrics.json")
    assert _sha("1")[:16] in paths.marker
    assert paths == derive_execution_paths(
        manifest_sha256=_sha("1"),
        case_fingerprint=_sha("2"),
        dataset_fingerprint=_sha("3"),
        model_checksum=_sha("4"),
    )


def test_manifest_and_yaml_cross_check_execution_without_training_identity_drift() -> None:
    manifest_payload = {
        "schema_version": "frozen-promotion/v1",
        "implementation_commit": "a" * 40,
        "training_config_path": "configs/v2_dense_latent_bfeat.yaml",
        "training_config_fingerprint": _sha("1"),
        "dataset_fingerprint": _sha("2"),
        "case_fingerprint": _sha("3"),
        "candidate_policy_fingerprint": _sha("4"),
        "model_checksum": _sha("5"),
        "feature_version": "v2b",
        "feature_fingerprint": _sha("6"),
        "score_calibration": "raw",
        "itemcf_top_k": 500,
        "semantic_top_k": 1500,
        "latent_top_k": 500,
        "ordered_user_ids": [9, 3],
        "members": _members(),
        "semantic": _semantic(),
    }
    manifest = PromotionManifest.model_validate(manifest_payload)
    manifest_sha = canonical_payload_sha256(manifest_payload)
    paths = derive_execution_paths(
        manifest_sha256=manifest_sha,
        case_fingerprint=manifest.case_fingerprint,
        dataset_fingerprint=manifest.dataset_fingerprint,
        model_checksum=manifest.model_checksum,
    )
    promotion = PromotionYaml.model_validate(
        {
            "schema_version": "frozen-promotion-execution/v1",
            "manifest_path": "reports/promotion/current-v2b-manifest.json",
            "manifest_sha256": manifest_sha,
            "execution": {"mode": "learned_frozen"},
            "training_config_fingerprint": manifest.training_config_fingerprint,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "case_fingerprint": manifest.case_fingerprint,
            "model_checksum": manifest.model_checksum,
            "marker_path": paths.marker,
            "output_path": paths.output,
        }
    )
    promotion.cross_check(manifest)

    changed_execution = promotion.model_copy(
        update={"execution": {"mode": "learned_frozen"}}
    )
    assert changed_execution.training_config_fingerprint == _sha("1")
    changed_marker = promotion.model_copy(update={"marker_path": "artifacts/frozen/x.json"})
    with pytest.raises(ValueError, match="derived marker"):
        changed_marker.cross_check(manifest)
    changed_candidate_contract = manifest.model_copy(update={"itemcf_top_k": 501})
    with pytest.raises(ValueError, match="manifest SHA"):
        promotion.cross_check(changed_candidate_contract)


def test_file_and_semantic_identity_reject_weak_values() -> None:
    with pytest.raises(ValidationError):
        FileIdentity(path="x", sha256="short", size_bytes=1)
    with pytest.raises(ValidationError):
        SemanticCacheIdentity.model_validate({**_semantic(), "dtype": "float64"})


def _inventory_for_sources(source_paths: dict[str, Path]) -> SourceInventory:
    members = {
        name: {
            "path": f"artifacts/promotion/current-v2b/{name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for name, path in source_paths.items()
    }
    payload = {
        "schema_version": "frozen-promotion-source-inventory/v1",
        "members": members,
        "semantic": _semantic(),
        "provenance": "observed_existing_bytes",
    }
    payload["fingerprint"] = canonical_payload_sha256(payload)
    return SourceInventory.model_validate(payload)


def test_source_verification_fails_closed_on_missing_or_changed_original(tmp_path) -> None:
    sources = {}
    for name in PACKAGE_MEMBER_NAMES:
        path = tmp_path / name.replace("/", "-")
        path.write_bytes(name.encode())
        sources[name] = path
    inventory = _inventory_for_sources(sources)
    verify_source_files(inventory, sources)

    sources["model.json"].write_bytes(b"retrained substitute")
    with pytest.raises(ValueError, match="model.json.*identity"):
        verify_source_files(inventory, sources)

    sources["model.json"].unlink()
    with pytest.raises(ValueError, match="model.json.*missing"):
        verify_source_files(inventory, sources)


def test_committed_source_inventory_is_strict_and_self_fingerprinted() -> None:
    path = Path("reports/promotion/current-v2b-source-inventory.json")
    inventory = load_source_inventory(path)
    assert inventory.fingerprint == canonical_payload_sha256(json.loads(path.read_text()))
    assert set(inventory.members) == set(PACKAGE_MEMBER_NAMES)


def _semantic_manifest(identity: SemanticCacheIdentity) -> dict[str, object]:
    return {
        "model_name": identity.model_name,
        "resolved_revision": identity.immutable_revision,
        "dataset_fingerprint": identity.dataset_fingerprint,
        "dimension": identity.dimension,
        "embedding_dtype": identity.dtype,
        "normalized": True,
    }


def _semantic_fixture() -> tuple[SemanticCacheIdentity, dict[str, object]]:
    payload = _semantic()
    provisional = SemanticCacheIdentity.model_validate(payload)
    manifest = _semantic_manifest(provisional)
    payload["cache_manifest_fingerprint"] = canonical_payload_sha256(manifest)
    return SemanticCacheIdentity.model_validate(payload), manifest


def test_atomic_package_publication_publishes_all_members_together(tmp_path) -> None:
    sources = {}
    for name in PACKAGE_MEMBER_NAMES:
        path = tmp_path / "sources" / name.replace("/", "-")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        sources[name] = path
    semantic, semantic_manifest = _semantic_fixture()
    sources["semantic.npz.json"].write_text(json.dumps(semantic_manifest))
    inventory = _inventory_for_sources(sources).model_copy(update={"semantic": semantic})
    inventory = inventory.model_copy(
        update={"fingerprint": canonical_payload_sha256(inventory.model_dump())}
    )
    validate_semantic_source(inventory, sources["semantic.npz.json"])

    destination = publish_promotion_package(tmp_path, inventory, sources)

    assert destination == tmp_path / "artifacts/promotion/current-v2b"
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        PACKAGE_MEMBER_NAMES
    )
    verify_source_files(
        inventory, {name: destination / name for name in PACKAGE_MEMBER_NAMES}
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        publish_promotion_package(tmp_path, inventory, sources)


def test_atomic_package_publication_rejects_unsafe_sources_and_rename_failure(
    tmp_path, monkeypatch
) -> None:
    sources = {}
    for name in PACKAGE_MEMBER_NAMES:
        path = tmp_path / "sources" / name.replace("/", "-")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        sources[name] = path
    semantic, semantic_manifest = _semantic_fixture()
    sources["semantic.npz.json"].write_text(json.dumps(semantic_manifest))
    inventory = _inventory_for_sources(sources).model_copy(update={"semantic": semantic})
    inventory = inventory.model_copy(
        update={"fingerprint": canonical_payload_sha256(inventory.model_dump())}
    )

    symlink = tmp_path / "semantic-link"
    symlink.symlink_to(sources["semantic.npz"])
    unsafe = {**sources, "semantic.npz": symlink}
    with pytest.raises(ValueError, match="unsafe"):
        publish_promotion_package(tmp_path, inventory, unsafe)

    hardlink = tmp_path / "model-hardlink"
    hardlink.hardlink_to(sources["model.json"])
    with pytest.raises(ValueError, match="unique regular"):
        publish_promotion_package(tmp_path, inventory, sources)
    hardlink.unlink()

    monkeypatch.setattr(
        "recagent_eval.promotion.os.rename",
        lambda *args: (_ for _ in ()).throw(OSError("simulated rename failure")),
    )
    with pytest.raises(OSError, match="rename failure"):
        publish_promotion_package(tmp_path, inventory, sources)
    assert not (tmp_path / "artifacts/promotion/current-v2b").exists()

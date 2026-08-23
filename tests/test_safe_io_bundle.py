from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from recagent_eval.bundle import load_ranker_bundle, publish_ranker_bundle
from recagent_eval.safe_io import ensure_distinct_files, read_regular_file


def _metadata() -> dict[str, str]:
    return {
        "run_fingerprint": "run",
        "config_fingerprint": "config",
        "dataset_fingerprint": "dataset",
        "candidate_policy_fingerprint": "policy",
        "feature_fingerprint": "feature",
    }


def test_bundle_publication_rejects_identical_and_hardlinked_paths(tmp_path: Path) -> None:
    same = tmp_path / "same.json"
    with pytest.raises(ValueError, match="distinct"):
        publish_ranker_bundle(
            b"model", b"evidence", same, same, tmp_path / "bundle.json", _metadata()
        )

    model = tmp_path / "model.json"
    model.write_bytes(b"old")
    evidence = tmp_path / "evidence.json"
    os.link(model, evidence)
    with pytest.raises(ValueError, match="alias"):
        publish_ranker_bundle(
            b"model",
            b"evidence",
            model,
            evidence,
            tmp_path / "bundle.json",
            _metadata(),
        )


def test_bundle_manifest_is_required_and_detects_mixed_pair(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"
    publish_ranker_bundle(b"model", b"evidence", model, evidence, manifest, _metadata())

    bundle = load_ranker_bundle(model, evidence, manifest)
    assert bundle.model_bytes == b"model"
    assert bundle.evidence_bytes == b"evidence"
    assert bundle.latent_bytes is None
    evidence.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="evidence hash"):
        load_ranker_bundle(model, evidence, manifest)
    manifest.unlink()
    with pytest.raises(ValueError, match="bundle manifest"):
        load_ranker_bundle(model, evidence, manifest)


def test_bundle_manifest_is_published_last(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"
    real_replace = os.replace

    def fail_before_manifest(source, destination):
        if Path(destination) == manifest:
            raise OSError("crash")
        real_replace(source, destination)

    monkeypatch.setattr("recagent_eval.bundle.os.replace", fail_before_manifest)
    with pytest.raises(OSError, match="crash"):
        publish_ranker_bundle(b"model", b"evidence", model, evidence, manifest, _metadata())

    assert model.read_bytes() == b"model"
    assert evidence.read_bytes() == b"evidence"
    assert not manifest.exists()
    with pytest.raises(ValueError, match="bundle manifest"):
        load_ranker_bundle(model, evidence, manifest)


def test_bundle_refuses_overwrite_and_preserves_existing_valid_pair(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"
    publish_ranker_bundle(b"old-model", b"old-evidence", model, evidence, manifest, _metadata())

    with pytest.raises(ValueError, match="refuses to overwrite"):
        publish_ranker_bundle(
            b"new-model", b"new-evidence", model, evidence, manifest, _metadata()
        )

    bundle = load_ranker_bundle(model, evidence, manifest)
    assert bundle.model_bytes == b"old-model"
    assert bundle.evidence_bytes == b"old-evidence"
    assert bundle.latent_bytes is None


def test_safe_reader_rejects_symlink_oversize_and_nonregular(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"ok")
    symlink = tmp_path / "link"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="safely open"):
        read_regular_file(symlink, max_bytes=10)

    with pytest.raises(ValueError, match="maximum size"):
        read_regular_file(target, max_bytes=1)
    with pytest.raises(ValueError, match="non-regular"):
        read_regular_file(tmp_path, max_bytes=10)


def test_safe_reader_rejects_partial_read(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"complete")
    real_read = os.read
    calls = 0

    def short_then_eof(fd: int, count: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_read(fd, 2)
        return b""

    monkeypatch.setattr("recagent_eval.safe_io.os.read", short_then_eof)
    with pytest.raises(ValueError, match="partially read"):
        read_regular_file(path, max_bytes=100)


def test_distinct_files_rejects_resolved_aliases(tmp_path: Path) -> None:
    value = tmp_path / "value.json"
    with pytest.raises(ValueError, match="distinct"):
        ensure_distinct_files({"input": value, "output": tmp_path / "." / "value.json"})


def test_bundle_manifest_schema_is_strict(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"
    publish_ranker_bundle(b"model", b"evidence", model, evidence, manifest, _metadata())
    payload = json.loads(manifest.read_text())
    payload["unexpected"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bundle manifest"):
        load_ranker_bundle(model, evidence, manifest)


def test_bundle_manifest_rejects_empty_provenance_even_when_rehashed(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"
    publish_ranker_bundle(b"model", b"evidence", model, evidence, manifest, _metadata())
    payload = json.loads(manifest.read_text())
    payload["config_fingerprint"] = ""
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bundle manifest"):
        load_ranker_bundle(model, evidence, manifest)


def test_concurrent_bundle_publishers_cannot_create_mixed_pair(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    evidence = tmp_path / "evidence.json"
    manifest = tmp_path / "bundle.json"

    def publish(label: str) -> str:
        try:
            publish_ranker_bundle(
                f"model-{label}".encode(),
                f"evidence-{label}".encode(),
                model,
                evidence,
                manifest,
                {**_metadata(), "run_fingerprint": label},
            )
        except ValueError as exc:
            assert "refuses to overwrite" in str(exc)
            return "rejected"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("first", "second")))

    assert sorted(outcomes) == ["published", "rejected"]
    bundle = load_ranker_bundle(model, evidence, manifest)
    model_bytes = bundle.model_bytes
    evidence_bytes = bundle.evidence_bytes
    assert model_bytes.removeprefix(b"model-") == evidence_bytes.removeprefix(
        b"evidence-"
    )


def test_bundle_v2_publishes_and_loads_latent_member(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"
    latent_path = tmp_path / "latent.npz"
    latent_manifest_path = tmp_path / "latent.npz.json"
    publish_ranker_bundle(
        b"model",
        b"evidence",
        model_path,
        evidence_path,
        manifest_path,
        _metadata(),
        latent_member=(latent_path, b"latent-data"),
        latent_manifest_member=(latent_manifest_path, b'{"checksum": "abc"}'),
    )
    bundle = load_ranker_bundle(
        model_path,
        evidence_path,
        manifest_path,
        latent_path=latent_path,
        latent_manifest_path=latent_manifest_path,
    )
    assert bundle.model_bytes == b"model"
    assert bundle.evidence_bytes == b"evidence"
    assert bundle.latent_bytes == b"latent-data"
    assert bundle.manifest.schema_version == "lambdamart-bundle/v2"


def test_bundle_v2_requires_latent_paths_and_rejects_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"
    latent_path = tmp_path / "latent.npz"
    latent_manifest_path = tmp_path / "latent.npz.json"
    publish_ranker_bundle(
        b"model",
        b"evidence",
        model_path,
        evidence_path,
        manifest_path,
        _metadata(),
        latent_member=(latent_path, b"latent-data"),
        latent_manifest_member=(latent_manifest_path, b'{"checksum": "abc"}'),
    )
    with pytest.raises(ValueError, match="latent"):
        load_ranker_bundle(model_path, evidence_path, manifest_path)
    other = tmp_path / "other.json"
    other.write_bytes(b'{"checksum": "different"}')
    with pytest.raises(ValueError, match="latent"):
        load_ranker_bundle(
            model_path,
            evidence_path,
            manifest_path,
            latent_path=latent_path,
            latent_manifest_path=other,
        )


def test_bundle_v1_rejects_latent_checksum_in_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"
    publish_ranker_bundle(
        b"m", b"e", model_path, evidence_path, manifest_path, _metadata()
    )
    payload = json.loads(manifest_path.read_text())
    payload["latent_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="v1 cannot carry latent"):
        load_ranker_bundle(model_path, evidence_path, manifest_path)


def test_bundle_latent_members_must_be_paired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        publish_ranker_bundle(
            b"m",
            b"e",
            tmp_path / "m.json",
            tmp_path / "e.json",
            tmp_path / "b.json",
            _metadata(),
            latent_member=(tmp_path / "latent.npz", b"data"),
        )

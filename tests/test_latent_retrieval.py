from __future__ import annotations

import json

import numpy as np
import pytest

from recagent_eval.data import Rating
from recagent_eval.latent_retrieval import LatentFactorRetriever, _read_regular_file


def _ratings() -> tuple[Rating, ...]:
    rows = []
    for user_id in range(1, 21):
        for movie_id in range(1, 9):
            rating = 5 if (user_id + movie_id) % 3 else 2
            rows.append(Rating(user_id, movie_id, rating, movie_id + user_id * 100))
    return tuple(rows)


def test_fit_is_bit_identical_across_two_runs() -> None:
    first = LatentFactorRetriever.fit(_ratings(), seed=42)
    second = LatentFactorRetriever.fit(_ratings(), seed=42)
    assert np.array_equal(first.item_factors, second.item_factors)
    assert first.training_fingerprint == second.training_fingerprint


def test_fit_changes_with_seed() -> None:
    first = LatentFactorRetriever.fit(_ratings(), seed=1)
    second = LatentFactorRetriever.fit(_ratings(), seed=2)
    assert not np.array_equal(first.item_factors, second.item_factors)


def test_retrieve_respects_top_k_ties_and_allowed_ids() -> None:
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    result = model.retrieve({1, 2, 3}, top_k=4, allowed_ids={3, 4, 5, 6, 7})
    ids = [movie_id for movie_id, _score in result]
    assert len(ids) == 4 and len(set(ids)) == 4
    assert all(movie_id in {3, 4, 5, 6, 7} for movie_id in ids)
    scores = dict(result)
    assert sorted(ids, key=lambda movie_id: (-scores[movie_id], movie_id)) == ids


def test_retrieve_empty_history_or_allowed_returns_empty() -> None:
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    assert model.retrieve(set(), top_k=5) == []
    assert model.retrieve({1, 2, 3}, top_k=5, allowed_ids=set()) == []


def test_retrieve_ignores_unseen_items_and_validates_top_k() -> None:
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    result = model.retrieve({999, 1, 2}, top_k=3)
    assert len(result) == 3
    with pytest.raises(ValueError, match="top_k"):
        model.retrieve({1}, top_k=0)


def test_retrieve_fold_in_matches_training_user_signal(tmp_path) -> None:
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    scores = dict(model.retrieve({1, 2, 3}, top_k=8))
    assert scores
    assert all(np.isfinite(value) for value in scores.values())


def test_save_load_roundtrip_and_checksum(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)
    manifest = json.loads((tmp_path / "latent.npz.json").read_text())
    assert manifest["training_fingerprint"] == model.training_fingerprint
    loaded = LatentFactorRetriever.load(path)
    assert loaded.rank == model.rank
    assert loaded.item_ids.tolist() == model.item_ids.tolist()
    assert np.array_equal(loaded.item_factors, model.item_factors)
    assert (
        LatentFactorRetriever.load(
            path, expected_training_fingerprint=model.training_fingerprint
        ).item_factors.shape
        == model.item_factors.shape
    )


def test_save_refuses_overwrite_and_corruption_fails(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)
    with pytest.raises(ValueError, match="overwrite"):
        model.save(path)
    manifest_path = tmp_path / "latent.npz.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_checksum"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        LatentFactorRetriever.load(path)


def test_invalid_hyperparameters_fail() -> None:
    with pytest.raises(ValueError, match="rank"):
        LatentFactorRetriever.fit(_ratings(), rank=0)
    with pytest.raises(ValueError, match="alpha"):
        LatentFactorRetriever.fit(_ratings(), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        LatentFactorRetriever.fit(_ratings(), alpha=float("nan"))
    with pytest.raises(ValueError, match="lambda_reg"):
        LatentFactorRetriever.fit(_ratings(), lambda_reg=-0.1)


def test_fit_requires_positive_rows() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        LatentFactorRetriever.fit((Rating(1, 1, 2, 1),))


def test_retrieve_unknown_history_returns_empty() -> None:
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    assert model.retrieve({9999}, top_k=5) == []


def test_save_refuses_when_manifest_exists(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    (tmp_path / "latent.npz.json").write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        LatentFactorRetriever.fit(_ratings(), seed=42).save(path)


def test_save_rejects_oversize_artifact(tmp_path, monkeypatch) -> None:
    import recagent_eval.latent_retrieval as module

    monkeypatch.setattr(module, "MAX_LATENT_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="too large"):
        LatentFactorRetriever.fit(_ratings(), seed=42).save(tmp_path / "latent.npz")


def test_load_rejects_missing_files_and_tampered_data(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)
    with pytest.raises(ValueError, match="latent artifact manifest"):
        LatentFactorRetriever.load(tmp_path / "missing.npz")
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    with pytest.raises(ValueError, match="checksum"):
        LatentFactorRetriever.load(path)


def test_load_rejects_wrong_training_fingerprint(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)
    with pytest.raises(ValueError, match="training fingerprint"):
        LatentFactorRetriever.load(path, expected_training_fingerprint="0" * 64)


def test_load_rejects_unsafe_npz_contents(tmp_path, monkeypatch) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)

    class FakePayload:
        files = {"item_ids", "item_factors", "evil"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "recagent_eval.latent_retrieval.np.load", lambda *args, **kwargs: FakePayload()
    )
    with pytest.raises(ValueError, match="unsafe latent artifact"):
        LatentFactorRetriever.load(path)


def test_load_rejects_malformed_manifest_fields(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    model = LatentFactorRetriever.fit(_ratings(), seed=42)
    model.save(path)
    manifest_path = tmp_path / "latent.npz.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dtype"] = "float64"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dtype"):
        LatentFactorRetriever.load(path)


def test_read_regular_file_rejects_nonregular_and_oversize(tmp_path) -> None:
    with pytest.raises(ValueError, match="file type"):
        _read_regular_file(tmp_path, max_bytes=10**6)
    path = tmp_path / "value"
    path.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="too large"):
        _read_regular_file(path, max_bytes=5)

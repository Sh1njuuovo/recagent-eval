import hashlib
import io
import json
import os
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from recagent_eval.data import Movie
from recagent_eval.retrieval import (
    DenseSemanticRetriever,
    SemanticRetriever,
    TfidfSemanticRetriever,
    movie_catalog_fingerprint,
)

MOVIES = {
    30: Movie(30, "Third", ("Drama",)),
    10: Movie(10, "First", ("Sci-Fi",)),
    20: Movie(20, "Second", ("Comedy",)),
}


class FakeEncoder:
    model_revision = "fake-revision"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        rows = {
            MOVIES[10].text: [3.0, 0.0],
            MOVIES[20].text: [0.0, 4.0],
            MOVIES[30].text: [6.0, 0.0],
            "space": [2.0, 0.0],
            "comedy": [0.0, 7.0],
        }
        return np.asarray([rows[text] for text in texts], dtype=np.float64)


def test_semantic_retriever_protocol_accepts_tfidf_and_dense() -> None:
    assert isinstance(TfidfSemanticRetriever.fit(MOVIES), SemanticRetriever)
    assert isinstance(
        DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake"),
        SemanticRetriever,
    )


def test_dense_fit_normalizes_float32_and_breaks_ties_by_movie_id() -> None:
    retriever = DenseSemanticRetriever.fit(
        MOVIES,
        encoder=FakeEncoder(),
        model_name="fake",
    )

    assert retriever.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(retriever.embeddings, axis=1), 1.0)
    assert retriever.retrieve("space") == [(10, 1.0), (30, 1.0), (20, 0.0)]


def test_dense_fit_records_encoder_resolved_revision_over_requested_alias() -> None:
    retriever = DenseSemanticRetriever.fit(
        MOVIES,
        encoder=FakeEncoder(),
        model_name="fake",
        model_revision="release-tag",
    )

    assert retriever.model_revision == "fake-revision"
    assert retriever.requested_revision == "release-tag"


def test_dense_fit_normalizes_finite_extreme_float32_vectors() -> None:
    class ExtremeEncoder:
        model_revision = "extreme-revision"

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[3e38, 0.0] for _ in texts], dtype=np.float32)

    retriever = DenseSemanticRetriever.fit(
        MOVIES,
        encoder=ExtremeEncoder(),
        model_name="extreme",
    )

    assert np.isfinite(retriever.embeddings).all()
    np.testing.assert_allclose(
        np.linalg.norm(retriever.embeddings.astype(np.float64), axis=1),
        1.0,
        atol=1e-6,
    )


def test_dense_retrieve_filters_allowed_ids_and_handles_empty_inputs() -> None:
    encoder = FakeEncoder()
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=encoder, model_name="fake")

    assert retriever.retrieve("comedy", top_k=1, allowed_ids={10, 20}) == [(20, 1.0)]
    call_count = len(encoder.calls)
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("space", allowed_ids=set()) == []
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("space", top_k=0)
    assert len(encoder.calls) == call_count


@pytest.mark.parametrize(
    "bad_rows",
    [
        [[float("nan"), 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[float("inf"), 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ],
)
def test_dense_fit_rejects_non_finite_and_zero_embeddings(bad_rows: list[list[float]]) -> None:
    class BadEncoder:
        model_revision = "bad-revision"

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.asarray(bad_rows[: len(texts)])

    with pytest.raises(ValueError, match="finite|non-zero"):
        DenseSemanticRetriever.fit(MOVIES, encoder=BadEncoder(), model_name="bad")


def test_dense_retrieve_rejects_non_finite_query_embedding() -> None:
    encoder = FakeEncoder()
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=encoder, model_name="fake")
    encoder.encode = lambda texts: np.asarray([[np.nan, 1.0]])  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="finite"):
        retriever.retrieve("broken")


def test_dataset_fingerprint_is_independent_of_mapping_order() -> None:
    reordered = {movie_id: MOVIES[movie_id] for movie_id in reversed(MOVIES)}

    assert movie_catalog_fingerprint(MOVIES) == movie_catalog_fingerprint(reordered)
    assert movie_catalog_fingerprint(MOVIES) != movie_catalog_fingerprint(
        {**MOVIES, 20: Movie(20, "Changed", ("Comedy",))}
    )


def test_dense_cache_round_trip_has_inspectable_manifest(tmp_path) -> None:
    cache = tmp_path / "movies.npz"
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    retriever.save(cache)

    manifest = json.loads((tmp_path / "movies.npz.json").read_text())
    assert manifest.keys() == {
        "schema_version",
        "model_name",
        "requested_revision",
        "resolved_revision",
        "dataset_fingerprint",
        "dimension",
        "generated_at",
        "normalized",
        "movie_ids",
        "embedding_checksum",
        "device",
        "encoder_metadata",
        "runtime_metadata",
        "library_versions",
        "item_text_schema",
        "embedding_shape",
        "embedding_dtype",
    }
    assert manifest["movie_ids"] == [10, 20, 30]
    assert manifest["normalized"] is True
    assert manifest["device"] == "cpu"
    assert manifest["item_text_schema"] == "v1-movie-text-title-genres"
    assert manifest["embedding_shape"] == [3, 2]
    assert manifest["embedding_dtype"] == "float32"
    assert manifest["encoder_metadata"]["class"].endswith("FakeEncoder")
    assert manifest["runtime_metadata"]["python_version"]
    assert manifest["library_versions"]["numpy"] == np.__version__
    loaded = DenseSemanticRetriever.load(
        cache,
        movies=MOVIES,
        encoder=FakeEncoder(),
        model_name="fake",
    )
    assert loaded.retrieve("space") == retriever.retrieve("space")


def test_dense_cache_rejects_old_manifest_and_provenance_mismatch(tmp_path) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    manifest_path = tmp_path / "movies.npz.json"
    original = json.loads(manifest_path.read_text())

    old_manifest = dict(original)
    old_manifest.pop("item_text_schema")
    manifest_path.write_text(json.dumps(old_manifest))
    with pytest.raises(ValueError, match="manifest fields"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")

    for field, replacement, message in (
        ("device", "cuda", "device"),
        ("encoder_metadata", {"class": "tampered.Encoder"}, "encoder"),
        (
            "runtime_metadata",
            {**original["runtime_metadata"], "python_version": "0.0"},
            "runtime",
        ),
        (
            "library_versions",
            {**original["library_versions"], "numpy": "0.0"},
            "library",
        ),
    ):
        manifest_path.write_text(json.dumps({**original, field: replacement}))
        with pytest.raises(ValueError, match=message):
            DenseSemanticRetriever.load(
                cache,
                movies=MOVIES,
                encoder=FakeEncoder(),
                model_name="fake",
                device="cpu",
            )


@pytest.mark.parametrize("field,value", [("model_name", "other"), ("model_revision", "other")])
def test_dense_cache_rejects_model_metadata_mismatch(tmp_path, field: str, value: str) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    kwargs: dict[str, Any] = {
        "movies": MOVIES,
        "encoder": FakeEncoder(),
        "model_name": "fake",
        "model_revision": None,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        DenseSemanticRetriever.load(cache, **kwargs)


def test_dense_cache_reuses_requested_alias_and_preserves_resolved_sha(tmp_path) -> None:
    resolved_sha = "a" * 40

    class AliasEncoder(FakeEncoder):
        model_revision = resolved_sha

    cache = tmp_path / "alias.npz"
    DenseSemanticRetriever.fit(
        MOVIES,
        encoder=AliasEncoder(),
        model_name="fake",
        model_revision="main",
    ).save(cache)

    by_alias = DenseSemanticRetriever.validate_cache(
        cache,
        movies=MOVIES,
        model_name="fake",
        model_revision="main",
        encoder_type=AliasEncoder,
    )
    by_sha = DenseSemanticRetriever.validate_cache(
        cache,
        movies=MOVIES,
        model_name="fake",
        model_revision=resolved_sha,
        encoder_type=AliasEncoder,
    )
    assert by_alias["requested_revision"] == "main"
    assert by_alias["resolved_revision"] == resolved_sha
    assert by_sha == by_alias


def test_dense_cache_rejects_dataset_dimension_checksum_and_duplicate_ids(tmp_path) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    manifest_path = tmp_path / "movies.npz.json"
    original = json.loads(manifest_path.read_text())

    changed_movies = {**MOVIES, 20: Movie(20, "Changed", ("Comedy",))}
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        DenseSemanticRetriever.load(
            cache, movies=changed_movies, encoder=FakeEncoder(), model_name="fake"
        )

    manifest = {**original, "dimension": 3}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dimension"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")

    manifest = {**original, "embedding_checksum": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")

    with np.load(cache, allow_pickle=False) as payload:
        embeddings = payload["embeddings"]
    duplicate_ids = np.asarray([10, 10, 30], dtype=np.int64)
    np.savez(cache, movie_ids=duplicate_ids, embeddings=embeddings)
    manifest = {
        **original,
        "movie_ids": duplicate_ids.tolist(),
        "embedding_checksum": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_rejects_partial_cache(tmp_path) -> None:
    cache = tmp_path / "partial.npz"
    np.savez(cache, movie_ids=np.asarray([1]), embeddings=np.asarray([[1.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="partial"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_rejects_movie_ids_that_do_not_match_catalog(tmp_path) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    manifest_path = tmp_path / "movies.npz.json"
    manifest = json.loads(manifest_path.read_text())
    altered_ids = np.asarray([10, 20, 40], dtype=np.int64)
    with np.load(cache, allow_pickle=False) as payload:
        embeddings = payload["embeddings"]
    np.savez(cache, movie_ids=altered_ids, embeddings=embeddings)
    manifest["movie_ids"] = altered_ids.tolist()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="movie IDs"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


@pytest.mark.parametrize("replacement", [[0.0, 0.0], [0.5, 0.0]])
def test_dense_cache_rejects_zero_or_non_unit_stored_rows(
    tmp_path,
    replacement: list[float],
) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    manifest_path = tmp_path / "movies.npz.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(cache, allow_pickle=False) as payload:
        movie_ids = payload["movie_ids"]
        embeddings = payload["embeddings"]
    embeddings[0] = replacement
    np.savez(cache, movie_ids=movie_ids, embeddings=embeddings)
    manifest["embedding_checksum"] = hashlib.sha256(embeddings.tobytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="normalized"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_save_does_not_follow_predictable_temp_symlink(tmp_path) -> None:
    cache = tmp_path / "movies.npz"
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    (tmp_path / ".movies.npz.tmp").symlink_to(victim)

    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)

    assert victim.read_text() == "untouched"


def test_dense_cache_concurrent_saves_use_distinct_secure_tempfiles(
    tmp_path,
    monkeypatch,
) -> None:
    cache = tmp_path / "movies.npz"
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    seen: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        descriptor, path = real_mkstemp(*args, **kwargs)
        seen.append(path)
        return descriptor, path

    monkeypatch.setattr("recagent_eval.retrieval.tempfile.mkstemp", recording_mkstemp)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: retriever.save(cache), range(2)))

    assert len(seen) == 4
    assert len(set(seen)) == 4
    DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_cleans_first_temp_if_second_creation_fails(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "movies.npz"
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    real_mkstemp = tempfile.mkstemp
    calls = 0

    def fail_second_mkstemp(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected tempfile failure")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr("recagent_eval.retrieval.tempfile.mkstemp", fail_second_mkstemp)

    with pytest.raises(OSError, match="injected"):
        retriever.save(cache)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_portable_lock_cleans_owned_file_if_token_write_fails(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "movies.npz"
    retriever = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    monkeypatch.setattr(
        "recagent_eval.retrieval.os.write",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("token write failed")),
    )

    with pytest.raises(OSError, match="token write failed"):
        retriever.save(cache)
    assert not Path(f"{cache}.lock").exists()


def test_dense_cache_rejects_oversized_files_before_numpy_load(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)

    monkeypatch.setattr("recagent_eval.retrieval.MAX_MANIFEST_BYTES", 1)
    with pytest.raises(ValueError, match="manifest.*large"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")

    monkeypatch.setattr("recagent_eval.retrieval.MAX_MANIFEST_BYTES", 1024 * 1024)
    monkeypatch.setattr("recagent_eval.retrieval.MAX_CACHE_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="archive.*large"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")

    monkeypatch.setattr("recagent_eval.retrieval.MAX_CACHE_ARCHIVE_BYTES", 512 * 1024 * 1024)
    monkeypatch.setattr("recagent_eval.retrieval.MAX_CACHE_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="uncompressed.*large"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_rejects_dimension_limit_before_numpy_load(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)
    monkeypatch.setattr("recagent_eval.retrieval.MAX_EMBEDDING_DIMENSION", 1)

    with pytest.raises(ValueError, match="dimension.*large"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_rejects_malicious_npy_shape_before_numpy_load(
    tmp_path,
    monkeypatch,
) -> None:
    cache = tmp_path / "movies.npz"
    DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake").save(cache)

    def header(shape: tuple[int, ...], dtype: np.dtype) -> bytes:
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream,
            {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape},
        )
        return stream.getvalue()

    with zipfile.ZipFile(cache, "w") as archive:
        archive.writestr("movie_ids.npy", header((3,), np.dtype(np.int64)))
        archive.writestr(
            "embeddings.npy",
            header((1_000_001, 2), np.dtype(np.float32)),
        )
    monkeypatch.setattr(
        "recagent_eval.retrieval.np.load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("np.load called")),
    )

    with pytest.raises(ValueError, match="array header"):
        DenseSemanticRetriever.load(cache, movies=MOVIES, encoder=FakeEncoder(), model_name="fake")


def test_dense_cache_load_uses_stable_open_archive_if_path_is_swapped(
    tmp_path,
    monkeypatch,
) -> None:
    cache = tmp_path / "movies.npz"
    original = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    original.save(cache)
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(b"attacker replacement")
    from recagent_eval import retrieval

    inspect_archive = retrieval._inspect_npz_archive

    def inspect_then_swap(archive, manifest):
        result = inspect_archive(archive, manifest)
        os.replace(replacement, cache)
        return result

    monkeypatch.setattr(retrieval, "_inspect_npz_archive", inspect_then_swap)

    loaded = DenseSemanticRetriever.load(
        cache,
        movies=MOVIES,
        encoder=FakeEncoder(),
        model_name="fake",
    )
    np.testing.assert_array_equal(loaded.embeddings, original.embeddings)


def test_portable_lock_pairs_concurrent_read_and_write_without_fcntl(
    tmp_path,
    monkeypatch,
) -> None:
    cache = tmp_path / "movies.npz"
    initial = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    initial.save(cache)
    writer = DenseSemanticRetriever.fit(MOVIES, encoder=FakeEncoder(), model_name="fake")
    writer.embeddings[:] = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    data_published = threading.Event()
    continue_publish = threading.Event()
    real_replace = os.replace

    def paused_replace(source, destination):
        real_replace(source, destination)
        if Path(destination) == cache:
            data_published.set()
            continue_publish.wait(timeout=2)

    monkeypatch.setattr("recagent_eval.retrieval.fcntl", None)
    monkeypatch.setattr("recagent_eval.retrieval.os.replace", paused_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        write_future = pool.submit(writer.save, cache)
        assert data_published.wait(timeout=2)
        read_future = pool.submit(
            DenseSemanticRetriever.load,
            cache,
            movies=MOVIES,
            encoder=FakeEncoder(),
            model_name="fake",
        )
        time.sleep(0.05)
        continue_publish.set()
        write_future.result(timeout=2)
        loaded = read_future.result(timeout=2)

    np.testing.assert_array_equal(loaded.embeddings, writer.embeddings)


def test_alias_validation_instantiates_model_at_stored_resolved_sha(
    tmp_path,
    monkeypatch,
) -> None:
    resolved_sha = "b" * 40

    class RecordingSentenceEncoder:
        revisions: list[str | None] = []
        model_revision = resolved_sha

        def __init__(self, model_name: str, *, revision: str | None, device: str) -> None:
            type(self).revisions.append(revision)

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.eye(len(texts), 2, dtype=np.float32)

    monkeypatch.setattr(
        "recagent_eval.retrieval.SentenceTransformerEncoder",
        RecordingSentenceEncoder,
    )
    cache = tmp_path / "alias.npz"
    DenseSemanticRetriever.fit(
        {10: MOVIES[10], 20: MOVIES[20]},
        model_name="fake",
        model_revision="main",
    ).save(cache)

    DenseSemanticRetriever.load(
        cache,
        movies={10: MOVIES[10], 20: MOVIES[20]},
        model_name="fake",
        model_revision="main",
    )

    assert RecordingSentenceEncoder.revisions == ["main", resolved_sha]

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import platform
import re
import stat
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

import numpy as np

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState

try:  # pragma: no cover - the alternate backend is exercised on Windows
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - platform dependent
    _fcntl = None

try:  # pragma: no cover - the alternate backend is exercised on POSIX
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - platform dependent
    _msvcrt = None

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_CACHE_SCHEMA_VERSION = 3
DENSE_ITEM_TEXT_SCHEMA = "v1-movie-text-title-genres"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CACHE_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_CACHE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_CACHE_ROWS = 1_000_000
MAX_EMBEDDING_DIMENSION = 16_384
MAX_EMBEDDING_BYTES = 512 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 4096
CACHE_LOCK_TIMEOUT_SECONDS = 5.0
CACHE_LOCK_POLL_SECONDS = 0.01


@runtime_checkable
class SemanticRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]: ...


class EmbeddingEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def hard_filter(
    movies: Iterable[Movie],
    state: PreferenceState,
) -> list[Movie]:
    blocked_ids = state.excluded_movie_ids | state.disliked_movie_ids | state.liked_movie_ids
    kept: list[Movie] = []
    for movie in movies:
        genres = set(movie.genres)
        if movie.movie_id in blocked_ids:
            continue
        if state.year_min is not None and (movie.year is None or movie.year < state.year_min):
            continue
        if state.year_max is not None and (movie.year is None or movie.year > state.year_max):
            continue
        if state.required_genres and not state.required_genres.issubset(genres):
            continue
        if state.excluded_genres & genres:
            continue
        kept.append(movie)
    return kept


@dataclass(frozen=True)
class ItemCFRetriever:
    similarities: dict[int, dict[int, float]]
    popularity: dict[int, int]

    @classmethod
    def fit(
        cls,
        ratings: Iterable[Rating],
        *,
        positive_threshold: int = 4,
    ) -> ItemCFRetriever:
        user_items: dict[int, set[int]] = defaultdict(set)
        popularity: Counter[int] = Counter()
        for row in ratings:
            if row.rating >= positive_threshold:
                user_items[row.user_id].add(row.movie_id)
                popularity[row.movie_id] += 1

        cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
        for items in user_items.values():
            for left in items:
                for right in items:
                    if left != right:
                        cooccurrence[left][right] += 1

        similarities: dict[int, dict[int, float]] = defaultdict(dict)
        for left, neighbors in cooccurrence.items():
            for right, count in neighbors.items():
                denominator = math.sqrt(popularity[left] * popularity[right])
                similarities[left][right] = count / denominator if denominator else 0.0
        return cls(dict(similarities), dict(popularity))

    def retrieve(
        self,
        history: set[int],
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores: Counter[int] = Counter()
        for source in history:
            for movie_id, similarity in self.similarities.get(source, {}).items():
                if movie_id not in history:
                    scores[movie_id] += similarity
        if not scores:
            for movie_id, count in self.popularity.items():
                if movie_id not in history:
                    scores[movie_id] = float(count)
        ranked = [
            (movie_id, float(score))
            for movie_id, score in scores.items()
            if allowed_ids is None or movie_id in allowed_ids
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_k]


@dataclass(frozen=True)
class TfidfSemanticRetriever:
    movies: dict[int, Movie]
    vectors: dict[int, dict[str, float]]
    idf: dict[str, float]

    @classmethod
    def fit(cls, movies: dict[int, Movie]) -> TfidfSemanticRetriever:
        documents = {movie_id: _tokens(movie.text) for movie_id, movie in movies.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))
        count = max(len(documents), 1)
        idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        vectors = {movie_id: _tfidf_vector(tokens, idf) for movie_id, tokens in documents.items()}
        return cls(movies=dict(movies), vectors=vectors, idf=idf)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = _tfidf_vector(_tokens(query), self.idf)
        scores = []
        for movie_id, vector in self.vectors.items():
            if allowed_ids is not None and movie_id not in allowed_ids:
                continue
            score = sum(query_vector.get(key, 0.0) * value for key, value in vector.items())
            if score > 0:
                scores.append((movie_id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]


class SentenceTransformerEncoder:
    """Lazy adapter that keeps the optional ML dependency out of TF-IDF runs."""

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        *,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires the ML extra; install with "
                "`pip install 'recagent-eval[ml]'` or `uv sync --extra ml`."
            ) from exc
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        resolved = _sentence_transformer_revision(self._model)
        self.requested_revision = revision
        self.resolved_revision = resolved or revision
        self.model_revision = self.resolved_revision
        if not self.resolved_revision:
            raise RuntimeError(
                "Could not resolve the model revision; pass an immutable --model-revision."
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )


@dataclass(frozen=True)
class DenseSemanticRetriever:
    movie_ids: np.ndarray
    embeddings: np.ndarray
    encoder: EmbeddingEncoder
    model_name: str
    model_revision: str
    dataset_fingerprint: str
    device: str
    requested_revision: str | None

    @classmethod
    def fit(
        cls,
        movies: dict[int, Movie],
        *,
        encoder: EmbeddingEncoder | None = None,
        model_name: str = DEFAULT_DENSE_MODEL,
        model_revision: str | None = None,
        device: str = "cpu",
    ) -> DenseSemanticRetriever:
        if device not in {"cpu", "cuda"}:
            raise ValueError("dense retrieval device must be cpu or cuda")
        active_encoder = encoder or SentenceTransformerEncoder(
            model_name,
            revision=model_revision,
            device=device,
        )
        resolved_revision = _encoder_revision(active_encoder, model_revision)
        movie_ids = np.asarray(sorted(movies), dtype=np.int64)
        if movie_ids.size:
            texts = [movies[int(movie_id)].text for movie_id in movie_ids]
            embeddings = _normalized_embeddings(active_encoder.encode(texts), len(texts))
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)
        return cls(
            movie_ids=movie_ids,
            embeddings=embeddings,
            encoder=active_encoder,
            model_name=model_name,
            model_revision=resolved_revision,
            dataset_fingerprint=movie_catalog_fingerprint(movies),
            device=device,
            requested_revision=model_revision,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self.movie_ids.size:
            return []
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            mask = np.isin(self.movie_ids, np.fromiter(allowed_ids, dtype=np.int64))
            candidate_ids = self.movie_ids[mask]
            candidate_embeddings = self.embeddings[mask]
        else:
            candidate_ids = self.movie_ids
            candidate_embeddings = self.embeddings
        if not candidate_ids.size:
            return []
        query_embedding = _normalized_embeddings(self.encoder.encode([query]), 1)
        if query_embedding.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "query embedding dimension does not match cached embedding dimension"
            )
        scores = candidate_embeddings @ query_embedding[0]
        if not np.isfinite(scores).all():
            raise ValueError("similarity scores must be finite")
        order = np.lexsort((candidate_ids, -scores))[:top_k]
        return [(int(candidate_ids[index]), float(scores[index])) for index in order]

    def save(self, path: Path) -> None:
        cache_path = Path(path)
        manifest_path = _manifest_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        _validate_normalized_matrix(embeddings)
        manifest = {
            "schema_version": DENSE_CACHE_SCHEMA_VERSION,
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.model_revision,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dimension": int(embeddings.shape[1]),
            "generated_at": datetime.now(UTC).isoformat(),
            "normalized": True,
            "movie_ids": self.movie_ids.tolist(),
            "embedding_checksum": _embedding_checksum(embeddings),
            "device": self.device,
            "encoder_metadata": {"class": _encoder_identifier(self.encoder)},
            "runtime_metadata": _runtime_metadata(),
            "library_versions": _library_versions(),
            "item_text_schema": DENSE_ITEM_TEXT_SCHEMA,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
        }
        _validate_manifest(manifest)
        with _cache_lock(cache_path):
            data_fd = -1
            manifest_fd = -1
            data_temp: Path | None = None
            manifest_temp: Path | None = None
            try:
                data_fd, data_name = tempfile.mkstemp(
                    prefix=f".{cache_path.name}.", suffix=".tmp", dir=cache_path.parent
                )
                data_temp = Path(data_name)
                manifest_fd, manifest_name = tempfile.mkstemp(
                    prefix=f".{manifest_path.name}.", suffix=".tmp", dir=cache_path.parent
                )
                manifest_temp = Path(manifest_name)
                with os.fdopen(data_fd, "wb") as stream:
                    data_fd = -1
                    np.savez(stream, movie_ids=self.movie_ids, embeddings=embeddings)
                    stream.flush()
                    os.fsync(stream.fileno())
                if data_temp.stat().st_size > MAX_CACHE_ARCHIVE_BYTES:
                    raise ValueError("dense cache archive is too large")
                inspection_fd = _open_regular_file(
                    data_temp,
                    max_bytes=MAX_CACHE_ARCHIVE_BYTES,
                    too_large_message="dense cache archive is too large",
                )
                with os.fdopen(inspection_fd, "rb") as inspection_stream:
                    _inspect_npz_archive(inspection_stream, manifest)
                manifest_bytes = (
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                    raise ValueError("dense cache manifest is too large")
                with os.fdopen(manifest_fd, "wb") as stream:
                    manifest_fd = -1
                    stream.write(manifest_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(data_temp, cache_path)
                os.replace(manifest_temp, manifest_path)
                _fsync_directory(cache_path.parent)
            finally:
                if data_fd >= 0:
                    os.close(data_fd)
                if manifest_fd >= 0:
                    os.close(manifest_fd)
                if data_temp is not None:
                    data_temp.unlink(missing_ok=True)
                if manifest_temp is not None:
                    manifest_temp.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        movies: dict[int, Movie],
        encoder: EmbeddingEncoder | None = None,
        model_name: str = DEFAULT_DENSE_MODEL,
        model_revision: str | None = None,
        device: str | None = None,
    ) -> DenseSemanticRetriever:
        manifest, movie_ids, embeddings = _read_dense_cache(
            Path(path),
            movies=movies,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
        )
        # An alias is only used to validate requested provenance. Actual loading is
        # pinned to the already-resolved immutable revision stored in the cache.
        revision_to_load = str(manifest["resolved_revision"])
        active_device = device or str(manifest["device"])
        active_encoder = encoder or SentenceTransformerEncoder(
            model_name,
            revision=revision_to_load,
            device=active_device,
        )
        expected_revision = _encoder_revision(active_encoder, revision_to_load)
        if manifest["resolved_revision"] != expected_revision:
            raise ValueError("dense cache model_revision mismatch")
        if manifest["encoder_metadata"] != {"class": _encoder_identifier(active_encoder)}:
            raise ValueError("dense cache encoder metadata mismatch")
        return cls(
            movie_ids=np.ascontiguousarray(movie_ids, dtype=np.int64),
            embeddings=np.ascontiguousarray(embeddings, dtype=np.float32),
            encoder=active_encoder,
            model_name=model_name,
            model_revision=expected_revision,
            dataset_fingerprint=movie_catalog_fingerprint(movies),
            device=active_device,
            requested_revision=manifest["requested_revision"],  # type: ignore[arg-type]
        )

    @classmethod
    def validate_cache(
        cls,
        path: Path,
        *,
        movies: dict[int, Movie],
        model_name: str = DEFAULT_DENSE_MODEL,
        model_revision: str | None = None,
        device: str | None = None,
        encoder_type: type[object] | None = None,
    ) -> dict[str, object]:
        """Validate a SentenceTransformer cache without loading model weights."""
        manifest, _, _ = _read_dense_cache(
            Path(path),
            movies=movies,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
        )
        expected_encoder = {
            "class": _encoder_identifier(encoder_type or SentenceTransformerEncoder)
        }
        if manifest["encoder_metadata"] != expected_encoder:
            raise ValueError("dense cache encoder metadata mismatch")
        return manifest


def movie_catalog_fingerprint(movies: dict[int, Movie]) -> str:
    canonical = json.dumps(
        [[movie_id, movies[movie_id].text] for movie_id in sorted(movies)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_embeddings(values: object, expected_rows: int) -> np.ndarray:
    embeddings = np.asarray(values, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError("encoder must return one two-dimensional embedding per text")
    if embeddings.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings must be non-zero")
    normalized = np.asarray(embeddings / norms, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized embeddings must contain only finite values")
    _validate_normalized_matrix(normalized)
    return np.ascontiguousarray(normalized)


def _encoder_revision(encoder: EmbeddingEncoder, requested: str | None) -> str:
    resolved = getattr(encoder, "model_revision", None) or getattr(
        encoder, "resolved_revision", None
    )
    revision = str(resolved or requested or "").strip()
    if not revision:
        raise ValueError("an injected encoder must provide a resolved model_revision")
    return revision


def _sentence_transformer_revision(model: object) -> str | None:
    try:
        first_module = model[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None
    auto_model = getattr(first_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None


def _manifest_path(cache_path: Path) -> Path:
    return Path(f"{cache_path}.json")


def _embedding_checksum(embeddings: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(embeddings).tobytes()).hexdigest()


@contextmanager
def _cache_lock(cache_path: Path):
    lock_path = Path(f"{cache_path}.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError("unsafe dense cache lock path") from exc
    locked = False
    backend: str | None = None
    try:
        opened_stat = os.fstat(descriptor)
        try:
            path_stat = os.lstat(lock_path)
        except OSError as exc:
            raise ValueError("unsafe dense cache lock path") from exc
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ValueError("unsafe dense cache lock path")
        if _fcntl is not None:
            backend = "fcntl"
            _acquire_advisory_lock(descriptor, lock_path, backend)
        elif _msvcrt is not None:
            backend = "msvcrt"
            _ensure_windows_lock_byte(descriptor, opened_stat)
            _acquire_advisory_lock(descriptor, lock_path, backend)
        else:
            raise RuntimeError(
                "no supported OS advisory lock backend is available "
                "(requires fcntl on POSIX or msvcrt on Windows)"
            )
        locked = True
        yield
    finally:
        if locked:
            _release_advisory_lock(descriptor, backend)
        os.close(descriptor)


def _acquire_advisory_lock(descriptor: int, lock_path: Path, backend: str) -> None:
    deadline = time.monotonic() + CACHE_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            if backend == "fcntl":
                assert _fcntl is not None
                _fcntl.flock(descriptor, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            else:
                assert _msvcrt is not None
                os.lseek(descriptor, 0, os.SEEK_SET)
                _msvcrt.locking(descriptor, _msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for dense cache lock {lock_path}") from exc
            time.sleep(CACHE_LOCK_POLL_SECONDS)


def _ensure_windows_lock_byte(descriptor: int, opened_stat: os.stat_result) -> None:
    if opened_stat.st_size:
        return
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.write(descriptor, b"\0") != 1:
        raise OSError("short write while initializing dense cache lock")
    os.fsync(descriptor)


def _release_advisory_lock(descriptor: int, backend: str | None) -> None:
    try:
        if backend == "fcntl":
            assert _fcntl is not None
            _fcntl.flock(descriptor, _fcntl.LOCK_UN)
        elif backend == "msvcrt":
            assert _msvcrt is not None
            os.lseek(descriptor, 0, os.SEEK_SET)
            _msvcrt.locking(descriptor, _msvcrt.LK_UNLCK, 1)
    except OSError:
        # Closing the descriptor below is the final, crash-safe release mechanism.
        pass


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoder_identifier(encoder: object) -> str:
    encoder_type = encoder if isinstance(encoder, type) else type(encoder)
    return f"{encoder_type.__module__}.{encoder_type.__qualname__}"


def _runtime_metadata() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def _library_versions() -> dict[str, str | None]:
    try:
        sentence_transformers_version = version("sentence-transformers")
    except PackageNotFoundError:
        sentence_transformers_version = None
    return {
        "numpy": np.__version__,
        "sentence_transformers": sentence_transformers_version,
    }


def _validate_manifest(manifest: object) -> None:
    required = {
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
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("unsafe dense embedding cache manifest fields")
    if manifest["schema_version"] != DENSE_CACHE_SCHEMA_VERSION:
        raise ValueError("dense cache schema_version mismatch")
    if manifest["normalized"] is not True:
        raise ValueError("dense cache normalized flag mismatch")
    if not isinstance(manifest["dimension"], int) or manifest["dimension"] < 0:
        raise ValueError("dense cache dimension is invalid")
    if manifest["dimension"] > MAX_EMBEDDING_DIMENSION:
        raise ValueError("dense cache dimension is too large")
    if manifest["device"] not in {"cpu", "cuda"}:
        raise ValueError("dense cache device is invalid")
    if manifest["item_text_schema"] != DENSE_ITEM_TEXT_SCHEMA:
        raise ValueError("dense cache item_text_schema mismatch")
    if manifest["embedding_dtype"] != "float32":
        raise ValueError("dense cache embedding dtype mismatch")
    shape = manifest["embedding_shape"]
    if not isinstance(shape, list) or len(shape) != 2 or not all(
        isinstance(value, int) and value >= 0 for value in shape
    ):
        raise ValueError("dense cache embedding shape is invalid")
    encoder_metadata = manifest["encoder_metadata"]
    if (
        not isinstance(encoder_metadata, dict)
        or set(encoder_metadata) != {"class"}
        or not isinstance(encoder_metadata["class"], str)
        or not encoder_metadata["class"]
    ):
        raise ValueError("dense cache encoder metadata is invalid")
    runtime_metadata = manifest["runtime_metadata"]
    if (
        not isinstance(runtime_metadata, dict)
        or set(runtime_metadata) != {"python_implementation", "python_version"}
        or not all(isinstance(value, str) and value for value in runtime_metadata.values())
    ):
        raise ValueError("dense cache runtime metadata is invalid")
    library_versions = manifest["library_versions"]
    if (
        not isinstance(library_versions, dict)
        or set(library_versions) != {"numpy", "sentence_transformers"}
        or not isinstance(library_versions["numpy"], str)
        or not library_versions["numpy"]
        or not (
            library_versions["sentence_transformers"] is None
            or isinstance(library_versions["sentence_transformers"], str)
        )
    ):
        raise ValueError("dense cache library metadata is invalid")
    if not isinstance(manifest["movie_ids"], list) or not all(
        isinstance(movie_id, int) for movie_id in manifest["movie_ids"]
    ):
        raise ValueError("dense cache movie_ids are invalid")
    if len(manifest["movie_ids"]) != len(set(manifest["movie_ids"])):
        raise ValueError("dense cache contains duplicate movie IDs")
    if len(manifest["movie_ids"]) > MAX_CACHE_ROWS:
        raise ValueError("dense cache row count is too large")
    if shape != [len(manifest["movie_ids"]), manifest["dimension"]]:
        raise ValueError("dense cache embedding shape/dimension mismatch")
    if len(manifest["movie_ids"]) * manifest["dimension"] * 4 > MAX_EMBEDDING_BYTES:
        raise ValueError("dense cache expected embedding bytes are too large")
    try:
        generated_at = datetime.fromisoformat(str(manifest["generated_at"]))
    except ValueError as exc:
        raise ValueError("dense cache generated_at is invalid") from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() != UTC.utcoffset(None):
        raise ValueError("dense cache generated_at must be UTC")
    for field in ("model_name", "resolved_revision", "dataset_fingerprint"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"dense cache {field} is invalid")
    requested_revision = manifest["requested_revision"]
    if requested_revision is not None and (
        not isinstance(requested_revision, str) or not requested_revision
    ):
        raise ValueError("dense cache requested_revision is invalid")
    checksum = manifest["embedding_checksum"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("dense cache embedding checksum is invalid")


def _validate_cached_arrays(
    movie_ids: np.ndarray,
    embeddings: np.ndarray,
    manifest: dict[str, object],
) -> None:
    if movie_ids.ndim != 1 or movie_ids.dtype.kind not in "iu":
        raise ValueError("unsafe dense cache movie ID array")
    ids = [int(movie_id) for movie_id in movie_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("dense cache contains duplicate movie IDs")
    if ids != manifest["movie_ids"] or ids != sorted(ids):
        raise ValueError("dense cache ordered movie IDs mismatch")
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise ValueError("unsafe dense cache embedding array")
    if list(embeddings.shape) != manifest["embedding_shape"]:
        raise ValueError("dense cache embedding shape mismatch")
    if embeddings.shape != (len(ids), manifest["dimension"]):
        raise ValueError("dense cache dimension mismatch")
    _validate_normalized_matrix(embeddings)
    if _embedding_checksum(embeddings) != manifest["embedding_checksum"]:
        raise ValueError("dense cache embedding checksum mismatch")


def _validate_normalized_matrix(embeddings: np.ndarray) -> None:
    if not np.isfinite(embeddings).all():
        raise ValueError("dense cache embeddings must be finite")
    if embeddings.shape[0] and not np.allclose(
        np.linalg.norm(embeddings.astype(np.float64), axis=1),
        1.0,
        atol=1e-5,
        rtol=1e-6,
    ):
        raise ValueError("dense cache embeddings are not normalized")


def _read_dense_cache(
    cache_path: Path,
    *,
    movies: dict[int, Movie],
    model_name: str,
    model_revision: str | None,
    device: str | None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    manifest_path = _manifest_path(cache_path)
    with _cache_lock(cache_path):
        manifest_fd = _try_open_regular_file(
            manifest_path,
            max_bytes=MAX_MANIFEST_BYTES,
            too_large_message="dense cache manifest is too large",
        )
        cache_fd = _try_open_regular_file(
            cache_path,
            max_bytes=MAX_CACHE_ARCHIVE_BYTES,
            too_large_message="dense cache archive is too large",
        )
        if (manifest_fd is None) != (cache_fd is None):
            _close_optional_fd(manifest_fd)
            _close_optional_fd(cache_fd)
            raise ValueError("partial dense embedding cache")
        if manifest_fd is None or cache_fd is None:
            raise ValueError("dense embedding cache files are missing or unsafe")
        with (
            os.fdopen(manifest_fd, "rb") as manifest_stream,
            os.fdopen(cache_fd, "rb") as cache_stream,
        ):
            try:
                manifest = json.loads(manifest_stream.read(MAX_MANIFEST_BYTES + 1))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("unsafe dense embedding cache manifest") from exc
            _validate_manifest(manifest)
            if manifest["model_name"] != model_name:
                raise ValueError("dense cache model_name mismatch")
            _validate_revision_request(manifest, model_revision)
            if device is not None and manifest["device"] != device:
                raise ValueError("dense cache device mismatch")
            if manifest["runtime_metadata"] != _runtime_metadata():
                raise ValueError("dense cache runtime metadata mismatch")
            if manifest["library_versions"] != _library_versions():
                raise ValueError("dense cache library metadata mismatch")
            expected_dataset = movie_catalog_fingerprint(movies)
            if manifest["dataset_fingerprint"] != expected_dataset:
                raise ValueError("dense cache dataset_fingerprint mismatch")
            if manifest["movie_ids"] != sorted(movies):
                raise ValueError("dense cache movie IDs do not match the dataset catalog")
            _inspect_npz_archive(cache_stream, manifest)
            cache_stream.seek(0)
            try:
                with np.load(
                    cache_stream,
                    allow_pickle=False,
                    max_header_size=MAX_NPY_HEADER_BYTES,
                ) as payload:
                    if set(payload.files) != {"movie_ids", "embeddings"}:
                        raise ValueError("unsafe dense embedding cache contents")
                    movie_ids = payload["movie_ids"]
                    embeddings = payload["embeddings"]
            except (OSError, ValueError, TypeError) as exc:
                if isinstance(exc, ValueError) and "unsafe" in str(exc):
                    raise
                raise ValueError("unsafe dense embedding cache data") from exc
            _validate_cached_arrays(movie_ids, embeddings, manifest)
            return manifest, movie_ids, embeddings


def _close_optional_fd(descriptor: int | None) -> None:
    if descriptor is not None:
        os.close(descriptor)


def _try_open_regular_file(
    path: Path,
    *,
    max_bytes: int,
    too_large_message: str,
) -> int | None:
    try:
        return _open_regular_file(
            path,
            max_bytes=max_bytes,
            too_large_message=too_large_message,
        )
    except FileNotFoundError:
        return None


def _open_regular_file(
    path: Path,
    *,
    max_bytes: int,
    too_large_message: str,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("unsafe dense embedding cache file type")
        if file_stat.st_size > max_bytes:
            raise ValueError(too_large_message)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_revision_request(
    manifest: dict[str, object],
    requested: str | None,
) -> None:
    if requested is None:
        return
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", requested):
        if manifest["resolved_revision"] != requested:
            raise ValueError("dense cache model_revision mismatch")
    elif manifest["requested_revision"] != requested:
        raise ValueError("dense cache model_revision mismatch")


def _inspect_npz_archive(cache_stream: BinaryIO, manifest: dict[str, object]) -> None:
    try:
        cache_stream.seek(0)
        with zipfile.ZipFile(cache_stream) as archive:
            infos = archive.infolist()
            movie_shape, movie_fortran, movie_dtype = _read_npy_header(
                archive, "movie_ids.npy"
            )
            embedding_shape, embedding_fortran, embedding_dtype = _read_npy_header(
                archive, "embeddings.npy"
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("unsafe dense embedding cache archive") from exc
    if len(infos) != 2 or {info.filename for info in infos} != {
        "movie_ids.npy",
        "embeddings.npy",
    }:
        raise ValueError("unsafe dense embedding cache archive names")
    if any(info.flag_bits & 1 for info in infos):
        raise ValueError("unsafe encrypted dense embedding cache archive")
    if any(
        info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        for info in infos
    ):
        raise ValueError("unsafe dense embedding cache compression")
    total_uncompressed = sum(info.file_size for info in infos)
    if total_uncompressed > MAX_CACHE_UNCOMPRESSED_BYTES:
        raise ValueError("dense cache uncompressed data is too large")
    rows = len(manifest["movie_ids"])  # type: ignore[arg-type]
    dimension = int(manifest["dimension"])
    expected_array_bytes = rows * 8 + rows * dimension * 4
    info_by_name = {info.filename: info for info in infos}
    expected_member_bytes = {
        "movie_ids.npy": rows * 8,
        "embeddings.npy": rows * dimension * 4,
    }
    for name, expected_bytes in expected_member_bytes.items():
        file_size = info_by_name[name].file_size
        if not expected_bytes <= file_size <= expected_bytes + MAX_NPY_HEADER_BYTES:
            raise ValueError("dense cache uncompressed member size mismatch")
    if total_uncompressed > expected_array_bytes + 2 * MAX_NPY_HEADER_BYTES:
        raise ValueError("dense cache uncompressed array headers are too large")
    if (
        movie_shape != (rows,)
        or movie_fortran
        or movie_dtype != np.dtype(np.int64)
        or embedding_shape != (rows, dimension)
        or embedding_fortran
        or embedding_dtype != np.dtype(np.float32)
    ):
        raise ValueError("dense cache array header shape/dtype mismatch")


def _read_npy_header(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[tuple[int, ...], bool, np.dtype]:
    try:
        with archive.open(name) as stream:
            version_number = np.lib.format.read_magic(stream)
            if version_number == (1, 0):
                return np.lib.format.read_array_header_1_0(
                    stream, max_header_size=MAX_NPY_HEADER_BYTES
                )
            if version_number == (2, 0):
                return np.lib.format.read_array_header_2_0(
                    stream, max_header_size=MAX_NPY_HEADER_BYTES
                )
    except (EOFError, OSError, ValueError, KeyError) as exc:
        raise ValueError("unsafe dense cache array header") from exc
    raise ValueError("unsafe dense cache array header version")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    vector = {token: count * idf[token] for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        return {token: value / norm for token, value in vector.items()}
    return {}

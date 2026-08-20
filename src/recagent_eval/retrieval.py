from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_CACHE_SCHEMA_VERSION = 2
DENSE_ITEM_TEXT_SCHEMA = "v1-movie-text-title-genres"


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
        self.model_revision = resolved or revision
        if not self.model_revision:
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
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0 or not query.strip() or not self.movie_ids.size:
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
            "model_revision": self.model_revision,
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
        data_temp = cache_path.with_name(f".{cache_path.name}.tmp")
        manifest_temp = manifest_path.with_name(f".{manifest_path.name}.tmp")
        try:
            with data_temp.open("wb") as stream:
                np.savez(stream, movie_ids=self.movie_ids, embeddings=embeddings)
            manifest_temp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(data_temp, cache_path)
            os.replace(manifest_temp, manifest_path)
        finally:
            data_temp.unlink(missing_ok=True)
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
        revision_to_load = model_revision or str(manifest["model_revision"])
        active_device = device or str(manifest["device"])
        active_encoder = encoder or SentenceTransformerEncoder(
            model_name,
            revision=revision_to_load,
            device=active_device,
        )
        expected_revision = _encoder_revision(active_encoder, revision_to_load)
        if manifest["model_revision"] != expected_revision:
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
    ) -> dict[str, object]:
        """Validate a SentenceTransformer cache without loading model weights."""
        manifest, _, _ = _read_dense_cache(
            Path(path),
            movies=movies,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
        )
        expected_encoder = {"class": _encoder_identifier(SentenceTransformerEncoder)}
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
        "model_revision",
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
    try:
        generated_at = datetime.fromisoformat(str(manifest["generated_at"]))
    except ValueError as exc:
        raise ValueError("dense cache generated_at is invalid") from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() != UTC.utcoffset(None):
        raise ValueError("dense cache generated_at must be UTC")
    for field in ("model_name", "model_revision", "dataset_fingerprint"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"dense cache {field} is invalid")
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
    if cache_path.exists() != manifest_path.exists():
        raise ValueError("partial dense embedding cache")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise ValueError("dense embedding cache files are missing or unsafe")
    if cache_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("unsafe dense embedding cache path")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("unsafe dense embedding cache manifest") from exc
    _validate_manifest(manifest)
    if manifest["model_name"] != model_name:
        raise ValueError("dense cache model_name mismatch")
    if model_revision is not None and manifest["model_revision"] != model_revision:
        raise ValueError("dense cache model_revision mismatch")
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
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    vector = {token: count * idf[token] for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        return {token: value / norm for token, value in vector.items()}
    return {}

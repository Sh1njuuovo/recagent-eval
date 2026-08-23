from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from recagent_eval.data import Rating

LATENT_ARTIFACT_SCHEMA_VERSION = 1
MAX_LATENT_MANIFEST_BYTES = 1024 * 1024
MAX_LATENT_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_LATENT_ITEMS = 1_000_000
MAX_LATENT_RANK = 512
MAX_NPY_HEADER_BYTES = 4096


@dataclass(frozen=True)
class LatentFactorRetriever:
    item_factors: np.ndarray
    item_ids: np.ndarray
    rank: int
    iterations: int
    alpha: float
    lambda_reg: float
    seed: int
    positive_threshold: int
    training_fingerprint: str

    @classmethod
    def fit(
        cls,
        ratings: Iterable[Rating],
        *,
        rank: int = 20,
        iterations: int = 12,
        alpha: float = 40.0,
        lambda_reg: float = 0.1,
        positive_threshold: int = 4,
        seed: int = 42,
    ) -> LatentFactorRetriever:
        if rank <= 0 or rank > MAX_LATENT_RANK:
            raise ValueError("latent rank must be in (0, 512]")
        if iterations <= 0:
            raise ValueError("latent iterations must be positive")
        if alpha <= 0.0 or not math.isfinite(alpha):
            raise ValueError("latent alpha must be a positive finite number")
        if lambda_reg < 0.0 or not math.isfinite(lambda_reg):
            raise ValueError("latent lambda_reg must be a non-negative finite number")
        rows = tuple(ratings)
        with threadpool_limits(limits=1):
            item_factors, item_ids, training_fingerprint = _fit_als(
                rows,
                rank=rank,
                iterations=iterations,
                alpha=alpha,
                lambda_reg=lambda_reg,
                positive_threshold=positive_threshold,
                seed=seed,
            )
        return cls(
            item_factors=item_factors,
            item_ids=item_ids,
            rank=rank,
            iterations=iterations,
            alpha=alpha,
            lambda_reg=lambda_reg,
            seed=seed,
            positive_threshold=positive_threshold,
            training_fingerprint=training_fingerprint,
        )

    def retrieve(
        self,
        history: set[int],
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.item_ids.size or not history:
            return []
        known = set(self.item_ids.tolist()) & set(history)
        if not known:
            return []
        history_ids = np.fromiter(sorted(known), dtype=np.int64)
        mask = np.isin(self.item_ids, history_ids)
        history_factors = self.item_factors[mask]
        matrix = (
            self.item_factors.T @ self.item_factors
            + self.alpha * (history_factors.T @ history_factors)
            + self.lambda_reg * np.eye(self.rank)
        )
        vector = (1.0 + self.alpha) * history_factors.sum(axis=0)
        user_factor = np.linalg.solve(matrix, vector)
        scores = self.item_factors @ user_factor
        if not np.isfinite(scores).all():
            raise ValueError("latent scores must be finite")
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            allowed_mask = np.isin(
                self.item_ids, np.fromiter(sorted(allowed_ids), dtype=np.int64)
            )
            candidate_ids = self.item_ids[allowed_mask]
            candidate_scores = scores[allowed_mask]
        else:
            candidate_ids = self.item_ids
            candidate_scores = scores
        order = np.lexsort((candidate_ids, -candidate_scores))[:top_k]
        return [
            (int(candidate_ids[index]), float(candidate_scores[index]))
            for index in order
        ]

    def save(self, path: Path) -> None:
        artifact_path = Path(path)
        manifest_path = Path(f"{path}.json")
        if artifact_path.exists() or manifest_path.exists():
            raise ValueError("refusing to overwrite existing latent artifact")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        factors = np.ascontiguousarray(self.item_factors, dtype=np.float32)
        if factors.shape != (self.item_ids.size, self.rank):
            raise ValueError("latent item factors shape mismatch")
        if not np.isfinite(factors).all():
            raise ValueError("latent item factors must be finite")
        payload: dict[str, Any] = {
            "schema_version": LATENT_ARTIFACT_SCHEMA_VERSION,
            "rank": self.rank,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "seed": self.seed,
            "positive_threshold": self.positive_threshold,
            "item_ids": self.item_ids.tolist(),
            "shape": list(factors.shape),
            "dtype": str(factors.dtype),
            "training_fingerprint": self.training_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "runtime_metadata": _runtime_metadata(),
        }
        _validate_manifest(payload)
        data_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            data_fd, data_name = tempfile.mkstemp(
                prefix=f".{artifact_path.name}.", suffix=".tmp", dir=artifact_path.parent
            )
            data_temp = Path(data_name)
            with os.fdopen(data_fd, "wb") as stream:
                np.savez(stream, item_ids=self.item_ids, item_factors=factors)
                stream.flush()
                os.fsync(stream.fileno())
            if data_temp.stat().st_size > MAX_LATENT_ARTIFACT_BYTES:
                raise ValueError("latent artifact is too large")
            payload["artifact_checksum"] = hashlib.sha256(
                data_temp.read_bytes()
            ).hexdigest()
            payload["manifest_sha256"] = _manifest_digest(payload)
            manifest_fd, manifest_name = tempfile.mkstemp(
                prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
            )
            manifest_temp = Path(manifest_name)
            with os.fdopen(manifest_fd, "wb") as stream:
                stream.write(
                    (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(data_temp, artifact_path)
            data_temp = None
            os.replace(manifest_temp, manifest_path)
            manifest_temp = None
        finally:
            if data_temp is not None:
                with suppress(FileNotFoundError):
                    data_temp.unlink()
            if manifest_temp is not None:
                with suppress(FileNotFoundError):
                    manifest_temp.unlink()

    @classmethod
    def load(
        cls, path: Path, *, expected_training_fingerprint: str | None = None
    ) -> LatentFactorRetriever:
        artifact_path = Path(path)
        manifest_path = Path(f"{path}.json")
        try:
            manifest = json.loads(
                _read_regular_file(manifest_path, max_bytes=MAX_LATENT_MANIFEST_BYTES)
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid latent artifact manifest") from exc
        _validate_manifest(manifest)
        if not isinstance(manifest.get("artifact_checksum"), str) or not isinstance(
            manifest.get("manifest_sha256"), str
        ):
            raise ValueError("invalid latent artifact manifest checksums")
        if _manifest_digest(manifest) != manifest["manifest_sha256"]:
            raise ValueError("latent artifact manifest checksum mismatch")
        if (
            expected_training_fingerprint is not None
            and manifest["training_fingerprint"] != expected_training_fingerprint
        ):
            raise ValueError("latent artifact training fingerprint mismatch")
        try:
            data = _read_regular_file(artifact_path, max_bytes=MAX_LATENT_ARTIFACT_BYTES)
        except (OSError, ValueError) as exc:
            raise ValueError("invalid latent artifact data") from exc
        if hashlib.sha256(data).hexdigest() != manifest["artifact_checksum"]:
            raise ValueError("latent artifact checksum mismatch")
        try:
            with np.load(
                io.BytesIO(data),
                allow_pickle=False,
                max_header_size=MAX_NPY_HEADER_BYTES,
            ) as payload:
                if set(payload.files) != {"item_ids", "item_factors"}:
                    raise ValueError("unsafe latent artifact contents")
                item_ids = np.ascontiguousarray(payload["item_ids"], dtype=np.int64)
                item_factors = np.ascontiguousarray(
                    payload["item_factors"], dtype=np.float32
                )
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("unsafe latent artifact data") from exc
        if item_ids.ndim != 1 or item_factors.ndim != 2:
            raise ValueError("unsafe latent artifact array shapes")
        if item_ids.tolist() != manifest["item_ids"] or item_ids.tolist() != sorted(
            item_ids.tolist()
        ):
            raise ValueError("latent artifact item IDs mismatch")
        if list(item_factors.shape) != manifest["shape"]:
            raise ValueError("latent artifact shape mismatch")
        if not np.isfinite(item_factors).all():
            raise ValueError("latent item factors must be finite")
        return cls(
            item_factors=item_factors,
            item_ids=item_ids,
            rank=manifest["rank"],
            iterations=manifest["iterations"],
            alpha=manifest["alpha"],
            lambda_reg=manifest["lambda_reg"],
            seed=manifest["seed"],
            positive_threshold=manifest["positive_threshold"],
            training_fingerprint=manifest["training_fingerprint"],
        )


def _fit_als(
    rows: tuple[Rating, ...],
    *,
    rank: int,
    iterations: int,
    alpha: float,
    lambda_reg: float,
    positive_threshold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    user_items: dict[int, list[Rating]] = defaultdict(list)
    item_raters: dict[int, list[Rating]] = defaultdict(list)
    for row in rows:
        if row.rating >= positive_threshold:
            user_items[row.user_id].append(row)
            item_raters[row.movie_id].append(row)
    users = sorted(user_items)
    item_ids = np.asarray(sorted(item_raters), dtype=np.int64)
    if not users or not item_ids.size:
        raise ValueError("latent fit requires at least one positive user and item")
    user_index = {user_id: index for index, user_id in enumerate(users)}
    item_index = {int(movie_id): index for index, movie_id in enumerate(item_ids)}
    user_index_to_items: list[np.ndarray] = []
    for user_id in users:
        user_index_to_items.append(
            np.asarray(
                [item_index[row.movie_id] for row in user_items[user_id]],
                dtype=np.int64,
            )
        )
    item_index_to_users: list[np.ndarray] = []
    for movie_id in item_ids:
        item_index_to_users.append(
            np.asarray(
                [user_index[row.user_id] for row in item_raters[int(movie_id)]],
                dtype=np.int64,
            )
        )
    random = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(rank)
    user_factors = random.normal(0.0, scale, size=(len(users), rank))
    item_factors = random.normal(0.0, scale, size=(len(item_ids), rank))
    eye = lambda_reg * np.eye(rank)
    for _ in range(iterations):
        item_gram = item_factors.T @ item_factors
        for user_position, items in enumerate(user_index_to_items):
            y = item_factors[items]
            matrix = item_gram + alpha * (y.T @ y) + eye
            vector = (1.0 + alpha) * y.sum(axis=0)
            user_factors[user_position] = np.linalg.solve(matrix, vector)
        user_gram = user_factors.T @ user_factors
        for item_position, raters in enumerate(item_index_to_users):
            x = user_factors[raters]
            matrix = user_gram + alpha * (x.T @ x) + eye
            vector = (1.0 + alpha) * x.sum(axis=0)
            item_factors[item_position] = np.linalg.solve(matrix, vector)
    factors = np.ascontiguousarray(item_factors, dtype=np.float32)
    training_fingerprint = _training_fingerprint(
        rows, rank, iterations, alpha, lambda_reg, positive_threshold, seed
    )
    return factors, item_ids, training_fingerprint


def _training_fingerprint(
    rows: tuple[Rating, ...],
    rank: int,
    iterations: int,
    alpha: float,
    lambda_reg: float,
    positive_threshold: int,
    seed: int,
) -> str:
    canonical = json.dumps(
        {
            "rank": rank,
            "iterations": iterations,
            "alpha": alpha,
            "lambda_reg": lambda_reg,
            "positive_threshold": positive_threshold,
            "seed": seed,
            "rows": [
                [row.user_id, row.movie_id, row.rating, row.timestamp]
                for row in sorted(
                    rows,
                    key=lambda row: (row.user_id, row.timestamp, row.movie_id, row.rating),
                )
            ],
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "rank",
        "iterations",
        "alpha",
        "lambda_reg",
        "seed",
        "positive_threshold",
        "item_ids",
        "shape",
        "dtype",
        "training_fingerprint",
        "created_at",
        "runtime_metadata",
    }
    if not required.issubset(manifest) or not set(manifest) <= required | {
        "artifact_checksum",
        "manifest_sha256",
    }:
        raise ValueError("invalid latent artifact manifest fields")
    if manifest["schema_version"] != LATENT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("latent artifact schema_version mismatch")
    if not isinstance(manifest["rank"], int) or not 0 < manifest["rank"] <= MAX_LATENT_RANK:
        raise ValueError("latent artifact rank is invalid")
    if not isinstance(manifest["item_ids"], list) or not all(
        isinstance(item, int) for item in manifest["item_ids"]
    ):
        raise ValueError("latent artifact item_ids are invalid")
    if len(manifest["item_ids"]) > MAX_LATENT_ITEMS:
        raise ValueError("latent artifact item count is too large")
    if manifest["dtype"] != "float32":
        raise ValueError("latent artifact dtype mismatch")
    if manifest["shape"] != [len(manifest["item_ids"]), manifest["rank"]]:
        raise ValueError("latent artifact shape mismatch")
    if not isinstance(manifest["training_fingerprint"], str) or not re_fullmatch_64(
        manifest["training_fingerprint"]
    ):
        raise ValueError("latent artifact training fingerprint is invalid")
    if not isinstance(manifest["runtime_metadata"], dict):
        raise ValueError("latent artifact runtime metadata is invalid")


def re_fullmatch_64(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    signed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _runtime_metadata() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("unsafe latent artifact file type")
        if file_stat.st_size > max_bytes:
            raise ValueError("latent artifact member is too large")
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read(max_bytes + 1)
    except BaseException:
        os.close(descriptor)
        raise

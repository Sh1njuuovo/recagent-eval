from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import stat
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recagent_eval.baseline_eval import MetricRow, register_baseline, score_ranking
from recagent_eval.data import LeakageSafeRankingSplit, Movie, Rating
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    _state_from_history,
    ranking_dataset_fingerprint,
)
from recagent_eval.resource_usage import read_process_peak_rss
from recagent_eval.retrieval import hard_filter

BPR_SCHEMA_VERSION = 1
MAX_BPR_MEMBER_BYTES = 512 * 1024 * 1024

BPR_PARAMETER_GRID: tuple[Mapping[str, float], ...] = tuple(
    {
        "rank": rank,
        "learning_rate": learning_rate,
        "reg": reg,
        "epochs": epochs,
    }
    for rank in (16, 32)
    for learning_rate in (1e-3, 5e-3)
    for reg in (1e-4, 1e-3)
    for epochs in (10, 20)
)


@dataclass(frozen=True)
class BPRMatrixFactorization:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    rank: int
    learning_rate: float
    reg: float
    epochs: int
    seed: int
    training_fingerprint: str
    _user_index: dict[int, int] = None  # type: ignore[assignment]
    _item_index: dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._user_index is None:
            object.__setattr__(
                self,
                "_user_index",
                {int(user_id): index for index, user_id in enumerate(self.user_ids)},
            )
        if self._item_index is None:
            object.__setattr__(
                self,
                "_item_index",
                {int(movie_id): index for index, movie_id in enumerate(self.item_ids)},
            )

    @classmethod
    def fit(
        cls,
        ratings: Iterable[Rating],
        *,
        rank: int = 32,
        learning_rate: float = 1e-3,
        reg: float = 1e-3,
        epochs: int = 20,
        positive_threshold: int = 4,
        seed: int = 42,
    ) -> BPRMatrixFactorization:
        if rank <= 0:
            raise ValueError("bpr rank must be positive")
        if epochs <= 0:
            raise ValueError("bpr epochs must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("bpr learning_rate must be a positive finite number")
        if reg < 0.0 or not math.isfinite(reg):
            raise ValueError("bpr reg must be a non-negative finite number")
        rows = tuple(ratings)
        user_items: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if row.rating >= positive_threshold:
                user_items[row.user_id].append(row.movie_id)
        users = sorted(user_items)
        item_ids = np.asarray(
            sorted({movie_id for items in user_items.values() for movie_id in items}),
            dtype=np.int64,
        )
        if not users or not item_ids.size:
            raise ValueError("bpr fit requires at least one positive user and item")
        item_index = {int(movie_id): index for index, movie_id in enumerate(item_ids)}
        user_positives = [
            np.asarray([item_index[movie_id] for movie_id in user_items[user_id]], dtype=np.int64)
            for user_id in users
        ]
        user_factors, item_factors = _fit_bpr_torch(
            user_positives,
            n_items=len(item_ids),
            rank=rank,
            learning_rate=learning_rate,
            reg=reg,
            epochs=epochs,
            seed=seed,
        )
        training_fingerprint = _training_fingerprint(
            rows, rank, learning_rate, reg, epochs, positive_threshold, seed
        )
        return cls(
            user_ids=np.asarray(users, dtype=np.int64),
            item_ids=item_ids,
            user_factors=user_factors,
            item_factors=item_factors,
            rank=rank,
            learning_rate=learning_rate,
            reg=reg,
            epochs=epochs,
            seed=seed,
            training_fingerprint=training_fingerprint,
        )

    def score_user(self, user_id: int, movie_ids: Iterable[int]) -> dict[int, float]:
        user_position = self._user_index.get(int(user_id))
        if user_position is None:
            return {}
        wanted = [int(movie_id) for movie_id in movie_ids]
        if not wanted:
            return {}
        positions: list[int] = []
        present: list[int] = []
        for movie_id in wanted:
            if movie_id in self._item_index:
                positions.append(self._item_index[movie_id])
                present.append(movie_id)
        if not positions:
            return {}
        scores = (
            self.item_factors[np.asarray(positions, dtype=np.int64)]
            @ self.user_factors[user_position]
        )
        if not np.isfinite(scores).all():
            raise ValueError("bpr scores must be finite")
        result = {
            movie_id: float(score) for movie_id, score in zip(present, scores, strict=True)
        }
        for movie_id in wanted:
            if movie_id not in self._item_index:
                result[movie_id] = 0.0
        return result

    def save(self, path: Path) -> None:
        artifact_path = Path(path)
        manifest_path = Path(f"{path}.json")
        if artifact_path.exists() or manifest_path.exists():
            raise ValueError("refusing to overwrite existing bpr artifact")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        data_bytes, manifest_bytes = self.to_artifact_bytes()
        data_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            data_fd, data_name = tempfile.mkstemp(
                prefix=f".{artifact_path.name}.", suffix=".tmp", dir=artifact_path.parent
            )
            data_temp = Path(data_name)
            with os.fdopen(data_fd, "wb") as stream:
                stream.write(data_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            manifest_fd, manifest_name = tempfile.mkstemp(
                prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
            )
            manifest_temp = Path(manifest_name)
            with os.fdopen(manifest_fd, "wb") as stream:
                stream.write(manifest_bytes)
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

    def to_artifact_bytes(self) -> tuple[bytes, bytes]:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            user_factors=np.ascontiguousarray(self.user_factors, dtype=np.float32),
            item_factors=np.ascontiguousarray(self.item_factors, dtype=np.float32),
        )
        data_bytes = buffer.getvalue()
        if len(data_bytes) > MAX_BPR_MEMBER_BYTES:
            raise ValueError("bpr artifact is too large")
        payload: dict[str, Any] = {
            "schema_version": BPR_SCHEMA_VERSION,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "reg": self.reg,
            "epochs": self.epochs,
            "seed": self.seed,
            "user_ids": self.user_ids.tolist(),
            "item_ids": self.item_ids.tolist(),
            "training_fingerprint": self.training_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "artifact_checksum": hashlib.sha256(data_bytes).hexdigest(),
            "manifest_sha256": None,
        }
        payload["manifest_sha256"] = _manifest_digest(payload)
        manifest_bytes = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        return data_bytes, manifest_bytes

    @classmethod
    def load(cls, path: Path) -> BPRMatrixFactorization:
        artifact_path = Path(path)
        manifest_path = Path(f"{path}.json")
        try:
            manifest = json.loads(
                _read_regular_file(manifest_path, max_bytes=MAX_BPR_MEMBER_BYTES)
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid bpr artifact manifest") from exc
        if manifest.get("schema_version") != BPR_SCHEMA_VERSION:
            raise ValueError("bpr artifact schema_version mismatch")
        if _manifest_digest(manifest) != manifest.get("manifest_sha256"):
            raise ValueError("bpr artifact manifest checksum mismatch")
        try:
            data = _read_regular_file(artifact_path, max_bytes=MAX_BPR_MEMBER_BYTES)
        except (OSError, ValueError) as exc:
            raise ValueError("invalid bpr artifact data") from exc
        if hashlib.sha256(data).hexdigest() != manifest["artifact_checksum"]:
            raise ValueError("bpr artifact checksum mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=4096) as payload:
            if set(payload.files) != {
                "user_ids",
                "item_ids",
                "user_factors",
                "item_factors",
            }:
                raise ValueError("unsafe bpr artifact contents")
            user_ids = np.ascontiguousarray(payload["user_ids"], dtype=np.int64)
            item_ids = np.ascontiguousarray(payload["item_ids"], dtype=np.int64)
            user_factors = np.ascontiguousarray(payload["user_factors"], dtype=np.float32)
            item_factors = np.ascontiguousarray(payload["item_factors"], dtype=np.float32)
        if user_ids.tolist() != manifest["user_ids"] or item_ids.tolist() != manifest["item_ids"]:
            raise ValueError("bpr artifact id mismatch")
        return cls(
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=user_factors,
            item_factors=item_factors,
            rank=manifest["rank"],
            learning_rate=manifest["learning_rate"],
            reg=manifest["reg"],
            epochs=manifest["epochs"],
            seed=manifest["seed"],
            training_fingerprint=manifest["training_fingerprint"],
        )


def _fit_bpr_torch(
    user_positives: Sequence[np.ndarray],
    *,
    n_items: int,
    rank: int,
    learning_rate: float,
    reg: float,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_users = len(user_positives)
    user_factors = torch.nn.Parameter(torch.randn(n_users, rank) * 0.1)
    item_factors = torch.nn.Parameter(torch.randn(n_items, rank) * 0.1)
    optimizer = torch.optim.SGD([user_factors, item_factors], lr=learning_rate)
    positives_set = [set(items.tolist()) for items in user_positives]
    batch_size = 256
    steps_per_epoch = max(64, (n_users * 50) // batch_size)
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            batch = rng.integers(0, n_users, size=batch_size)
            positive = np.asarray(
                [int(rng.choice(user_positives[user])) for user in batch],
                dtype=np.int64,
            )
            negative = np.asarray(
                [
                    _sample_negative(positives_set[user], n_items, rng)
                    for user in batch
                ],
                dtype=np.int64,
            )
            user_batch = user_factors[torch.from_numpy(batch)]
            positive_batch = item_factors[torch.from_numpy(positive)]
            negative_batch = item_factors[torch.from_numpy(negative)]
            positive_score = (user_batch * positive_batch).sum(dim=1)
            negative_score = (user_batch * negative_batch).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(
                positive_score - negative_score
            ).sum()
            loss = loss + reg * (
                user_batch.norm() + positive_batch.norm() + negative_batch.norm()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return (
        np.ascontiguousarray(user_factors.detach().numpy(), dtype=np.float32),
        np.ascontiguousarray(item_factors.detach().numpy(), dtype=np.float32),
    )


def _sample_negative(positives: set[int], n_items: int, rng: np.random.Generator) -> int:
    for _ in range(5):
        candidate = int(rng.integers(0, n_items))
        if candidate not in positives:
            return candidate
    return int(rng.integers(0, n_items))


def _training_fingerprint(
    rows: tuple[Rating, ...],
    rank: int,
    learning_rate: float,
    reg: float,
    epochs: int,
    positive_threshold: int,
    seed: int,
) -> str:
    canonical = json.dumps(
        {
            "rank": rank,
            "learning_rate": learning_rate,
            "reg": reg,
            "epochs": epochs,
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


def _manifest_digest(manifest: dict[str, Any]) -> str:
    signed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("unsafe bpr artifact file type")
        if file_stat.st_size > max_bytes:
            raise ValueError("bpr artifact member is too large")
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read(max_bytes + 1)
    except BaseException:
        os.close(descriptor)
        raise


def select_bpr_params(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    dev_users: Sequence[int],
) -> dict[str, object]:
    dev_set = set(dev_users)
    dev_rows = tuple(
        row for row in split.legal_retrieval_train if row.user_id in dev_set
    )
    histories = _positive_histories(split.legal_retrieval_train, movies)
    results: list[tuple[float, Mapping[str, float]]] = []
    for params in BPR_PARAMETER_GRID:
        model = BPRMatrixFactorization.fit(
            dev_rows,
            rank=int(params["rank"]),
            learning_rate=float(params["learning_rate"]),
            reg=float(params["reg"]),
            epochs=int(params["epochs"]),
            seed=42,
        )
        ndcgs: list[float] = []
        for user_id in dev_users:
            history_ids = {row.movie_id for row in histories.get(user_id, ())}
            state = _state_from_history(history_ids, movies)
            allowed = {
                movie.movie_id for movie in hard_filter(movies.values(), state)
            } - history_ids
            target = split.validation_targets[user_id]
            if not history_ids:
                ndcgs.append(0.0)
                continue
            scores = model.score_user(user_id, allowed)
            ranked = sorted(
                allowed,
                key=lambda movie_id: (-scores.get(movie_id, 0.0), movie_id),
            )[:10]
            ndcgs.append(
                1.0 / math.log2(ranked.index(target) + 2) if target in ranked else 0.0
            )
        mean_ndcg = sum(ndcgs) / len(dev_users) if dev_users else 0.0
        results.append((mean_ndcg, params))
    best_mean, best_params = max(
        results,
        key=lambda item: (
            item[0],
            -int(item[1]["rank"]),
            -int(item[1]["epochs"]),
            -float(item[1]["learning_rate"]),
            float(item[1]["reg"]),
        ),
    )
    payload = {
        "selected_params": dict(best_params),
        "mean_ndcg_at_10": best_mean,
        "dev_user_fingerprint": _fingerprint(sorted(dev_users)),
        "seed": 42,
    }
    return {
        **payload,
        "fingerprint": _fingerprint(payload),
        "grid": [dict(params) for params in BPR_PARAMETER_GRID],
    }


@register_baseline("bpr_mf")
def score_bpr_mf(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    users: Sequence[int],
    *,
    ledger: Mapping[str, object] | None = None,
    max_training_users: int | None = None,
    selected_params: Mapping[str, float] | None = None,
    selection_fingerprint: str | None = None,
    seed: int = 42,
) -> dict[str, object]:
    if selected_params is None:
        dev_users = (
            [int(user) for user in ledger["cohorts"]["development"]]
            if ledger is not None
            else sorted(split.validation_targets)[:10]
        )
        selection = select_bpr_params(movies, split, dev_users)
        params = selection["selected_params"]
        config_fingerprint = str(selection["fingerprint"])
    else:
        if not selection_fingerprint:
            raise ValueError("locked BPR parameters require selection_fingerprint")
        params = dict(selected_params)
        config_fingerprint = selection_fingerprint
    train_rows = split.legal_retrieval_train
    if max_training_users is not None:
        train_users = set(sorted({row.user_id for row in train_rows})[:max_training_users])
        train_rows = tuple(row for row in train_rows if row.user_id in train_users)
    started = time.perf_counter()
    model = BPRMatrixFactorization.fit(
        train_rows,
        rank=int(params["rank"]),
        learning_rate=float(params["learning_rate"]),
        reg=float(params["reg"]),
        epochs=int(params["epochs"]),
        seed=seed,
    )
    training_seconds = time.perf_counter() - started
    histories = _positive_histories(split.legal_retrieval_train, movies)
    rows: list[MetricRow] = []
    for user_id in users:
        history_ids = {row.movie_id for row in histories.get(user_id, ())}
        state = _state_from_history(history_ids, movies)
        allowed = {
            movie.movie_id for movie in hard_filter(movies.values(), state)
        } - history_ids
        target = split.validation_targets[user_id]
        t0 = time.perf_counter()
        if not history_ids:
            ranked: list[int] = []
        else:
            scores = model.score_user(user_id, allowed)
            ranked = sorted(
                allowed,
                key=lambda movie_id: (-scores.get(movie_id, 0.0), movie_id),
            )[:10]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            score_ranking(
                user_id=user_id,
                ranked_ids=ranked,
                target=target,
                allowed=allowed,
                history=history_ids,
                candidate_recall=1.0 if target in allowed else 0.0,
                latency_ms=latency_ms,
            )
        )
    return {
        "rows": rows,
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": ranking_dataset_fingerprint(movies, split),
        "model_fingerprint": model.training_fingerprint,
        "selected_params": dict(params),
        "parameter_grid": [dict(grid_params) for grid_params in BPR_PARAMETER_GRID],
        "seed": seed,
        "training_seconds": training_seconds,
        "resource_usage": read_process_peak_rss(),
        "model_size_bytes": int(
            model.user_factors.nbytes + model.item_factors.nbytes
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

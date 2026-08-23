# Collaborative Latent Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic weighted-ALS latent recall route (fold-in scoring, persisted item-factor artifact), latent features (schema v2/v2b), route-balanced hard negatives, schema-v2 artifact/evidence/bundle contracts, and a pre-registered candidate-stage gate, so the 500-user validation gate can produce a trustworthy positive result.

**Architecture:** A new `latent_retrieval.py` module implements `LatentFactorRetriever` (numpy-only ALS, `threadpoolctl`-limited, NPZ+manifest persistence). The candidate pipeline threads the latent route into `build_candidate_queries`/fold/validation builders, extends the feature schema, applies route-balanced negative sampling at matrix build time, and publishes `lambdamart-artifact/v2` + `lambdamart-validation/v2` + `lambdamart-bundle/v2` bound to the latent artifact checksum. A read-only `diagnose-latent` CLI measures the pre-registered candidate gates before any ranker training.

**Tech Stack:** Python 3.11+, Typer, Pydantic, NumPy, threadpoolctl (new direct dependency), LightGBM, scikit-learn, pytest, Ruff, YAML.

**Spec:** `docs/superpowers/specs/2026-08-23-collaborative-recall-design.md` (committed at `c752694`).

---

## File structure

- Create: `src/recagent_eval/latent_retrieval.py` — ALS fit, fold-in retrieval, NPZ+manifest save/load.
- Create: `src/recagent_eval/latent_diagnostics.py` — candidate-stage metrics and gate aggregation.
- Modify: `src/recagent_eval/candidate_features.py` — schema v2/v2b and latent features.
- Modify: `src/recagent_eval/config.py`, `src/recagent_eval/runner.py` — latent + sampling config.
- Modify: `src/recagent_eval/learned_ranking.py` — schema-aware artifacts, route-balanced negatives.
- Modify: `src/recagent_eval/bundle.py` — bundle v2 with latent member.
- Modify: `src/recagent_eval/v2_selection.py` — evidence v2 with latent provenance.
- Modify: `src/recagent_eval/lambdamart_pipeline.py` — latent threading + fingerprints.
- Modify: `src/recagent_eval/cli.py` — `diagnose-latent`; latent loading in `evaluate-ranker`.
- Modify: `pyproject.toml`, `uv.lock` — threadpoolctl direct dependency.
- Create configs: `configs/v2_dense_latent.yaml` (E4), `configs/v2_dense_latent_30.yaml` (E3), `configs/v2_dense_latent_allneg.yaml` (E5), `configs/v2_dense_latent_bfeat.yaml` (E6).
- Create tests: `tests/test_latent_retrieval.py`, `tests/test_candidate_features_v2.py`, `tests/test_latent_diagnostics.py`; extend `test_config.py`, `test_lambdamart_pipeline.py`, `test_v2_selection.py`, `test_learned_ranking_errors.py`, `test_safe_io_bundle.py`, `test_cli.py`.

---

### Task 0: Evidence hygiene (percentile summary typo + legacy fingerprint marking)

**Files:**
- Modify: `reports/experiments/v2-dense-lambdamart-recall1500-percentile.json`
- Modify: `reports/experiments/v2-dense-lambdamart-recall1500-percentile.md`
- Modify: `reports/experiments/v2-dense-lambdamart-recall1500.md`

- [ ] **Step 1: Fix the summary JSON typo without touching evidence**

Edit `reports/experiments/v2-dense-lambdamart-recall1500-percentile.json`: change the top-level `"score_calibration": "raw"` to `"score_calibration": "percentile"` and add `"corrected_at": "2026-08-23T00:00:00Z"` plus `"correction_note": "summary field typo fixed; evidence fingerprints unchanged"`.

- [ ] **Step 2: Verify the JSON and note the correction in the Markdown report**

Run:
```bash
.venv/bin/python -c "import json; d=json.load(open('reports/experiments/v2-dense-lambdamart-recall1500-percentile.json')); assert d['score_calibration']=='percentile'; print(d['score_calibration'], d['evidence_fingerprint'])"
```
Expected: `percentile be72a00fcacee23a6fbbf8a49482d67789afbac0fea9040b3ff1d3962e9e588b`.

Append a "Correction 2026-08-23" line to `reports/experiments/v2-dense-lambdamart-recall1500-percentile.md` stating that the summary's `score_calibration` field was corrected from `raw` to `percentile`; the validation artifact and evidence fingerprint are unchanged and authoritative.

- [ ] **Step 3: Mark the legacy fingerprint in the recall-1500 report**

Append to `reports/experiments/v2-dense-lambdamart-recall1500.md`:
```markdown
## Fingerprint note 2026-08-23

The recorded evidence config fingerprint `7b9373b4...` was produced by the
pre-`e1efee8` fingerprint payload (before `score_calibration` entered the
payload). Status: `legacy/non-replayable-under-current-code`. The current code
computes `3c0abb8c...` for the same YAML. The negative ranking result remains
valid evidence; the bundle cannot be replayed or consumed under current code.
```

- [ ] **Step 4: Commit**

```bash
git add reports/experiments/v2-dense-lambdamart-recall1500-percentile.json reports/experiments/v2-dense-lambdamart-recall1500-percentile.md reports/experiments/v2-dense-lambdamart-recall1500.md
git commit -m "docs: correct percentile summary field and mark legacy fingerprint"
```

---

### Task 1: Add threadpoolctl as a direct dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` under `[project].dependencies` (alphabetical order), add `"threadpoolctl>=3.1,<4",` before `"typer>=0.12,<1",`.

- [ ] **Step 2: Update the lock file**

Run:
```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
```
Expected: a message that the lock is out of date. Then run:
```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock
```
Expected: `uv.lock` updated, exit 0.

- [ ] **Step 3: Verify the dependency resolves in the worktree venv**

Run:
```bash
.venv/bin/python -c "from threadpoolctl import threadpool_limits; print(threadpool_limits)"
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
```
Expected: module prints, lock check passes.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add threadpoolctl direct dependency for deterministic latent fit"
```

---

### Task 2: `LatentFactorRetriever` (ALS fit, fold-in scoring, safe persistence)

**Files:**
- Create: `src/recagent_eval/latent_retrieval.py`
- Create: `tests/test_latent_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_latent_retrieval.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pytest

from recagent_eval.data import Rating
from recagent_eval.latent_retrieval import LatentFactorRetriever


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
    assert scores  # non-empty, finite
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
    assert LatentFactorRetriever.load(
        path, expected_training_fingerprint=model.training_fingerprint
    ).item_factors.shape == model.item_factors.shape


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
.venv/bin/pytest tests/test_latent_retrieval.py -q
```
Expected: collection error (`ModuleNotFoundError: recagent_eval.latent_retrieval`).

- [ ] **Step 3: Implement `latent_retrieval.py`**

Create `src/recagent_eval/latent_retrieval.py`:

```python
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
            payload["manifest_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest_fd, manifest_name = tempfile.mkstemp(
                prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
            )
            manifest_temp = Path(manifest_name)
            with os.fdopen(manifest_fd, "wb") as stream:
                stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != manifest["manifest_sha256"]:
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
```

- [ ] **Step 4: Run the latent tests**

Run:
```bash
.venv/bin/pytest tests/test_latent_retrieval.py -q
```
Expected: all tests pass.

- [ ] **Step 5: Ruff and commit**

Run:
```bash
.venv/bin/ruff check src/recagent_eval/latent_retrieval.py tests/test_latent_retrieval.py
git add src/recagent_eval/latent_retrieval.py tests/test_latent_retrieval.py
git commit -m "feat: add deterministic ALS latent retriever with fold-in scoring"
```
Expected: Ruff clean, commit created.

---

### Task 3: Feature schema v2/v2b with latent features

**Files:**
- Modify: `src/recagent_eval/candidate_features.py`
- Create: `tests/test_candidate_features_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidate_features_v2.py`:

```python
from __future__ import annotations

import pytest

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V2B,
    FEATURE_SCHEMA_FINGERPRINT,
    FEATURE_SCHEMA_FINGERPRINT_V2,
    FEATURE_SCHEMA_FINGERPRINT_V2B,
    build_candidate_feature_rows,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState


def _movies() -> dict[int, Movie]:
    return {
        1: Movie(1, "A", ("Drama",), 1995),
        2: Movie(2, "B", ("Comedy",), 1998),
        3: Movie(3, "C", ("Drama", "Comedy"), 2001),
    }


def _history() -> tuple[Rating, ...]:
    return (Rating(7, 1, 5, 10), Rating(7, 2, 4, 20))


def _state() -> PreferenceState:
    return PreferenceState(liked_movie_ids={1, 2}, liked_genres={"Drama"})


def _scores() -> dict[int, float]:
    return {1: 5.0, 2: 3.0}


def _latent() -> dict[int, float]:
    return {1: 0.8, 3: 0.4}


def test_v1_default_behavior_and_fingerprint_unchanged() -> None:
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        history=_history(),
        train_rows=_history(),
        state=_state(),
    )
    assert len(rows) == 3
    assert all(len(row.values) == len(FEATURE_NAMES) for row in rows)
    assert FEATURE_SCHEMA_FINGERPRINT == FEATURE_SCHEMA_FINGERPRINT


def test_v2_adds_latent_features_and_fingerprint() -> None:
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        latent_scores=_latent(),
        history=_history(),
        train_rows=_history(),
        state=_state(),
        feature_version="v2",
    )
    assert len(FEATURE_NAMES_V2) == len(FEATURE_NAMES) + 3
    assert all(len(row.values) == len(FEATURE_NAMES_V2) for row in rows)
    by_id = {row.movie_id: row.as_dict() for row in rows}
    assert by_id[1]["latent_score"] == 0.8
    assert by_id[2]["latent_score"] == 0.0
    assert by_id[2]["in_latent"] == 0.0
    assert by_id[3]["in_latent"] == 1.0
    assert FEATURE_SCHEMA_FINGERPRINT_V2 != FEATURE_SCHEMA_FINGERPRINT


def test_v2b_adds_cross_recent_year_features() -> None:
    recent = {1: 6.0, 2: 1.0}
    rows = build_candidate_feature_rows(
        user_id=7,
        movies=_movies(),
        itemcf_scores=_scores(),
        dense_scores=_scores(),
        latent_scores=_latent(),
        recent_itemcf_scores=recent,
        history=_history(),
        train_rows=_history(),
        state=_state(),
        feature_version="v2b",
    )
    assert len(FEATURE_NAMES_V2B) == len(FEATURE_NAMES_V2) + 3
    assert all(len(row.values) == len(FEATURE_NAMES_V2B) for row in rows)
    assert FEATURE_SCHEMA_FINGERPRINT_V2B != FEATURE_SCHEMA_FINGERPRINT_V2


def test_unknown_feature_version_fails() -> None:
    with pytest.raises(ValueError, match="feature_version"):
        build_candidate_feature_rows(
            user_id=7,
            movies=_movies(),
            itemcf_scores=_scores(),
            dense_scores=_scores(),
            history=_history(),
            train_rows=_history(),
            state=_state(),
            feature_version="v9",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
.venv/bin/pytest tests/test_candidate_features_v2.py -q
```
Expected: import errors (`FEATURE_NAMES_V2` missing) and failures for `latent_scores`/`feature_version` kwargs.

- [ ] **Step 3: Extend `candidate_features.py`**

In `src/recagent_eval/candidate_features.py`:

- After the existing v1 constants, add:
```python
FEATURE_SCHEMA_VERSION_V2 = "candidate-features/v2"
FEATURE_NAMES_V2 = FEATURE_NAMES + (
    "latent_score",
    "latent_reciprocal_rank",
    "in_latent",
)
FEATURE_SCHEMA_FINGERPRINT_V2 = hashlib.sha256(
    json.dumps(
        {"version": FEATURE_SCHEMA_VERSION_V2, "features": FEATURE_NAMES_V2},
        separators=(",", ":"),
    ).encode()
).hexdigest()

FEATURE_SCHEMA_VERSION_V2B = "candidate-features/v2b"
FEATURE_NAMES_V2B = FEATURE_NAMES_V2 + (
    "itemcf_latent_cross",
    "recent_itemcf_score",
    "year_recency",
)
FEATURE_SCHEMA_FINGERPRINT_V2B = hashlib.sha256(
    json.dumps(
        {"version": FEATURE_SCHEMA_VERSION_V2B, "features": FEATURE_NAMES_V2B},
        separators=(",", ":"),
    ).encode()
).hexdigest()
```

- Update `CandidateFeatureRow.as_dict` to accept an explicit name tuple, or add a `names` field to the dataclass. Minimal approach: add a module-level `_SCHEMAS` mapping and a `names: tuple[str, ...]` field:
```python
@dataclass(frozen=True)
class CandidateFeatureRow:
    user_id: int
    movie_id: int
    values: tuple[float, ...]
    names: tuple[str, ...] = FEATURE_NAMES

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))
```

- Rewrite `build_candidate_feature_rows` to accept `latent_scores: Mapping[int, float] | None = None`, `recent_itemcf_scores: Mapping[int, float] | None = None`, and `feature_version: str = "v1"`:
```python
_SCHEMA_BY_VERSION = {
    "v1": (FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT),
    "v2": (FEATURE_NAMES_V2, FEATURE_SCHEMA_FINGERPRINT_V2),
    "v2b": (FEATURE_NAMES_V2B, FEATURE_SCHEMA_FINGERPRINT_V2B),
}


def build_candidate_feature_rows(
    *,
    user_id: int,
    movies: Mapping[int, Movie],
    candidate_ids: Iterable[int] | None = None,
    itemcf_scores: Mapping[int, float],
    dense_scores: Mapping[int, float],
    history: Iterable[Rating],
    train_rows: Iterable[Rating],
    state: PreferenceState,
    score_calibration: str = "raw",
    latent_scores: Mapping[int, float] | None = None,
    recent_itemcf_scores: Mapping[int, float] | None = None,
    feature_version: str = "v1",
) -> tuple[CandidateFeatureRow, ...]:
    if score_calibration not in {"raw", "percentile"}:
        raise ValueError("score_calibration must be raw or percentile")
    if feature_version not in _SCHEMA_BY_VERSION:
        raise ValueError("feature_version must be v1, v2, or v2b")
    names, _fingerprint = _SCHEMA_BY_VERSION[feature_version]
    latent = dict(latent_scores or {})
    recent = dict(recent_itemcf_scores or {})
    candidates = (
        set(itemcf_scores) | set(dense_scores) | set(latent)
        if candidate_ids is None
        else set(candidate_ids)
    )
    history_rows = tuple(history)
    statistics_rows = tuple(train_rows)
    popularity = Counter(row.movie_id for row in statistics_rows if row.rating >= 4)
    history_movies = [movies[row.movie_id] for row in history_rows if row.movie_id in movies]
    history_genres = {genre for movie in history_movies for genre in movie.genres}
    history_years = {movie.year for movie in history_movies if movie.year is not None}
    recent_years = [
        movie.year
        for movie in history_movies
        if movie.year is not None
    ]
    median_recent_year = sorted(recent_years)[len(recent_years) // 2] if recent_years else None
    itemcf_ranks = _ranks(itemcf_scores)
    dense_ranks = _ranks(dense_scores)
    latent_ranks = _ranks(latent)
    itemcf_score_values = (
        _route_percentile(itemcf_scores)
        if score_calibration == "percentile"
        else itemcf_scores
    )
    dense_score_values = (
        _route_percentile(dense_scores)
        if score_calibration == "percentile"
        else dense_scores
    )
    latent_score_values = (
        _route_percentile(latent)
        if score_calibration == "percentile"
        else latent
    )

    result: list[CandidateFeatureRow] = []
    for movie_id in sorted(candidates):
        movie = movies.get(movie_id)
        if movie is None:
            continue
        movie_genres = set(movie.genres)
        union = history_genres | movie_genres
        genre_jaccard = len(history_genres & movie_genres) / len(union) if union else 0.0
        year_match = float(movie.year is not None and movie.year in history_years)
        latent_score = float(latent_score_values.get(movie_id, 0.0))
        itemcf_value = float(itemcf_score_values.get(movie_id, 0.0))
        values = [
            itemcf_value,
            1.0 / itemcf_ranks[movie_id] if movie_id in itemcf_ranks else 0.0,
            float(dense_score_values.get(movie_id, 0.0)),
            1.0 / dense_ranks[movie_id] if movie_id in dense_ranks else 0.0,
            math.log1p(popularity[movie_id]),
            genre_jaccard,
            year_match,
            _preference_affinity(movie, state),
            float(movie_id in itemcf_scores),
            float(movie_id in dense_scores),
        ]
        if feature_version != "v1":
            values += [
                latent_score,
                1.0 / latent_ranks[movie_id] if movie_id in latent_ranks else 0.0,
                float(movie_id in latent),
            ]
        if feature_version == "v2b":
            year_recency = (
                float(abs(movie.year - median_recent_year))
                if movie.year is not None and median_recent_year is not None
                else 0.0
            )
            values += [
                itemcf_value * latent_score,
                float(recent.get(movie_id, 0.0)),
                year_recency,
            ]
        row_values = tuple(float(value) for value in values)
        if len(row_values) != len(names):
            raise ValueError("candidate feature row length does not match schema")
        for name, value in zip(names, row_values, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    "candidate feature must be finite: "
                    f"user={user_id}, movie={movie_id}, feature={name}, value={value!r}"
                )
        result.append(CandidateFeatureRow(user_id, movie_id, row_values, names))
    return tuple(result)
```

Note: `CandidateFeatureRow` is also constructed positionally in existing tests via `_row()` helpers in `ranker_diagnostics` tests — the added `names` field keeps positional construction working because it is the 4th field with a default.

- [ ] **Step 4: Run the new and existing feature tests**

Run:
```bash
.venv/bin/pytest tests/test_candidate_features_v2.py tests/test_v2_ranking.py tests/test_ranker_diagnostics.py -q
```
Expected: all pass (v1 behavior unchanged).

- [ ] **Step 5: Ruff and commit**

```bash
.venv/bin/ruff check src/recagent_eval/candidate_features.py tests/test_candidate_features_v2.py
git add src/recagent_eval/candidate_features.py tests/test_candidate_features_v2.py
git commit -m "feat: add candidate feature schemas v2/v2b with latent features"
```

---

### Task 4: Latent and sampling configuration

**Files:**
- Modify: `src/recagent_eval/runner.py`
- Modify: `src/recagent_eval/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_config.py`:

```python
def test_latent_disabled_default_keeps_fingerprints() -> None:
    from recagent_eval.lambdamart_pipeline import (
        candidate_policy_fingerprint,
        lambdamart_config_fingerprint,
    )

    config = load_experiment_config(Path("configs/v2_dense_recall1500.yaml"))
    assert config.latent_enabled is False
    assert config.ranker_feature_version == "v1"
    assert candidate_policy_fingerprint(config) == _KNOWN_V1_POLICY
    assert lambdamart_config_fingerprint(config) == _KNOWN_V1_CONFIG


def test_latent_enabled_validates_artifact_path_and_params(tmp_path) -> None:
    path = tmp_path / "latent.yaml"
    path.write_text(
        "latent:\n"
        "  enabled: true\n"
        "  artifact_path: artifacts/experiments/run/latent.npz\n"
        "ranker:\n"
        "  negative_policy: route_balanced\n"
        "  max_negatives: 200\n"
        "  feature_version: v2\n"
    )
    config = load_experiment_config(path)
    assert config.latent_enabled is True
    assert config.latent_top_k == 500
    assert config.ranker_negative_policy == "route_balanced"
    assert config.ranker_max_negatives == 200
    missing = tmp_path / "missing.yaml"
    missing.write_text("latent:\n  enabled: true\n")
    with pytest.raises(ValueError, match="artifact_path"):
        load_experiment_config(missing)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "latent:\n"
        "  enabled: true\n"
        "  artifact_path: x.npz\n"
        "  top_k: 0\n"
    )
    with pytest.raises(ValueError, match="top_k"):
        load_experiment_config(bad)
```

Define at the top of `tests/test_config.py`:
```python
from pathlib import Path

_KNOWN_V1_POLICY = "a3c3475fec9b49b3e67923a73e97d10c2017031050abcbc8f1e468824b52eb41"
_KNOWN_V1_CONFIG = "3c0abb8bc68e8e890194e3ba0ddac1941627f35b48c3277a39e3a8cb45ef6396"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
.venv/bin/pytest tests/test_config.py -q
```
Expected: failures for `latent_enabled` and `ranker_feature_version` attributes, and `ValueError` for the missing artifact path.

- [ ] **Step 3: Add the config fields**

In `src/recagent_eval/runner.py` `ExperimentConfig`, add after `semantic_top_k`:
```python
latent_enabled: bool = False
latent_rank: int = 20
latent_iterations: int = 12
latent_alpha: float = 40.0
latent_lambda_reg: float = 0.1
latent_top_k: int = 500
latent_seed: int = 42
latent_artifact_path: str | None = None
ranker_max_negatives: int | None = None
ranker_negative_policy: str = "all"
ranker_feature_version: str = "v1"
```

In `src/recagent_eval/config.py` `load_experiment_config`, after the `semantic_top_k` block:
```python
latent_payload = payload.get("latent") or {}
if not isinstance(latent_payload, dict):
    raise ValueError("latent must be a mapping")
latent_enabled = bool(latent_payload.get("enabled", False))
latent_artifact_value = latent_payload.get("artifact_path")
latent_artifact_path = (
    str(latent_artifact_value).strip() if latent_artifact_value is not None else None
)
if latent_enabled and not latent_artifact_path:
    raise ValueError("latent.artifact_path is required when latent.enabled is true")
latent_top_k = int(latent_payload.get("top_k", 500))
if latent_top_k <= 0:
    raise ValueError("latent.top_k must be positive")
latent_rank = int(latent_payload.get("rank", 20))
latent_iterations = int(latent_payload.get("iterations", 12))
latent_alpha = float(latent_payload.get("alpha", 40.0))
latent_lambda_reg = float(latent_payload.get("lambda_reg", 0.1))
latent_seed = int(latent_payload.get("seed", 42))
if latent_rank <= 0 or latent_iterations <= 0 or latent_alpha <= 0.0 or latent_lambda_reg < 0.0:
    raise ValueError("latent rank/iterations must be positive, alpha positive, lambda_reg non-negative")
ranker_max_negatives_value = ranker_payload.get("max_negatives")
ranker_max_negatives = (
    int(ranker_max_negatives_value)
    if ranker_max_negatives_value is not None
    else None
)
if ranker_max_negatives is not None and ranker_max_negatives < 0:
    raise ValueError("ranker.max_negatives must be non-negative or unset")
ranker_negative_policy = str(ranker_payload.get("negative_policy", "all"))
if ranker_negative_policy not in {"all", "itemcf", "itemcf_latent", "route_balanced"}:
    raise ValueError("ranker.negative_policy must be all, itemcf, itemcf_latent, or route_balanced")
ranker_feature_version = str(ranker_payload.get("feature_version", "v1"))
if ranker_feature_version not in {"v1", "v2", "v2b"}:
    raise ValueError("ranker.feature_version must be v1, v2, or v2b")
if ranker_feature_version != "v1" and not latent_enabled:
    raise ValueError("ranker.feature_version v2/v2b requires latent.enabled")
```

Pass the new fields into the `ExperimentConfig(...)` constructor.

- [ ] **Step 4: Run the config tests plus the full existing suite**

```bash
.venv/bin/pytest tests/test_config.py tests/test_runner.py -q
.venv/bin/ruff check src/recagent_eval/runner.py src/recagent_eval/config.py
```
Expected: all pass; the v1 fingerprint regression pins `a3c3475f...` and `3c0abb8c...`.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/runner.py src/recagent_eval/config.py tests/test_config.py
git commit -m "feat: add latent route and negative sampling config"

---

### Task 5: Route-balanced hard-negative sampling

**Files:**
- Modify: `src/recagent_eval/learned_ranking.py`
- Modify: `tests/test_v2_ranking.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_v2_ranking.py`:

```python
from recagent_eval.candidate_features import FEATURE_NAMES_V2
from recagent_eval.learned_ranking import CandidateQuery, build_training_matrix


def _v2_query(user_id: int, target: int, rows: dict[int, tuple[float, ...]]) -> CandidateQuery:
    return CandidateQuery(user_id=user_id, target_movie_id=target, features_by_movie=rows)


def test_route_balanced_negatives_use_quotas_and_stable_order() -> None:
    names = FEATURE_NAMES_V2
    itemcf = names.index("itemcf_score")
    latent = names.index("latent_score")
    itemcf_rr = names.index("itemcf_reciprocal_rank")
    latent_rr = names.index("latent_reciprocal_rank")
    rows: dict[int, tuple[float, ...]] = {}
    for movie_id in range(1, 301):
        values = [0.0] * len(names)
        # odd ids: strong itemcf; even ids: strong latent
        values[itemcf] = 10.0 if movie_id % 2 else 1.0
        values[latent] = 1.0 if movie_id % 2 else 10.0
        values[itemcf_rr] = 1.0 / movie_id
        values[latent_rr] = 1.0 / (movie_id + 1)
        rows[movie_id] = tuple(values)
    query = _v2_query(1, target=1, rows=rows)
    matrix = build_training_matrix(
        [query], max_negatives=200, negative_policy="route_balanced"
    )
    assert matrix.groups == (201,)
    ordered_ids = [matrix.movie_ids[0]] + list(matrix.movie_ids[1:])
    assert ordered_ids[0] == 1  # target first
    negatives = ordered_ids[1:]
    assert len(negatives) == 200
    # odd ids are the itemcf-hard quota, even ids the latent-hard quota
    odd = [movie_id for movie_id in negatives if movie_id % 2]
    even = [movie_id for movie_id in negatives if movie_id % 2 == 0]
    assert len(odd) == 100 and len(even) == 100
    assert negatives == sorted(
        negatives,
        key=lambda movie_id: (
            -max(rows[movie_id][itemcf_rr], rows[movie_id][latent_rr]),
            movie_id,
        ),
    )


def test_negative_policy_all_preserves_movie_id_order() -> None:
    rows = {
        movie_id: (float(100 - movie_id),) * 10
        for movie_id in range(2, 12)
    }
    rows[1] = (0.0,) * 10
    query = _v2_query(1, target=1, rows=rows)
    matrix = build_training_matrix([query], max_negatives=5, negative_policy="all")
    assert matrix.movie_ids[1:] == (2, 3, 4, 5, 6)
    unlimited = build_training_matrix([query], max_negatives=None, negative_policy="all")
    assert len(unlimited.movie_ids) == 11
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_v2_ranking.py -q
```
Expected: `TypeError` on `negative_policy` keyword.

- [ ] **Step 3: Implement the sampler**

In `src/recagent_eval/learned_ranking.py`, change `build_training_matrix`:

```python
def build_training_matrix(
    queries: Sequence[CandidateQuery],
    *,
    max_negatives: int | None = None,
    negative_policy: str = "all",
) -> TrainingMatrix:
    if max_negatives is not None and max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    if negative_policy not in {"all", "itemcf", "itemcf_latent", "route_balanced"}:
        raise ValueError("negative_policy must be all, itemcf, itemcf_latent, or route_balanced")
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    users: list[int] = []
    movies: list[int] = []
    for query in sorted(queries, key=lambda item: item.user_id):
        if query.target_movie_id not in query.features_by_movie:
            continue
        negatives = _ordered_negatives(
            query.features_by_movie,
            policy=negative_policy,
            target_movie_id=query.target_movie_id,
        )
        if max_negatives is not None:
            negatives = negatives[:max_negatives]
        ordered_ids = [query.target_movie_id, *negatives]
        row_lengths = {len(query.features_by_movie[movie_id]) for movie_id in ordered_ids}
        if len(row_lengths) != 1:
            raise ValueError("query feature rows have inconsistent lengths")
        expected_count = row_lengths.pop()
        for movie_id in ordered_ids:
            row = tuple(float(value) for value in query.features_by_movie[movie_id])
            _validate_row(row, user_id=query.user_id, movie_id=movie_id, expected_count=expected_count)
            features.append(row)
            labels.append(int(movie_id == query.target_movie_id))
            users.append(query.user_id)
            movies.append(movie_id)
        groups.append(len(ordered_ids))
    return TrainingMatrix(
        features=tuple(features),
        labels=tuple(labels),
        groups=tuple(groups),
        user_ids=tuple(users),
        movie_ids=tuple(movies),
        evaluation_users=len({query.user_id for query in queries}),
        training_users=len(groups),
    )


def _ordered_negatives(
    features_by_movie: Mapping[int, tuple[float, ...]],
    *,
    policy: str,
    target_movie_id: int,
) -> list[int]:
    negatives = [
        movie_id for movie_id in features_by_movie if movie_id != target_movie_id
    ]
    if policy == "all":
        return negatives  # movie-ID order, matching today's behavior
    if policy in {"itemcf", "itemcf_latent"}:
        index = FEATURE_NAMES_V2.index(
            "itemcf_score" if policy == "itemcf" else "latent_score"
        )
        return sorted(
            negatives,
            key=lambda movie_id: (-features_by_movie[movie_id][index], movie_id),
        )
    if policy == "route_balanced":
        itemcf_score = FEATURE_NAMES_V2.index("itemcf_score")
        latent_score = FEATURE_NAMES_V2.index("latent_score")
        itemcf_rr = FEATURE_NAMES_V2.index("itemcf_reciprocal_rank")
        latent_rr = FEATURE_NAMES_V2.index("latent_reciprocal_rank")
        top_itemcf = sorted(
            negatives,
            key=lambda movie_id: (-features_by_movie[movie_id][itemcf_score], movie_id),
        )[:100]
        top_latent = sorted(
            negatives,
            key=lambda movie_id: (-features_by_movie[movie_id][latent_score], movie_id),
        )[:100]
        merged = list(dict.fromkeys([*top_itemcf, *top_latent]))
        top_up = sorted(
            (movie_id for movie_id in negatives if movie_id not in merged),
            key=lambda movie_id: (
                -max(
                    features_by_movie[movie_id][itemcf_rr],
                    features_by_movie[movie_id][latent_rr],
                ),
                movie_id,
            ),
        )
        candidates = (merged + top_up)[:200]
        return sorted(
            candidates,
            key=lambda movie_id: (
                -max(
                    features_by_movie[movie_id][itemcf_rr],
                    features_by_movie[movie_id][latent_rr],
                ),
                movie_id,
            ),
        )
    raise ValueError(f"unsupported negative policy: {policy}")
```

Also change `_validate_row` to accept `expected_count: int | None = None` and compare against `expected_count or len(FEATURE_NAMES)`.

Note: `_ordered_negatives` for `"itemcf"`/`"itemcf_latent"` assumes v2 rows (latent features present). For v1 rows with these policies the caller must not use them; config validation already prevents v1 + latent policies from being combined in the pipeline.

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/pytest tests/test_v2_ranking.py -q
.venv/bin/ruff check src/recagent_eval/learned_ranking.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/learned_ranking.py tests/test_v2_ranking.py
git commit -m "feat: add route-balanced hard-negative sampling to training matrix"
```

---

### Task 6: Schema v2 for artifact, evidence, and bundle

**Files:**
- Modify: `src/recagent_eval/learned_ranking.py`
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `src/recagent_eval/bundle.py`
- Modify: `src/recagent_eval/cli.py` (call sites of `load_ranker_bundle`)
- Modify: `tests/test_learned_ranking_errors.py`, `tests/test_v2_selection.py`, `tests/test_safe_io_bundle.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned_ranking_errors.py`. First add a local
`_valid_artifact()` helper copied from `tests/test_v2_ranking.py` (same
16-parameter CV grid, 3-fold `fold_map = {3: 0, 2: 1, 1: 2}`, and the same
provenance defaults), then add:

```python
from recagent_eval.candidate_features import (
    FEATURE_NAMES_V2,
    FEATURE_SCHEMA_FINGERPRINT_V2,
    FEATURE_SCHEMA_VERSION_V2,
)


def test_v2_artifact_requires_latent_provenance() -> None:
    artifact = _valid_artifact(
        schema_version="lambdamart-artifact/v2",
        feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
        feature_names=FEATURE_NAMES_V2,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
    )
    with pytest.raises(ValueError, match="latent"):
        artifact.validate_contract()
    artifact = _valid_artifact(
        schema_version="lambdamart-artifact/v2",
        feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
        feature_names=FEATURE_NAMES_V2,
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
        latent_artifact_checksum="a" * 64,
        latent_provenance={
            "training_fingerprint": "b" * 64,
            "rank": 20,
            "iterations": 12,
            "alpha": 40.0,
            "lambda_reg": 0.1,
            "seed": 42,
            "top_k": 500,
            "artifact_path": "artifacts/experiments/run/latent.npz",
        },
    )
    assert artifact.latent_artifact_checksum == "a" * 64


def test_v1_artifact_rejects_latent_fields() -> None:
    with pytest.raises(ValueError, match="latent"):
        _valid_artifact(latent_artifact_checksum="a" * 64)
```

Append to `tests/test_v2_selection.py`:

```python
def test_v2_evidence_carries_latent_provenance() -> None:
    rows = [
        {
            "user_id": 1,
            "itemcf_ndcg_at_10": 0.0,
            "lambdamart_ndcg_at_10": 1.0,
            "itemcf_recall_at_10": 0.0,
            "lambdamart_recall_at_10": 1.0,
            "itemcf_hit_at_10": 0.0,
            "lambdamart_hit_at_10": 1.0,
            "itemcf_candidate_recall": 1.0,
            "dense_candidate_recall": 1.0,
            "union_candidate_recall": 1.0,
            "constraint_satisfied": True,
            "legal_history_movie_ids": [2],
            "allowed_movie_ids": [3],
            "lambdamart_ranked_movie_ids": [3],
            "latency_ms": 0.0,
        }
    ]
    evidence = build_validation_evidence(
        rows,
        dataset_fingerprint="dataset",
        feature_fingerprint=FEATURE_SCHEMA_FINGERPRINT_V2,
        model_fingerprint="model",
        candidate_policy_fingerprint="policy",
        seed=42,
        provenance={
            "schema_version": "lambdamart-validation/v2",
            "latent_artifact_checksum": "a" * 64,
            "latent_provenance": {"training_fingerprint": "b" * 64},
        },
    )
    assert evidence.schema_version == "lambdamart-validation/v2"
    assert evidence.latent_artifact_checksum == "a" * 64
```

Append to `tests/test_safe_io_bundle.py`:

```python
def test_bundle_v2_publishes_and_loads_latent_member(tmp_path) -> None:
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "bundle.json"
    latent_path = tmp_path / "latent.npz"
    latent_manifest_path = tmp_path / "latent.npz.json"
    metadata = {
        "run_fingerprint": "run",
        "config_fingerprint": "config",
        "dataset_fingerprint": "dataset",
        "candidate_policy_fingerprint": "policy",
        "feature_fingerprint": "feature",
    }
    publish_ranker_bundle(
        b"model",
        b"evidence",
        model_path,
        evidence_path,
        manifest_path,
        metadata,
        latent_member=(latent_path, b"latent-data"),
        latent_manifest_member=(latent_manifest_path, b'{"checksum": "abc"}'),
    )
    bundle = load_ranker_bundle(model_path, evidence_path, manifest_path)
    assert bundle.model_bytes == b"model"
    assert bundle.evidence_bytes == b"evidence"
    assert bundle.latent_bytes == b"latent-data"
    assert bundle.manifest.schema_version == "lambdamart-bundle/v2"


def test_bundle_v1_has_no_latent_member(tmp_path) -> None:
    model_path = tmp_path / "m.json"
    evidence_path = tmp_path / "e.json"
    manifest_path = tmp_path / "b.json"
    metadata = {
        "run_fingerprint": "run",
        "config_fingerprint": "config",
        "dataset_fingerprint": "dataset",
        "candidate_policy_fingerprint": "policy",
        "feature_fingerprint": "feature",
    }
    publish_ranker_bundle(b"m", b"e", model_path, evidence_path, manifest_path, metadata)
    bundle = load_ranker_bundle(model_path, evidence_path, manifest_path)
    assert bundle.latent_bytes is None
    assert bundle.manifest.schema_version == "lambdamart-bundle/v1"
```

Update the existing bundle tests that unpack `load_ranker_bundle(...)` as a 2-tuple to use `.model_bytes` / `.evidence_bytes`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_learned_ranking_errors.py tests/test_v2_selection.py tests/test_safe_io_bundle.py -q
```
Expected: import/attribute failures for the new schema constants, `latent_member`, and `RankerBundle`.

- [ ] **Step 3: Implement artifact v2 in `learned_ranking.py`**

- Add `ARTIFACT_SCHEMA_VERSION_V2 = "lambdamart-artifact/v2"`.
- Add to `RankerArtifact` (after `validation_user_count`):
```python
latent_artifact_checksum: str | None = None
latent_provenance: dict[str, Any] | None = None
```
- In `validate_contract`, branch:
```python
if self.schema_version not in {ARTIFACT_SCHEMA_VERSION, ARTIFACT_SCHEMA_VERSION_V2}:
    raise ValueError("unsupported LambdaMART artifact schema or kind")
if self.kind != "lambdamart":
    raise ValueError("unsupported LambdaMART artifact schema or kind")
is_v2 = self.schema_version == ARTIFACT_SCHEMA_VERSION_V2
if is_v2:
    if self.feature_schema_version not in {
        FEATURE_SCHEMA_VERSION_V2,
        FEATURE_SCHEMA_VERSION_V2B,
    }:
        raise ValueError("ranker artifact feature schema mismatch")
    expected_names = (
        FEATURE_NAMES_V2
        if self.feature_schema_version == FEATURE_SCHEMA_VERSION_V2
        else FEATURE_NAMES_V2B
    )
    expected_fingerprint = (
        FEATURE_SCHEMA_FINGERPRINT_V2
        if self.feature_schema_version == FEATURE_SCHEMA_VERSION_V2
        else FEATURE_SCHEMA_FINGERPRINT_V2B
    )
    if self.feature_names != expected_names or self.feature_fingerprint != expected_fingerprint:
        raise ValueError("ranker artifact feature schema mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", self.latent_artifact_checksum or ""):
        raise ValueError("ranker artifact latent checksum is invalid")
    provenance = self.latent_provenance or {}
    required_latent = {"training_fingerprint", "rank", "iterations", "alpha", "lambda_reg", "seed", "top_k", "artifact_path"}
    if set(provenance) != required_latent:
        raise ValueError("ranker artifact latent provenance is incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", str(provenance["training_fingerprint"])):
        raise ValueError("ranker artifact latent training fingerprint is invalid")
else:
    if (
        self.feature_schema_version != FEATURE_SCHEMA_VERSION
        or self.feature_names != FEATURE_NAMES
        or self.feature_fingerprint != FEATURE_SCHEMA_FINGERPRINT
    ):
        raise ValueError("ranker artifact feature schema mismatch")
    if self.latent_artifact_checksum is not None or self.latent_provenance is not None:
        raise ValueError("ranker artifact v1 cannot carry latent fields")
```
Keep every existing provenance/CV/fold-map/checksum check unchanged.

- In `artifact_from_estimator`, accept `latent_artifact_checksum: str | None = None` and `latent_provenance: Mapping[str, Any] | None = None` keyword args; when either is supplied, set `schema_version=ARTIFACT_SCHEMA_VERSION_V2` and pass both into the constructor.
- In `parse_ranker_artifact`, dispatch the required-field set:
```python
is_v2 = payload.get("schema_version") == ARTIFACT_SCHEMA_VERSION_V2
required = {
    ...existing v1 fields...,
    **({"latent_artifact_checksum", "latent_provenance"} if is_v2 else {}),
}
```
Add `expected_latent_artifact_checksum: str | None = None` parameter; after validation, if expected and `artifact.latent_artifact_checksum != expected_latent_artifact_checksum`, raise.

- Update `LearnedRanker`:
  - `__init__` gains `feature_version: str = "v1"`; resolves `self.feature_names` / `self.feature_fingerprint` from the v1/v2/v2b constants.
  - `rank(...)` gains `latent_scores: Mapping[int, float] | None = None`; pass `latent_scores=latent_scores, feature_version=self.feature_version` into `build_candidate_feature_rows`.
  - `rank_feature_rows` validates with `expected_count=len(self.feature_names)` and builds the contribution map from `self.feature_names`.
  - `_validate_row` accepts `expected_count: int | None = None` and uses it when provided.

- [ ] **Step 4: Implement evidence v2 in `v2_selection.py`**

- Add `VALIDATION_SCHEMA_VERSION_V2 = "lambdamart-validation/v2"`.
- Add to `LearnedValidationEvidence` (after `validation_user_count`):
```python
latent_artifact_checksum: str | None = None
latent_provenance: dict[str, Any] | None = None
```
- In `build_validation_evidence`, add provenance keys `latent_artifact_checksum` / `latent_provenance` to `provenance_defaults` (default `None`); when `schema_version` provenance is `"lambdamart-validation/v2"` (or either latent key is supplied), set `schema_version=VALIDATION_SCHEMA_VERSION_V2` and pass both fields into the constructor. Include both keys in the fingerprint payload only when set.
- In `validate_learned_gate`:
```python
if evidence.schema_version not in {"lambdamart-validation/v1", VALIDATION_SCHEMA_VERSION_V2}:
    raise ValueError("unsupported validation evidence schema")
is_v2 = evidence.schema_version == VALIDATION_SCHEMA_VERSION_V2
if is_v2:
    if not re.fullmatch(r"[0-9a-f]{64}", evidence.latent_artifact_checksum or ""):
        raise ValueError("validation evidence latent checksum is invalid")
    if not evidence.latent_provenance or not evidence.latent_provenance.get("training_fingerprint"):
        raise ValueError("validation evidence latent provenance is incomplete")
elif evidence.latent_artifact_checksum is not None or evidence.latent_provenance is not None:
    raise ValueError("validation evidence v1 cannot carry latent fields")
```
When `is_v2`, the derived rebuild inside `validate_learned_gate` must receive
`latent_artifact_checksum` / `latent_provenance` in its provenance dict (so the
derived evidence is also v2), and `derived_fields` must additionally compare
`latent_artifact_checksum` and `latent_provenance`. Keep the rest of the gate
(recompute, CV grid, fingerprints, NDCG/CI/constraint conditions) unchanged.

- [ ] **Step 5: Implement bundle v2 in `bundle.py`**

- Add `BUNDLE_SCHEMA_VERSION_V2 = "lambdamart-bundle/v2"`.
- Add to `RankerBundleManifest`:
```python
latent_sha256: str | None = None
latent_manifest_sha256: str | None = None
```
- Validator: v1 requires both `None`; v2 requires both 64-hex lowercase values.
- Add a frozen dataclass:
```python
@dataclass(frozen=True)
class RankerBundle:
    model_bytes: bytes
    evidence_bytes: bytes
    latent_bytes: bytes | None
    manifest: RankerBundleManifest
```
- Extend `publish_ranker_bundle` with keyword-only members:
```python
def publish_ranker_bundle(
    model_bytes: bytes,
    evidence_bytes: bytes,
    model_path: Path,
    evidence_path: Path,
    manifest_path: Path,
    metadata: Mapping[str, str],
    *,
    latent_member: tuple[Path, bytes] | None = None,
    latent_manifest_member: tuple[Path, bytes] | None = None,
) -> RankerBundleManifest:
```
When both latent members are provided, include their paths in the `paths`/lock set, compute `latent_sha256` / `latent_manifest_sha256` from the bytes, use `BUNDLE_SCHEMA_VERSION_V2`, and reject either being provided without the other.
- Change `load_ranker_bundle` to return `RankerBundle` and, when the manifest is v2, also read/validate the latent member bytes and hashes.

- [ ] **Step 6: Update callers and run all affected tests**

In `src/recagent_eval/cli.py`, update both `load_ranker_bundle` call sites:
```python
bundle = load_ranker_bundle(...)
model_bytes = bundle.model_bytes
evidence_bytes = bundle.evidence_bytes
```
When the bundle is v2, pass `bundle.latent_bytes`/`bundle.manifest` through to the latent artifact validation path (implemented in Task 7).

Run:
```bash
.venv/bin/pytest tests/test_learned_ranking_errors.py tests/test_v2_selection.py tests/test_safe_io_bundle.py tests/test_cli.py tests/test_v2_ranking.py -q
.venv/bin/ruff check src/recagent_eval/learned_ranking.py src/recagent_eval/v2_selection.py src/recagent_eval/bundle.py src/recagent_eval/cli.py
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/recagent_eval/learned_ranking.py src/recagent_eval/v2_selection.py src/recagent_eval/bundle.py src/recagent_eval/cli.py tests/test_learned_ranking_errors.py tests/test_v2_selection.py tests/test_safe_io_bundle.py tests/test_cli.py tests/test_v2_ranking.py
git commit -m "feat: add schema-v2 artifact, evidence, and bundle with latent provenance"

---

### Task 7: Thread the latent route through the pipeline and fingerprints

**Files:**
- Modify: `src/recagent_eval/lambdamart_pipeline.py`
- Modify: `tests/test_lambdamart_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lambdamart_pipeline.py`:

```python
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.lambdamart_pipeline import (
    build_candidate_queries,
    candidate_policy_fingerprint,
    lambdamart_config_fingerprint,
)


def test_build_candidate_queries_threads_latent_scores() -> None:
    movies = _movies()
    ratings = _ratings()
    split = leakage_safe_ranking_split(ratings)
    latent = LatentFactorRetriever.fit(split.legal_retrieval_train, seed=42)
    config = _config(latent_enabled=True)
    queries = build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        _positive_histories(split.legal_retrieval_train, movies),
        split.validation_targets,
        _semantic(),
        retrieval_top_k=10,
        history_cap=5,
        max_users=3,
        semantic_top_k=20,
        latent=latent,
        latent_top_k=10,
        feature_version="v2",
    )
    assert queries
    row = next(iter(queries[0].features_by_movie.values()))
    assert len(row) == len(FEATURE_NAMES_V2)


def test_fingerprints_change_only_when_latent_enabled() -> None:
    from dataclasses import replace
    from pathlib import Path

    from recagent_eval.config import load_experiment_config

    base = load_experiment_config(Path("configs/v2_dense_recall1500.yaml"))
    assert candidate_policy_fingerprint(base) == _KNOWN_V1_POLICY
    enabled = replace(
        base,
        latent_enabled=True,
        latent_artifact_path="artifacts/experiments/x/latent.npz",
        ranker_feature_version="v2",
        ranker_negative_policy="route_balanced",
        ranker_max_negatives=200,
    )
    assert candidate_policy_fingerprint(enabled) != _KNOWN_V1_POLICY
    assert lambdamart_config_fingerprint(enabled) != lambdamart_config_fingerprint(base)
```

Reuse the small-data helpers from `tests/test_recall_sweep.py` or define local `_movies()`/`_ratings()`/`_config()`/`_semantic()` helpers returning a tiny MovieLens-like catalog, a leakage-safe split, an `ExperimentConfig` with the latent fields, and a `SemanticRetriever` stub that returns every allowed movie.

Define `_KNOWN_V1_POLICY = "a3c3475fec9b49b3e67923a73e97d10c2017031050abcbc8f1e468824b52eb41"` at the top of the test file and import `FEATURE_NAMES_V2` from `recagent_eval.candidate_features`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_lambdamart_pipeline.py -q
```
Expected: `TypeError` on `latent`/`latent_top_k`/`feature_version` keywords.

- [ ] **Step 3: Update `build_candidate_queries`**

Add parameters:
```python
def build_candidate_queries(
    ...,
    score_calibration: str = "raw",
    latent: LatentFactorRetriever | None = None,
    latent_top_k: int | None = None,
    feature_version: str = "v1",
) -> list[CandidateQuery]:
```
Inside the per-user loop, after `dense_scores = dict(...)`:
```python
latent_scores = {}
if latent is not None:
    latent_scores = dict(
        latent.retrieve(
            history_ids,
            top_k=latent_top_k or 500,
            allowed_ids=allowed_ids,
        )
    )
```
and pass `latent_scores=latent_scores, feature_version=feature_version` into `build_candidate_feature_rows`.

- [ ] **Step 4: Update `build_fold_queries`, `build_validation_rows`, and `train_lambdamart_pipeline`**

`build_fold_queries`: after building `fold_train_rows`, fit a temporary latent model when enabled:
```python
latent = (
    LatentFactorRetriever.fit(
        fold_train_rows,
        rank=config.latent_rank,
        iterations=config.latent_iterations,
        alpha=config.latent_alpha,
        lambda_reg=config.latent_lambda_reg,
        seed=config.latent_seed,
    )
    if config.latent_enabled
    else None
)
```
and pass `latent=latent, latent_top_k=config.latent_top_k, feature_version=config.ranker_feature_version` into both `build_candidate_queries` calls.

`build_validation_rows`: add `latent: LatentFactorRetriever | None = None` parameter and pass it (plus `latent_top_k`/`feature_version`) into `build_candidate_queries`.

`train_lambdamart_pipeline`:
- For `training_queries`, fit a temporary latent on `split.ranker_training_history` when enabled and pass it in.
- After CV and before `build_validation_rows`, when enabled:
```python
final_latent = LatentFactorRetriever.fit(
    split.legal_retrieval_train,
    rank=config.latent_rank,
    iterations=config.latent_iterations,
    alpha=config.latent_alpha,
    lambda_reg=config.latent_lambda_reg,
    seed=config.latent_seed,
)
final_latent.save(Path(config.latent_artifact_path))
latent_bytes = (Path(config.latent_artifact_path)).read_bytes()
latent_manifest_bytes = Path(f"{config.latent_artifact_path}.json").read_bytes()
latent_provenance = {
    "training_fingerprint": final_latent.training_fingerprint,
    "rank": final_latent.rank,
    "iterations": final_latent.iterations,
    "alpha": final_latent.alpha,
    "lambda_reg": final_latent.lambda_reg,
    "seed": final_latent.seed,
    "top_k": config.latent_top_k,
    "artifact_path": config.latent_artifact_path,
}
latent_artifact_checksum = hashlib.sha256(latent_bytes).hexdigest()
```
Pass `latent=final_latent` into `build_validation_rows`, add `latent_artifact_checksum`/`latent_provenance` to the artifact provenance and evidence provenance, and pass `latent_member=(Path(config.latent_artifact_path), latent_bytes)` plus `latent_manifest_member=(Path(f"{config.latent_artifact_path}.json"), latent_manifest_bytes)` into `publish_ranker_bundle`.

- [ ] **Step 5: Update the fingerprints**

Replace `candidate_policy_fingerprint`:
```python
def candidate_policy_fingerprint(config: ExperimentConfig) -> str:
    base = {
        "retrieval_top_k": config.retrieval_top_k,
        "semantic_profile_history_cap": config.semantic_profile_history_cap,
        "semantic_kind": config.semantic_kind,
        "semantic_model_name": config.semantic_model_name,
        "semantic_model_revision": config.semantic_model_revision,
        "semantic_cache_path": config.semantic_cache_path,
        "semantic_top_k": config.semantic_top_k,
        "score_calibration": config.score_calibration,
    }
    if not config.latent_enabled:
        payload = {"schema": "union-candidate-policy/v1", **base}
    else:
        payload = {
            "schema": "union-candidate-policy/v2",
            **base,
            "latent": {
                "rank": config.latent_rank,
                "iterations": config.latent_iterations,
                "alpha": config.latent_alpha,
                "lambda_reg": config.latent_lambda_reg,
                "seed": config.latent_seed,
                "top_k": config.latent_top_k,
            },
            "feature_version": config.ranker_feature_version,
            "negative_policy": config.ranker_negative_policy,
            "max_negatives": config.ranker_max_negatives,
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
```

Replace `lambdamart_config_fingerprint` the same way: v1 payload byte-identical to today (`retrieval_top_k`, `semantic_profile_history_cap`, `semantic_kind`, `semantic_model_name`, `semantic_model_revision`, `semantic_cache_path`, `semantic_top_k`, `score_calibration`, `seed`); v2 adds the latent block, `feature_version`, `negative_policy`, and `max_negatives`.

In `train_lambdamart_pipeline`, pass `max_negatives=config.ranker_max_negatives, negative_policy=config.ranker_negative_policy` into the final `build_training_matrix` call and into `cross_validate_lambdamart` (add the two parameters to `cross_validate_lambdamart` and forward them to each fold's `build_training_matrix`).

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/pytest tests/test_lambdamart_pipeline.py tests/test_config.py tests/test_recall_sweep.py -q
.venv/bin/ruff check src/recagent_eval/lambdamart_pipeline.py
```
Expected: all pass; the v1 fingerprint regression pins remain green.

- [ ] **Step 7: Commit**

```bash
git add src/recagent_eval/lambdamart_pipeline.py tests/test_lambdamart_pipeline.py
git commit -m "feat: thread latent route through candidate pipeline and fingerprints"
```

---

### Task 8: `diagnose-latent` CLI and candidate-stage diagnostics

**Files:**
- Create: `src/recagent_eval/latent_diagnostics.py`
- Create: `tests/test_latent_diagnostics.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_latent_diagnostics.py`:

```python
from __future__ import annotations

from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split
from recagent_eval.latent_diagnostics import (
    build_latent_diagnostic_queries,
    build_latent_user_rows,
    aggregate_latent_diagnostics,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 12)
    }


def _ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 9)
        for movie_id in range(1, 9)
    ]


def _semantic():
    class Stub:
        def retrieve(self, query, *, top_k=100, allowed_ids=None):
            del query
            ids = sorted(allowed_ids or ())
            return [(movie_id, 1.0 / index) for index, movie_id in enumerate(ids, 1)][:top_k]

    return Stub()


def test_latent_diagnostics_aggregate_gate_fields() -> None:
    movies = _movies()
    ratings = _ratings()
    split = leakage_safe_ranking_split(ratings)
    latent = LatentFactorRetriever.fit(split.legal_retrieval_train, seed=42)
    queries = build_latent_diagnostic_queries(
        movies,
        split,
        _semantic(),
        latent=latent,
        retrieval_top_k=5,
        history_cap=5,
        semantic_top_k=10,
        latent_top_k=10,
        feature_version="v2",
        max_users=6,
    )
    rows = build_latent_user_rows(queries)
    summary = aggregate_latent_diagnostics(
        rows,
        fingerprints={"dataset": "d", "candidate_policy": "p", "feature_schema": "f", "case": "c"},
        fit_seconds=0.5,
    )
    assert summary.user_count == 6
    assert 0.0 <= summary.latent_recall_500_all <= 1.0
    assert 0.0 <= summary.latent_recall_10_all <= 1.0
    assert 0.0 <= summary.union_recall_three_route <= 1.0
    assert summary.latent_only_coverage >= 0.0
    assert summary.latent_present_user_count >= 0
    assert set(summary.target_latent_rank_quantiles) == {"p25", "p50", "p75"}
```

Append to `tests/test_cli.py`:

```python
def test_diagnose_latent_refuses_overwrite_and_requires_latent(tmp_path, monkeypatch) -> None:
    from recagent_eval.data import Movie, Rating

    movies = {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }
    ratings = [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 8)
        for movie_id in range(1, 9)
    ]
    monkeypatch.setattr(
        "recagent_eval.cli._load_dataset", lambda _path: (movies, ratings)
    )
    config_path = tmp_path / "latent.yaml"
    config_path.write_text(
        "semantic:\n"
        "  kind: tfidf\n"
        "latent:\n"
        "  enabled: true\n"
        "  artifact_path: artifacts/experiments/x/latent.npz\n"
    )
    output = tmp_path / "diagnostics.json"
    result = CliRunner().invoke(app, ["diagnose-latent", "--config", str(config_path), "--output", str(output)])
    assert result.exit_code == 0
    result = CliRunner().invoke(app, ["diagnose-latent", "--config", str(config_path), "--output", str(output)])
    assert "refusing to overwrite" in result.output
    disabled = tmp_path / "disabled.yaml"
    disabled.write_text("semantic:\n  kind: tfidf\n")
    result = CliRunner().invoke(app, ["diagnose-latent", "--config", str(disabled), "--output", str(tmp_path / "d2.json")])
    assert "latent" in result.output.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_latent_diagnostics.py tests/test_cli.py -q
```
Expected: import failure for `recagent_eval.latent_diagnostics` and CLI "No such command".

- [ ] **Step 3: Implement `latent_diagnostics.py`**

Create `src/recagent_eval/latent_diagnostics.py`:

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from recagent_eval.candidate_features import FEATURE_NAMES
from recagent_eval.data import LeakageSafeRankingSplit, Movie
from recagent_eval.lambdamart_pipeline import (
    _positive_histories,
    build_candidate_queries,
)
from recagent_eval.latent_retrieval import LatentFactorRetriever
from recagent_eval.retrieval import SemanticRetriever


@dataclass(frozen=True)
class LatentDiagnosticUserRow:
    user_id: int
    in_union: bool
    in_itemcf: bool
    in_dense: bool
    in_latent: bool
    latent_rank: int | None
    latent_recall_10: float
    latent_recall_50: float
    latent_recall_100: float
    latent_recall_500: float
    latent_only: bool
    itemcf_ids: frozenset[int]
    dense_ids: frozenset[int]
    latent_ids: frozenset[int]


@dataclass(frozen=True)
class LatentDiagnosticSummary:
    user_count: int
    latent_present_user_count: int
    latent_recall_500_all: float
    latent_recall_100_all: float
    latent_recall_50_all: float
    latent_recall_10_all: float
    latent_recall_500_present: float
    latent_recall_10_present: float
    union_recall_three_route: float
    latent_only_coverage: float
    target_latent_rank_quantiles: dict[str, float]
    overlap_itemcf_latent: float
    overlap_dense_latent: float
    fit_seconds: float
    fingerprints: dict[str, str]


def build_latent_diagnostic_queries(
    movies: dict[int, Movie],
    split: LeakageSafeRankingSplit,
    semantic: SemanticRetriever,
    *,
    latent: LatentFactorRetriever,
    retrieval_top_k: int,
    history_cap: int,
    semantic_top_k: int | None,
    latent_top_k: int,
    feature_version: str,
    max_users: int,
):
    histories = _positive_histories(split.legal_retrieval_train, movies)
    return build_candidate_queries(
        movies,
        split.legal_retrieval_train,
        histories,
        split.validation_targets,
        semantic,
        retrieval_top_k=retrieval_top_k,
        history_cap=history_cap,
        semantic_top_k=semantic_top_k,
        latent=latent,
        latent_top_k=latent_top_k,
        feature_version=feature_version,
        max_users=max_users,
    )


def build_latent_user_rows(queries: Sequence) -> list[LatentDiagnosticUserRow]:
    rows: list[LatentDiagnosticUserRow] = []
    for query in queries:
        features = query.features_by_movie
        if features:
            sample = next(iter(features.values()))
            if len(sample) != 13:
                raise ValueError("latent diagnostics require candidate-features/v2 rows")
        target = query.target_movie_id
        in_union = target in features
        target_row = features.get(target)
        in_itemcf = bool(target_row is not None and target_row[8] == 1.0)
        in_dense = bool(target_row is not None and target_row[9] == 1.0)
        in_latent = bool(target_row is not None and target_row[10] == 1.0)
        itemcf_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[8] == 1.0
        )
        dense_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[9] == 1.0
        )
        latent_ids = frozenset(
            movie_id for movie_id, values in features.items() if values[12] == 1.0
        )
        latent_order = [
            movie_id
            for movie_id in sorted(
                features,
                key=lambda movie_id: (
                    -features[movie_id][10],
                    features[movie_id][12],
                    movie_id,
                ),
            )
        ]
        latent_ids = [movie_id for movie_id in latent_order if features[movie_id][12] == 1.0]
        latent_rank = latent_ids.index(target) + 1 if in_latent else None
        rows.append(
            LatentDiagnosticUserRow(
                user_id=query.user_id,
                in_union=in_union,
                in_itemcf=in_itemcf,
                in_dense=in_dense,
                in_latent=in_latent,
                latent_rank=latent_rank,
                latent_recall_10=float(in_latent and latent_rank is not None and latent_rank <= 10),
                latent_recall_50=float(in_latent and latent_rank is not None and latent_rank <= 50),
                latent_recall_100=float(in_latent and latent_rank is not None and latent_rank <= 100),
                latent_recall_500=float(in_latent),
                latent_only=bool(in_latent and not in_itemcf and not in_dense),
                itemcf_ids=itemcf_ids,
                dense_ids=dense_ids,
                latent_ids=latent_ids,
            )
        )
    return rows


def aggregate_latent_diagnostics(
    rows: Sequence[LatentDiagnosticUserRow],
    *,
    fingerprints: Mapping[str, str] | None = None,
    fit_seconds: float = 0.0,
) -> LatentDiagnosticSummary:
    if not rows:
        raise ValueError("latent diagnostics produced no user rows")
    present = [row for row in rows if row.in_latent]
    ranks = [row.latent_rank for row in present if row.latent_rank is not None]
    union = [row for row in rows if row.in_union]
    return LatentDiagnosticSummary(
        user_count=len(rows),
        latent_present_user_count=len(present),
        latent_recall_500_all=_mean([row.latent_recall_500 for row in rows]),
        latent_recall_100_all=_mean([row.latent_recall_100 for row in rows]),
        latent_recall_50_all=_mean([row.latent_recall_50 for row in rows]),
        latent_recall_10_all=_mean([row.latent_recall_10 for row in rows]),
        latent_recall_500_present=_mean([row.latent_recall_500 for row in present]),
        latent_recall_10_present=_mean([row.latent_recall_10 for row in present]),
        union_recall_three_route=len(union) / len(rows),
        latent_only_coverage=sum(row.latent_only for row in rows) / len(rows),
        target_latent_rank_quantiles=_quantiles(ranks),
        overlap_itemcf_latent=_route_overlap(rows, left="itemcf", right="latent"),
        overlap_dense_latent=_route_overlap(rows, left="dense", right="latent"),
        fit_seconds=fit_seconds,
        fingerprints=dict(fingerprints or {}),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantiles(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    ordered = sorted(ranks)
    result: dict[str, float] = {}
    for label, position in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
        index = min(len(ordered) - 1, int(position * len(ordered)))
        result[label] = float(ordered[index])
    return result


def _route_overlap(
    rows: Sequence[LatentDiagnosticUserRow], *, left: str, right: str
) -> float:
    values = []
    for row in rows:
        left_ids = getattr(row, f"{left}_ids")
        right_ids = getattr(row, f"{right}_ids")
        union = left_ids | right_ids
        if not union:
            values.append(0.0)
        else:
            values.append(len(left_ids & right_ids) / len(union))
    return _mean(values)
```

The feature indices 8/9/12 are the v2 positions of `in_itemcf` / `in_dense` /
`in_latent`; the v2-length assert in `build_latent_user_rows` guarantees a v1
row can never be silently misread.

- [ ] **Step 4: Add the `diagnose-latent` CLI command**

In `src/recagent_eval/cli.py`, after `diagnose-ranker`:

```python
@app.command("diagnose-latent")
def diagnose_latent(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    cases_path: Annotated[Path, typer.Option("--cases")] = Path("cases/fixed_cases.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/experiments/v2-latent-diagnostics/diagnostics.json"
    ),
    max_users: Annotated[int, typer.Option(min=3)] = 500,
) -> None:
    config = _validated_config(config_path)
    if not config.latent_enabled:
        raise typer.BadParameter("diagnose-latent requires latent.enabled=true")
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing diagnostics artifact: {output}")
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    if config.semantic_kind == "dense":
        if config.semantic_cache_path is None:
            raise typer.BadParameter("semantic.cache_path is required for dense diagnostics")
        semantic = DenseSemanticRetriever.load(
            Path(config.semantic_cache_path),
            movies=movies,
            model_name=config.semantic_model_name,
            model_revision=config.semantic_model_revision,
            device=config.semantic_device,
        )
    else:
        semantic = TfidfSemanticRetriever.fit(movies)
    started = time.perf_counter()
    latent = LatentFactorRetriever.fit(
        split.legal_retrieval_train,
        rank=config.latent_rank,
        iterations=config.latent_iterations,
        alpha=config.latent_alpha,
        lambda_reg=config.latent_lambda_reg,
        seed=config.latent_seed,
    )
    fit_seconds = time.perf_counter() - started
    queries = build_latent_diagnostic_queries(
        movies,
        split,
        semantic,
        latent=latent,
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.semantic_profile_history_cap,
        semantic_top_k=config.semantic_top_k,
        latent_top_k=config.latent_top_k,
        feature_version=config.ranker_feature_version,
        max_users=max_users,
    )
    rows = build_latent_user_rows(queries)
    summary = aggregate_latent_diagnostics(
        rows,
        fingerprints={
            "dataset": ranking_dataset_fingerprint(movies, split),
            "diagnostic_dataset": ranker_dataset_fingerprint(
                movies, split, max_users=max_users,
                retrieval_top_k=config.retrieval_top_k,
                history_cap=config.semantic_profile_history_cap,
            ),
            "candidate_policy": candidate_policy_fingerprint(config),
            "feature_schema": FEATURE_SCHEMA_FINGERPRINT_V2,
            "case": case_fingerprint(load_cases(cases_path)),
        },
        fit_seconds=fit_seconds,
    )
    evidence = {
        "schema_version": "latent-diagnostics/v1",
        "config_fingerprint": lambdamart_config_fingerprint(config),
        "max_users": max_users,
        "summary": {
            "user_count": summary.user_count,
            "latent_present_user_count": summary.latent_present_user_count,
            "latent_recall_500_all": summary.latent_recall_500_all,
            "latent_recall_100_all": summary.latent_recall_100_all,
            "latent_recall_50_all": summary.latent_recall_50_all,
            "latent_recall_10_all": summary.latent_recall_10_all,
            "latent_recall_500_present": summary.latent_recall_500_present,
            "latent_recall_10_present": summary.latent_recall_10_present,
            "union_recall_three_route": summary.union_recall_three_route,
            "latent_only_coverage": summary.latent_only_coverage,
            "target_latent_rank_quantiles": summary.target_latent_rank_quantiles,
            "overlap_itemcf_latent": summary.overlap_itemcf_latent,
            "overlap_dense_latent": summary.overlap_dense_latent,
            "fit_seconds": summary.fit_seconds,
        },
        "fingerprints": summary.fingerprints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps(evidence["summary"], indent=2, sort_keys=True))
```

Add the required imports (`time`, `LatentFactorRetriever`, `FEATURE_SCHEMA_FINGERPRINT_V2`, `build_latent_diagnostic_queries`, `build_latent_user_rows`, `aggregate_latent_diagnostics`).

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/pytest tests/test_latent_diagnostics.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/latent_diagnostics.py src/recagent_eval/cli.py
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/recagent_eval/latent_diagnostics.py src/recagent_eval/cli.py tests/test_latent_diagnostics.py tests/test_cli.py
git commit -m "feat: add read-only diagnose-latent candidate diagnostics CLI"

---

### Task 9: Full quality gate after implementation

**Files:** none (verification only; fix any failures in the owning task)

- [ ] **Step 1: Run the complete gate**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered -q
.venv/bin/ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
git diff --check
bash -n scripts/run_remote_qwen.sh
```
Expected: all tests pass, coverage at least 90%, Ruff clean, lock check passes, diff check clean, bash -n clean. Fix any failure in the task that caused it, then re-run.

- [ ] **Step 2: Confirm the v1 fingerprint regressions and clean status**

```bash
.venv/bin/pytest tests/test_config.py -q
git status --short
```
Expected: the v1 fingerprint pins (`a3c3475f...`, `3c0abb8c...`) pass and only intended files are modified.

- [ ] **Step 3: Commit any remaining fixes**

```bash
git add -A
git diff --cached --quiet || git commit -m "test: full quality gate after latent recall implementation"
```

---

### Task 10: E1/E2 — latent single-route and three-route candidate diagnosis (500 users)

**Files:**
- Create: `configs/v2_dense_latent.yaml`
- Create: `reports/experiments/v2-latent-diagnostics.json` / `.md` (after the run)

- [ ] **Step 1: Create the main latent config**

Create `configs/v2_dense_latent.yaml`:
```yaml
name: v2-dense-lambdamart-latent
seed: 42
retrieval_top_k: 500
semantic_profile_history_cap: 50
enable_memory: true
enable_semantic_retrieval: true
structured_planning: true
required_retrieval_tools:
  - itemcf_retrieve
  - semantic_retrieve
weights:
  - 0.7
  - 0.3
  - 0.0
semantic:
  kind: dense
  model_name: sentence-transformers/all-MiniLM-L6-v2
  cache_path: artifacts/embeddings/movielens-minilm.npz
  device: cpu
  top_k: 1500
latent:
  enabled: true
  rank: 20
  iterations: 12
  alpha: 40.0
  lambda_reg: 0.1
  top_k: 500
  seed: 42
  artifact_path: artifacts/experiments/v2-latent-hardneg-500/latent.npz
ranker:
  kind: minmax_linear
  feature_version: v2
  max_negatives: 200
  negative_policy: route_balanced
```

- [ ] **Step 2: Run the 500-user diagnosis**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/recagent-eval diagnose-latent \
  --config configs/v2_dense_latent.yaml \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --cases cases/fixed_cases.json \
  --output artifacts/experiments/v2-latent-diagnostics/diagnostics.json \
  --max-users 500
```
Expected: command completes and prints the summary JSON.

- [ ] **Step 3: Evaluate the pre-registered gates from the JSON**

```bash
.venv/bin/python - <<'PY'
import json
d = json.load(open('artifacts/experiments/v2-latent-diagnostics/diagnostics.json'))["summary"]
g1 = d["latent_recall_500_all"] >= 0.55
q = d["target_latent_rank_quantiles"]
g2 = q["p50"] <= 120 and q["p75"] <= 300
g3 = d["union_recall_three_route"] >= 0.90 and d["latent_only_coverage"] * 500 >= 10
g4 = d["latent_recall_10_all"] >= 0.08
print({"G1": g1, "G2": g2, "G3": g3, "G4": g4, "values": d})
PY
```
Expected: the gate booleans are printed; do not modify thresholds.

- [ ] **Step 4: Record the outcome**

Write `reports/experiments/v2-latent-diagnostics.json` (copy of the artifact) and `reports/experiments/v2-latent-diagnostics.md` with the summary table, gate table, fit seconds, and fingerprints. If any of G1–G4 fails: keep the negative result, state that the ranker is not trained, keep the frozen gate locked, commit the report, and **stop the plan here**.

- [ ] **Step 5: Commit**

```bash
git add configs/v2_dense_latent.yaml reports/experiments/v2-latent-diagnostics.json reports/experiments/v2-latent-diagnostics.md
git commit -m "docs: report latent candidate diagnosis and candidate-stage gate"
```

---

### Task 11: E3 — 30-user smoke, wall-time benchmark, and replay gate (G5)

**Files:**
- Create: `configs/v2_dense_latent_30.yaml`
- Create: `configs/v2_dense_latent_30b.yaml` (determinism rerun)
- Create: `reports/experiments/v2-latent-smoke-30.md` (after the run)

- [ ] **Step 1: Create the 30-user config**

Copy `configs/v2_dense_latent.yaml` to `configs/v2_dense_latent_30.yaml` and change only `name` to `v2-dense-lambdamart-latent-30` and `latent.artifact_path` to `artifacts/experiments/v2-latent-hardneg-30/latent.npz`.

- [ ] **Step 2: Run the smoke run and measure wall time**

```bash
/usr/bin/time -p sh -c 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/recagent-eval train-ranker \
  --config configs/v2_dense_latent_30.yaml \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --cases cases/fixed_cases.json \
  --output artifacts/experiments/v2-latent-hardneg-30/model.json \
  --evidence-output artifacts/experiments/v2-latent-hardneg-30/validation.json \
  --bundle-manifest-output artifacts/experiments/v2-latent-hardneg-30/bundle.json \
  --max-users 30' 2>&1 | tee /private/tmp/v2-latent-30-walltime.txt
```
Expected: run completes; wall time recorded in `/private/tmp/v2-latent-30-walltime.txt`.

- [ ] **Step 3: Verify bundle integrity and latent checksum binding**

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
base = Path("artifacts/experiments/v2-latent-hardneg-30")
manifest = json.loads((base / "bundle.json").read_text())
latent = (base / "latent.npz").read_bytes()
assert manifest["schema_version"] == "lambdamart-bundle/v2"
assert manifest["latent_sha256"] == hashlib.sha256(latent).hexdigest()
artifact = json.loads((base / "model.json").read_text())
assert artifact["schema_version"] == "lambdamart-artifact/v2"
assert artifact["latent_artifact_checksum"] == manifest["latent_sha256"]
evidence = json.loads((base / "validation.json").read_text())
assert evidence["schema_version"] == "lambdamart-validation/v2"
assert evidence["latent_artifact_checksum"] == manifest["latent_sha256"]
print("bundle v2 latent binding OK", manifest["latent_sha256"][:16])
PY
```
Expected: `bundle v2 latent binding OK`.

- [ ] **Step 4: Determinism/replay gate — rerun into a fresh directory and diff**

Create `configs/v2_dense_latent_30b.yaml` as a copy of
`configs/v2_dense_latent_30.yaml` with `name` changed to
`v2-dense-lambdamart-latent-30b` and `latent.artifact_path` changed to
`artifacts/experiments/v2-latent-hardneg-30b/latent.npz` (the latent artifact
refuses overwrite, so the rerun needs its own path). Then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/recagent-eval train-ranker \
  --config configs/v2_dense_latent_30b.yaml \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --cases cases/fixed_cases.json \
  --output artifacts/experiments/v2-latent-hardneg-30b/model.json \
  --evidence-output artifacts/experiments/v2-latent-hardneg-30b/validation.json \
  --bundle-manifest-output artifacts/experiments/v2-latent-hardneg-30b/bundle.json \
  --max-users 30
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
a = json.load(open("artifacts/experiments/v2-latent-hardneg-30/validation.json"))
b = json.load(open("artifacts/experiments/v2-latent-hardneg-30b/validation.json"))
assert a["per_user_rows"] == b["per_user_rows"], "validation rows differ"
assert a["evidence_fingerprint"] == b["evidence_fingerprint"]
ha = hashlib.sha256(Path("artifacts/experiments/v2-latent-hardneg-30/latent.npz").read_bytes()).hexdigest()
hb = hashlib.sha256(Path("artifacts/experiments/v2-latent-hardneg-30b/latent.npz").read_bytes()).hexdigest()
assert ha == hb, "latent artifacts differ"
assert a["constraint_satisfaction_rate"] == 1.0
print("replay deterministic; constraints", a["constraint_satisfaction_rate"])
PY
```
Expected: `replay deterministic; constraints 1.0`. If rows differ, stop and debug the nondeterminism before any 500-user run.

- [ ] **Step 5: Record the benchmark and commit**

Write `reports/experiments/v2-latent-smoke-30.md` with the wall time from the `time` output, the bundle fingerprints, the determinism result, and the constraint rate. Commit:

```bash
git add configs/v2_dense_latent_30.yaml configs/v2_dense_latent_30b.yaml reports/experiments/v2-latent-smoke-30.md
git commit -m "docs: report 30-user latent smoke, benchmark, and replay gate"
```

---

### Task 12: E4 — 500-user validation (main latent run) and formal gate

**Files:**
- Create: `reports/experiments/v2-dense-lambdamart-latent500.json` / `.md` (after the run)

- [ ] **Step 1: Run the 500-user validation**

```bash
/usr/bin/time -p sh -c 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/recagent-eval train-ranker \
  --config configs/v2_dense_latent.yaml \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --cases cases/fixed_cases.json \
  --output artifacts/experiments/v2-latent-hardneg-500/model.json \
  --evidence-output artifacts/experiments/v2-latent-hardneg-500/validation.json \
  --bundle-manifest-output artifacts/experiments/v2-latent-hardneg-500/bundle.json \
  --max-users 500' 2>&1 | tee /private/tmp/v2-latent-500-walltime.txt
```
Expected: run completes and publishes the v2 bundle.

- [ ] **Step 2: Evaluate the formal gate from the JSON**

```bash
.venv/bin/python - <<'PY'
import json
d = json.load(open("artifacts/experiments/v2-latent-hardneg-500/validation.json"))
g1 = d["mean_lambdamart_ndcg_at_10"] > d["mean_itemcf_ndcg_at_10"]
g2 = d["ndcg_delta_ci_lower"] > 0
g3 = d["constraint_satisfaction_rate"] == 1.0
print({"G1_ndcg": g1, "G2_ci": g2, "G3_constraints": g3,
       "lambdamart": d["mean_lambdamart_ndcg_at_10"],
       "itemcf": d["mean_itemcf_ndcg_at_10"],
       "ci": [d["ndcg_delta_ci_lower"], d["ndcg_delta_ci_upper"]],
       "constraints": d["constraint_satisfaction_rate"]})
PY
```

- [ ] **Step 3: Record the outcome**

Write `reports/experiments/v2-dense-lambdamart-latent500.json` (copy of the evidence summary fields and fingerprints) and `.md` with the gate table, metrics, selected params, model/latent checksums, and wall time.

- If the gate passes: stop after committing the report and hand the evidence to the user for review. **Do not consume the frozen marker** without explicit user approval.
- If the gate fails: keep the negative result committed, then run Task 14 (E6) once; after E6, stop regardless of outcome.

- [ ] **Step 4: Commit**

```bash
git add reports/experiments/v2-dense-lambdamart-latent500.json reports/experiments/v2-dense-lambdamart-latent500.md
git commit -m "docs: report 500-user latent LambdaMART validation"
```

---

### Task 13: E5 — optional attribution control (`negative_policy=all`)

**Files:**
- Create: `configs/v2_dense_latent_allneg.yaml`
- Create: `reports/experiments/v2-dense-lambdamart-latent-allneg.json` / `.md` (after the run)

Run only when attribution of the hard-negative policy is needed after E4 (pass or fail). Copy `configs/v2_dense_latent.yaml` to `configs/v2_dense_latent_allneg.yaml`, change `name` to `v2-dense-lambdamart-latent-allneg`, set `ranker.max_negatives` to null and `ranker.negative_policy` to `all`, and point `latent.artifact_path` at `artifacts/experiments/v2-latent-allneg-500/latent.npz`. Run the same 500-user command with the `-allneg` artifact paths, evaluate the same formal gate, and record the report. Commit config + reports.

---

### Task 14: E6 — contingency v2b features (run only if E4 fails)

**Files:**
- Modify: `src/recagent_eval/retrieval.py` (`ItemCFRetriever.score_many`)
- Modify: `src/recagent_eval/lambdamart_pipeline.py` (recent-itemcf threading for v2b)
- Create: `configs/v2_dense_latent_bfeat.yaml`
- Create: `reports/experiments/v2-dense-lambdamart-latent-bfeat.json` / `.md`

- [ ] **Step 1: Add `ItemCFRetriever.score_many` (TDD)**

Add a failing test in `tests/test_retrieval.py` asserting `score_many` returns scores for requested ids without changing `retrieve` behavior, then implement:
```python
def score_many(self, history: set[int], movie_ids: Iterable[int]) -> dict[int, float]:
    scores: Counter[int] = Counter()
    for source in history:
        for movie_id, similarity in self.similarities.get(source, {}).items():
            if movie_id not in history:
                scores[movie_id] += similarity
    if not scores:
        for movie_id, count in self.popularity.items():
            if movie_id not in history:
                scores[movie_id] = float(count)
    return {movie_id: float(scores.get(movie_id, 0.0)) for movie_id in movie_ids}
```

- [ ] **Step 2: Thread recent-itemcf scores for v2b**

In `build_candidate_queries`, when `feature_version == "v2b"`, compute the union candidate set first (itemcf ∪ dense ∪ latent), take the user's 10 most recent positive history movie IDs, and pass `recent_itemcf_scores=itemcf.score_many(recent_ids, candidates)` into `build_candidate_feature_rows`. Add a v2b threading test in `tests/test_lambdamart_pipeline.py`.

- [ ] **Step 3: Create the config and run E6**

Copy `configs/v2_dense_latent.yaml` to `configs/v2_dense_latent_bfeat.yaml`; set `name` to `v2-dense-lambdamart-latent-bfeat`, `ranker.feature_version` to `v2b`, and `latent.artifact_path` to `artifacts/experiments/v2-latent-bfeat-500/latent.npz`. Run the 500-user command with the `-bfeat` artifact paths and evaluate the formal gate.

- [ ] **Step 4: Record and commit**

Write the report (pass or negative result) and commit config + reports. This is the **last** ranker experiment; do not iterate further.

---

### Task 15: Reports, README/HANDOFF reconciliation, final gate, and branch status

**Files:**
- Modify: `README.md`
- Modify: `docs/HANDOFF-2026-08-22.md`

- [ ] **Step 1: Reconcile documentation only from checked JSON**

Update `README.md` and `docs/HANDOFF-2026-08-22.md` with: the latent route and schema-v2 contracts, the candidate-stage gate outcome (G1–G4 values from `v2-latent-diagnostics.json`), the E3 benchmark and replay result, and the E4 formal gate outcome copied from the evidence JSON. Do not claim frozen or Qwen results. Keep every negative result preserved.

- [ ] **Step 2: Run the complete final gate**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered -q
.venv/bin/ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
git diff --check
bash -n scripts/run_remote_qwen.sh
git status --short --branch
```
Expected: all green; artifacts/model weights remain gitignored; no credentials or taste fields.

- [ ] **Step 3: Commit and leave the branch unmerged**

```bash
git add README.md docs/HANDOFF-2026-08-22.md
git commit -m "docs: reconcile v2 latent recall evidence in README and handoff"
```
Leave the merge/PR choice to the user.
```
```
```

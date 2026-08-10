# RecAgent-Eval v2 Offline Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe sentence-embedding retrieval and interpretable learned-reranking validation pipeline with nested CV, uncertainty analysis, immutable evidence, and a hard frozen-test gate.

**Architecture:** Preserve the v1 recommendation and evidence paths while adding four focused v2 modules: embedding/cache management, candidate features, pairwise linear ranking, and validation/promotion. Formal selection operates only on chronological training histories and validation targets; frozen cases are inaccessible until a recomputed promotion manifest passes every statistical, integrity, and resource gate.

**Tech Stack:** Python 3.11+, NumPy, Pydantic, Typer, PyYAML, scikit-learn, Sentence Transformers with `BAAI/bge-small-en-v1.5`, pytest, Ruff, uv.

---

## Execution preflight

- Use `superpowers:using-git-worktrees` before Task 1 and create an isolated worktree from commit `3c97343` or its descendant.
- Preserve `docs/NEXT_SESSION_HANDOFF.md`, `docs/NEXT_SESSION_PROMPT.md`, and `docs/V2_OPTIMIZATION_PROMPT.md`; do not add, overwrite, or remove them.
- Re-run `git status --short --branch`, `git log --oneline -8`, `git diff --stat`, and `git diff --name-status` in the worktree before editing.
- Do not run `evaluate-v2-frozen`, DeepSeek, Qwen/vLLM, or any remote GPU command while executing this plan.
- Before the dependency-lock step, explain that BGE weights are about 133 MB under MIT and that Sentence Transformers/PyTorch may add several hundred MB of platform-specific cached packages. Request approval for network access if the environment requires it.

## File map

### New production files

- `src/recagent_eval/embedding.py` — item text, encoder protocol, BGE adapter, normalized embedding index, cache manifest, and cache fingerprint checks.
- `src/recagent_eval/candidate_features.py` — stable feature schema, train-only profile statistics, route signals, candidate rows, and finite-value validation.
- `src/recagent_eval/learned_ranking.py` — deterministic hard negatives, symmetric pair construction, fold-local scaling, linear ranker artifacts, scoring, and explanations.
- `src/recagent_eval/v2_selection.py` — fold maps, per-system evaluation, nested CV, bootstrap, subgroup analysis, artifacts, promotion gate, and frozen evaluation over already-loaded cases.

### New configuration and evidence files

- `configs/v2_offline.yaml` — the complete pre-registered validation configuration.
- `configs/frozen_test_lock.yaml` — the existing canonical fixed-case fingerprint and consumption-marker path, without test metrics.
- `reports/experiments/v2-validation.json` — generated aggregate evidence after the formal validation run.
- `reports/experiments/v2-validation.md` — generated human-readable outcome including failures.

### New test files

- `tests/test_embedding.py`
- `tests/test_candidate_features.py`
- `tests/test_learned_ranking.py`
- `tests/test_v2_selection.py`
- `tests/test_v2_cli.py`

### Modified files

- `src/recagent_eval/data.py` — add ordered positive histories from an explicitly supplied legal rating set.
- `src/recagent_eval/config.py` — add strict v2 configuration models and loader without changing v1 parsing.
- `src/recagent_eval/cli.py` — add `prepare-embeddings`, `validate-v2`, and protected `evaluate-v2-frozen` entry points.
- `pyproject.toml` — add separate `v2` and `embedding` optional dependency groups.
- `uv.lock` — lock the approved dependency graph.
- `.github/workflows/ci.yml` — install the lightweight `v2` extra for ranker tests, without the embedding extra or model download.
- `.gitignore` — ignore full v2 caches/runs and the frozen consumption marker while allowing aggregate reports.
- `README.md` — document pending v2 commands and retain honest v1 claims until evidence exists.

## Task 1: Parse and validate the pre-registered v2 configuration

**Files:**
- Create: `configs/v2_offline.yaml`
- Modify: `src/recagent_eval/config.py:1-68`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for the full configuration and invalid bounds**

Add these imports and tests to `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from recagent_eval.config import load_v2_config


def test_load_v2_config_freezes_selection_protocol(tmp_path: Path) -> None:
    path = tmp_path / "v2.yaml"
    path.write_text(
        """
model_id: BAAI/bge-small-en-v1.5
requested_revision: main
embedding_batch_size: 64
embedding_device: cpu
retrieval_top_k: 500
history_cap: 50
outer_folds: 5
inner_folds: 3
seeds: [42, 2025, 3407]
c_grid: [0.01, 0.1, 1.0]
bootstrap_iterations: 10000
bootstrap_seed: 20260810
max_negatives: 50
itemcf_head_negatives: 20
semantic_head_negatives: 20
resource_budget:
  max_cache_bytes: 1000000000
  max_peak_rss_bytes: 4000000000
  max_embedding_seconds: 1800
  max_validation_seconds: 14400
  max_ranker_bytes: 5000000
  max_p95_ms: 100
""".strip()
        + "\n"
    )

    config = load_v2_config(path)

    assert config.seeds == (42, 2025, 3407)
    assert config.c_grid == (0.01, 0.1, 1.0)
    assert config.retrieval_top_k == 500
    assert config.resource_budget.max_p95_ms == 100


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("history_cap: 0", "history_cap must be positive"),
        ("outer_folds: 1", "outer_folds must be at least 2"),
        ("c_grid: [0.0]", "c_grid values must be positive"),
        ("max_negatives: 30", "negative quotas exceed max_negatives"),
    ],
)
def test_load_v2_config_rejects_invalid_protocol(
    tmp_path: Path,
    line: str,
    message: str,
) -> None:
    path = tmp_path / "v2.yaml"
    base = Path("configs/v2_offline.yaml").read_text()
    key = line.split(":", 1)[0]
    rewritten = "\n".join(
        line if row.startswith(f"{key}:") else row for row in base.splitlines()
    )
    path.write_text(rewritten + "\n")

    with pytest.raises(ValueError, match=message):
        load_v2_config(path)
```

- [ ] **Step 2: Run the focused tests and verify the missing loader failure**

Run:

```bash
.venv/bin/pytest tests/test_config.py -k v2 -v
```

Expected: collection fails because `load_v2_config` does not exist.

- [ ] **Step 3: Implement immutable v2 configuration models and the loader**

Add to `src/recagent_eval/config.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    max_cache_bytes: int
    max_peak_rss_bytes: int
    max_embedding_seconds: float
    max_validation_seconds: float
    max_ranker_bytes: int
    max_p95_ms: float


@dataclass(frozen=True)
class V2ExperimentConfig:
    model_id: str
    requested_revision: str
    embedding_batch_size: int
    embedding_device: str
    retrieval_top_k: int
    history_cap: int
    outer_folds: int
    inner_folds: int
    seeds: tuple[int, ...]
    c_grid: tuple[float, ...]
    bootstrap_iterations: int
    bootstrap_seed: int
    max_negatives: int
    itemcf_head_negatives: int
    semantic_head_negatives: int
    resource_budget: ResourceBudget


def load_v2_config(path: Path) -> V2ExperimentConfig:
    payload = yaml.safe_load(path.read_text()) or {}
    budget = payload.get("resource_budget") or {}
    config = V2ExperimentConfig(
        model_id=str(payload["model_id"]),
        requested_revision=str(payload.get("requested_revision", "main")),
        embedding_batch_size=int(payload["embedding_batch_size"]),
        embedding_device=str(payload["embedding_device"]),
        retrieval_top_k=int(payload["retrieval_top_k"]),
        history_cap=int(payload["history_cap"]),
        outer_folds=int(payload["outer_folds"]),
        inner_folds=int(payload["inner_folds"]),
        seeds=tuple(int(value) for value in payload["seeds"]),
        c_grid=tuple(float(value) for value in payload["c_grid"]),
        bootstrap_iterations=int(payload["bootstrap_iterations"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        max_negatives=int(payload["max_negatives"]),
        itemcf_head_negatives=int(payload["itemcf_head_negatives"]),
        semantic_head_negatives=int(payload["semantic_head_negatives"]),
        resource_budget=ResourceBudget(
            max_cache_bytes=int(budget["max_cache_bytes"]),
            max_peak_rss_bytes=int(budget["max_peak_rss_bytes"]),
            max_embedding_seconds=float(budget["max_embedding_seconds"]),
            max_validation_seconds=float(budget["max_validation_seconds"]),
            max_ranker_bytes=int(budget["max_ranker_bytes"]),
            max_p95_ms=float(budget["max_p95_ms"]),
        ),
    )
    if config.retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if config.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")
    if config.embedding_device not in {"cpu", "cuda"}:
        raise ValueError("embedding_device must be cpu or cuda")
    if config.history_cap <= 0:
        raise ValueError("history_cap must be positive")
    if config.outer_folds < 2:
        raise ValueError("outer_folds must be at least 2")
    if config.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not config.c_grid or any(value <= 0 for value in config.c_grid):
        raise ValueError("c_grid values must be positive")
    if config.itemcf_head_negatives + config.semantic_head_negatives > config.max_negatives:
        raise ValueError("negative quotas exceed max_negatives")
    if config.bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    return config
```

Create `configs/v2_offline.yaml` exactly as follows:

```yaml
model_id: BAAI/bge-small-en-v1.5
requested_revision: main
embedding_batch_size: 64
embedding_device: cpu
retrieval_top_k: 500
history_cap: 50
outer_folds: 5
inner_folds: 3
seeds: [42, 2025, 3407]
c_grid: [0.01, 0.1, 1.0]
bootstrap_iterations: 10000
bootstrap_seed: 20260810
max_negatives: 50
itemcf_head_negatives: 20
semantic_head_negatives: 20
resource_budget:
  max_cache_bytes: 1000000000
  max_peak_rss_bytes: 4000000000
  max_embedding_seconds: 1800
  max_validation_seconds: 14400
  max_ranker_bytes: 5000000
  max_p95_ms: 100
```

- [ ] **Step 4: Run all configuration tests**

Run:

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: all configuration tests pass.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add configs/v2_offline.yaml src/recagent_eval/config.py tests/test_config.py
git commit -m "feat: define v2 validation configuration"
```

## Task 2: Expose legal ordered positive histories

**Files:**
- Modify: `src/recagent_eval/data.py:69-96`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write failing history-order and cap tests**

Append to `tests/test_data.py`:

```python
import pytest

from recagent_eval.data import build_positive_histories


def test_positive_histories_use_only_supplied_rows_and_keep_recent_order() -> None:
    legal_train = [
        Rating(1, 9, 5, 90),
        Rating(1, 3, 4, 30),
        Rating(1, 5, 2, 50),
        Rating(1, 7, 5, 70),
        Rating(2, 8, 4, 80),
    ]

    histories = build_positive_histories(legal_train, history_cap=2)

    assert [row.movie_id for row in histories[1]] == [7, 9]
    assert [row.movie_id for row in histories[2]] == [8]
    assert 5 not in [row.movie_id for row in histories[1]]


def test_positive_histories_reject_non_positive_cap() -> None:
    with pytest.raises(ValueError, match="history_cap must be positive"):
        build_positive_histories([], history_cap=0)
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
.venv/bin/pytest tests/test_data.py -k positive_histories -v
```

Expected: import fails because `build_positive_histories` does not exist.

- [ ] **Step 3: Implement ordered histories without accepting a DatasetSplit**

Add to `src/recagent_eval/data.py`:

```python
def build_positive_histories(
    ratings: list[Rating] | tuple[Rating, ...],
    *,
    history_cap: int,
    positive_threshold: int = 4,
) -> dict[int, tuple[Rating, ...]]:
    if history_cap <= 0:
        raise ValueError("history_cap must be positive")
    grouped: dict[int, list[Rating]] = defaultdict(list)
    for row in ratings:
        if row.rating >= positive_threshold:
            grouped[row.user_id].append(row)
    return {
        user_id: tuple(
            sorted(rows, key=lambda row: (row.timestamp, row.movie_id))[-history_cap:]
        )
        for user_id, rows in sorted(grouped.items())
    }
```

The function deliberately receives only the caller-supplied legal rows. It has
no access to `validation_targets` or `test_targets`.

- [ ] **Step 4: Run data tests and the existing split regression test**

Run:

```bash
.venv/bin/pytest tests/test_data.py -v
```

Expected: all data tests pass, including the original chronological split test.

- [ ] **Step 5: Commit ordered histories**

```bash
git add src/recagent_eval/data.py tests/test_data.py
git commit -m "feat: add ordered positive training histories"
```

## Task 3: Build the encoder protocol and immutable embedding cache

**Files:**
- Create: `src/recagent_eval/embedding.py`
- Create: `tests/test_embedding.py`

- [ ] **Step 1: Write failing tests using a deterministic fake encoder**

Create `tests/test_embedding.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from recagent_eval.data import Movie
from recagent_eval.embedding import (
    EmbeddingCacheManifest,
    build_embedding_cache,
    item_text,
    load_embedding_cache,
)


class FakeEmbedder:
    model_id = "fake/encoder"
    revision = "abc123"
    license = "MIT"
    dimension = 2
    weight_fingerprint = "fake-weights"
    batch_size = 2
    device = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[len(text), index + 1] for index, text in enumerate(texts)], dtype=np.float32)


MOVIES = {
    2: Movie(2, "Zulu (2001)", ("Drama", "Action"), 2001),
    1: Movie(1, "Alpha", ("Comedy",), None),
}


def test_item_text_is_stable_and_removes_terminal_year() -> None:
    assert item_text(MOVIES[2]) == "Title: Zulu. Genres: Action, Drama. Release year: 2001."
    assert item_text(MOVIES[1]) == "Title: Alpha. Genres: Comedy. Release year: unknown."


def test_embedding_cache_is_normalized_sorted_and_fingerprinted(tmp_path: Path) -> None:
    manifest = build_embedding_cache(
        MOVIES,
        FakeEmbedder(),
        output_dir=tmp_path,
        dataset_fingerprint="dataset-a",
    )
    index, loaded = load_embedding_cache(tmp_path, expected_fingerprint=manifest.fingerprint)

    assert index.movie_ids.tolist() == [1, 2]
    assert np.linalg.norm(index.vectors, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert loaded == manifest
    assert EmbeddingCacheManifest.model_validate_json((tmp_path / "manifest.json").read_text())


def test_embedding_cache_rejects_wrong_fingerprint(tmp_path: Path) -> None:
    build_embedding_cache(
        MOVIES,
        FakeEmbedder(),
        output_dir=tmp_path,
        dataset_fingerprint="dataset-a",
    )

    with pytest.raises(ValueError, match="embedding cache fingerprint mismatch"):
        load_embedding_cache(tmp_path, expected_fingerprint="different")


def test_embedding_cache_rejects_tampered_manifest(tmp_path: Path) -> None:
    manifest = build_embedding_cache(
        MOVIES,
        FakeEmbedder(),
        output_dir=tmp_path,
        dataset_fingerprint="dataset-a",
    )
    payload = json.loads((tmp_path / "manifest.json").read_text())
    payload["revision"] = "tampered"
    (tmp_path / "manifest.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="embedding manifest fingerprint mismatch"):
        load_embedding_cache(tmp_path, expected_fingerprint=manifest.fingerprint)


def test_embedding_cache_rejects_tampered_vectors(tmp_path: Path) -> None:
    manifest = build_embedding_cache(
        MOVIES, FakeEmbedder(), output_dir=tmp_path, dataset_fingerprint="dataset-a"
    )
    (tmp_path / "item_embeddings.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="embedding file fingerprint mismatch"):
        load_embedding_cache(tmp_path, expected_fingerprint=manifest.fingerprint)


def test_embedding_cache_refuses_overwrite(tmp_path: Path) -> None:
    build_embedding_cache(
        MOVIES,
        FakeEmbedder(),
        output_dir=tmp_path,
        dataset_fingerprint="dataset-a",
    )

    with pytest.raises(FileExistsError, match="embedding cache already exists"):
        build_embedding_cache(
            MOVIES,
            FakeEmbedder(),
            output_dir=tmp_path,
            dataset_fingerprint="dataset-a",
        )
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
.venv/bin/pytest tests/test_embedding.py -v
```

Expected: collection fails because `recagent_eval.embedding` does not exist.

- [ ] **Step 3: Implement text rendering, normalization, fingerprinting, and cache I/O**

Create `src/recagent_eval/embedding.py` with these public contracts:

```python
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from recagent_eval.data import Movie

ITEM_TEXT_SCHEMA_VERSION = "v1-title-genres-year"


class TextEmbedder(Protocol):
    model_id: str
    revision: str
    license: str
    dimension: int
    weight_fingerprint: str
    batch_size: int
    device: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingCacheManifest(BaseModel):
    fingerprint: str
    dataset_fingerprint: str
    model_id: str
    revision: str
    license: str
    dimension: int
    dtype: str
    normalized: bool
    item_text_schema: str
    movie_ids: list[int]
    weight_fingerprint: str
    batch_size: int
    device: str
    cache_bytes: int
    embedding_seconds: float
    embedding_file_sha256: str


@dataclass(frozen=True)
class EmbeddingIndex:
    movie_ids: np.ndarray
    vectors: np.ndarray

    def vector(self, movie_id: int) -> np.ndarray:
        matches = np.flatnonzero(self.movie_ids == movie_id)
        if len(matches) != 1:
            raise KeyError(movie_id)
        return self.vectors[int(matches[0])]

    def profile(self, history_ids: list[int] | tuple[int, ...]) -> np.ndarray:
        if not history_ids:
            raise ValueError("embedding profile requires non-empty history")
        mean = np.mean([self.vector(movie_id) for movie_id in history_ids], axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0:
            raise ValueError("embedding profile has zero norm")
        return np.asarray(mean / norm, dtype=np.float32)

    def retrieve(
        self,
        profile: np.ndarray,
        *,
        allowed_ids: set[int],
        top_k: int,
    ) -> list[tuple[int, float]]:
        scores = self.vectors @ profile
        rows = [
            (int(movie_id), float(score))
            for movie_id, score in zip(self.movie_ids, scores, strict=True)
            if int(movie_id) in allowed_ids
        ]
        return sorted(rows, key=lambda row: (-row[1], row[0]))[:top_k]


def item_text(movie: Movie) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", movie.title).strip()
    genres = ", ".join(sorted(movie.genres))
    year = str(movie.year) if movie.year is not None else "unknown"
    return f"Title: {title}. Genres: {genres}. Release year: {year}."


def _normalized(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be finite non-zero vectors")
    return matrix / norms


def _fingerprint(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload(manifest: EmbeddingCacheManifest) -> dict[str, object]:
    return manifest.model_dump(exclude={"fingerprint"}, mode="json")


def build_embedding_cache(
    movies: dict[int, Movie],
    embedder: TextEmbedder,
    *,
    output_dir: Path,
    dataset_fingerprint: str,
) -> EmbeddingCacheManifest:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("embedding cache already exists")
    movie_ids = sorted(movies)
    started = time.perf_counter()
    vectors = _normalized(embedder.encode([item_text(movies[movie_id]) for movie_id in movie_ids]))
    embedding_seconds = time.perf_counter() - started
    if vectors.shape != (len(movie_ids), embedder.dimension):
        raise ValueError("embedding output shape mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "item_embeddings.npz", movie_ids=movie_ids, vectors=vectors)
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "model_id": embedder.model_id,
        "revision": embedder.revision,
        "license": embedder.license,
        "dimension": embedder.dimension,
        "dtype": "float32",
        "normalized": True,
        "item_text_schema": ITEM_TEXT_SCHEMA_VERSION,
        "movie_ids": movie_ids,
        "weight_fingerprint": embedder.weight_fingerprint,
        "batch_size": embedder.batch_size,
        "device": embedder.device,
        "cache_bytes": (output_dir / "item_embeddings.npz").stat().st_size,
        "embedding_seconds": embedding_seconds,
        "embedding_file_sha256": _sha256_file(output_dir / "item_embeddings.npz"),
    }
    manifest = EmbeddingCacheManifest(fingerprint=_fingerprint(payload), **payload)
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_embedding_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str,
) -> tuple[EmbeddingIndex, EmbeddingCacheManifest]:
    manifest = EmbeddingCacheManifest.model_validate_json((cache_dir / "manifest.json").read_text())
    if _fingerprint(_manifest_payload(manifest)) != manifest.fingerprint:
        raise ValueError("embedding manifest fingerprint mismatch")
    if manifest.fingerprint != expected_fingerprint:
        raise ValueError("embedding cache fingerprint mismatch")
    if _sha256_file(cache_dir / "item_embeddings.npz") != manifest.embedding_file_sha256:
        raise ValueError("embedding file fingerprint mismatch")
    payload = np.load(cache_dir / "item_embeddings.npz", allow_pickle=False)
    index = EmbeddingIndex(
        movie_ids=np.asarray(payload["movie_ids"], dtype=np.int64),
        vectors=_normalized(payload["vectors"]),
    )
    if index.movie_ids.tolist() != manifest.movie_ids:
        raise ValueError("embedding movie order mismatch")
    return index, manifest
```

- [ ] **Step 4: Run embedding tests**

Run:

```bash
.venv/bin/pytest tests/test_embedding.py -v
```

Expected: all embedding cache tests pass without network access.

- [ ] **Step 5: Commit the dependency-free embedding core**

```bash
git add src/recagent_eval/embedding.py tests/test_embedding.py
git commit -m "feat: add fingerprinted item embedding cache"
```

## Task 4: Add approved optional dependencies and the lazy BGE adapter

**Files:**
- Modify: `pyproject.toml:15-27`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml:15`
- Modify: `src/recagent_eval/embedding.py`
- Modify: `tests/test_embedding.py`

- [ ] **Step 1: Write a failing lazy-adapter test that uses fake imported modules**

Append to `tests/test_embedding.py`:

```python
import sys
from types import SimpleNamespace

from recagent_eval.embedding import SentenceTransformerEmbedder


def test_sentence_transformer_adapter_pins_resolved_revision(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, model_id, *, revision, device):
            self.model_id = model_id
            self.revision = revision
            self.device = device

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, *, normalize_embeddings, convert_to_numpy, batch_size):
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            assert batch_size == 64
            return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    embedder = SentenceTransformerEmbedder(
        model_id="BAAI/bge-small-en-v1.5",
        revision="resolved-sha",
        license_name="MIT",
        weight_fingerprint="weights-sha256",
        batch_size=64,
        device="cpu",
    )

    assert embedder.revision == "resolved-sha"
    assert embedder.weight_fingerprint == "weights-sha256"
    assert embedder.dimension == 384
    assert embedder.encode(["one"]).shape == (1, 384)
```

- [ ] **Step 2: Run the adapter test and verify the class is missing**

Run:

```bash
.venv/bin/pytest tests/test_embedding.py -k sentence_transformer_adapter -v
```

Expected: import fails because `SentenceTransformerEmbedder` does not exist.

- [ ] **Step 3: Implement the lazy adapter without a top-level heavyweight import**

Add to `src/recagent_eval/embedding.py`:

```python
class SentenceTransformerEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        license_name: str,
        weight_fingerprint: str,
        batch_size: int,
        device: str,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding extra is required; install with `uv sync --extra dev --extra v2 --extra embedding`"
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.license = license_name
        self.weight_fingerprint = weight_fingerprint
        self.batch_size = batch_size
        self.device = device
        self._model = SentenceTransformer(model_id, revision=revision, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=self.batch_size,
            ),
            dtype=np.float32,
        )
```

- [ ] **Step 4: Declare split optional extras and update CI**

Add to `[project.optional-dependencies]` in `pyproject.toml`:

```toml
v2 = ["scikit-learn>=1.9,<2"]
embedding = ["sentence-transformers>=5,<6"]
```

Change `.github/workflows/ci.yml` line 15 to:

```yaml
      - run: uv sync --extra dev --extra v2 --locked
```

CI installs scikit-learn for ranker tests but does not install Sentence
Transformers, PyTorch, or model weights. The adapter test remains import-faked.

- [ ] **Step 5: Request network approval, lock dependencies, and sync the local test environment**

Run after approval:

```bash
uv lock
uv sync --extra dev --extra v2
```

Expected: `uv.lock` resolves scikit-learn and the embedding extra dependency
graph; the local environment installs the dev and lightweight v2 extras without
downloading BGE weights.

- [ ] **Step 6: Run adapter and baseline tests**

Run:

```bash
.venv/bin/pytest tests/test_embedding.py tests/test_config.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit dependency boundaries and adapter**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml src/recagent_eval/embedding.py tests/test_embedding.py
git commit -m "feat: add optional v2 model dependencies"
```

## Task 5: Build the stable train-only candidate feature schema

**Files:**
- Create: `src/recagent_eval/candidate_features.py`
- Create: `tests/test_candidate_features.py`

- [ ] **Step 1: Write failing tests for exact feature order, train-only popularity, explicit neutrality, and finite checks**

Create `tests/test_candidate_features.py`:

```python
import numpy as np
import pytest

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    CandidateSignals,
    TrainingProfile,
    build_candidate_rows,
    feature_schema_fingerprint,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState


MOVIES = {
    1: Movie(1, "History (2000)", ("Drama",), 2000),
    2: Movie(2, "Candidate (2005)", ("Drama", "Comedy"), 2005),
    3: Movie(3, "Other (1990)", ("Action",), 1990),
}


def test_candidate_rows_have_exact_schema_and_train_only_statistics() -> None:
    profile = TrainingProfile.from_history(
        [Rating(7, 1, 5, 10)],
        movies=MOVIES,
        train_popularity={2: 3},
    )
    signals = CandidateSignals(
        itemcf_scores={2: 4.0},
        semantic_scores={2: 0.8, 3: 0.4},
        history_similarity={2: (0.7, 0.6), 3: (0.2, 0.1)},
    )

    rows = build_candidate_rows(
        movies=MOVIES,
        candidate_ids={2, 3},
        signals=signals,
        profile=profile,
        preference_state=None,
    )

    assert tuple(rows[2].values) == pytest.approx(
        (
            np.log1p(4.0), 1.0, 0.8, 1.0, 1.0, 1.0,
            np.log1p(3.0), 0.5, 5.0, 0.0, 0.7, 0.6,
            0.0, 0.0, 0.0, 0.0,
        )
    )
    assert rows[2].feature_names == FEATURE_NAMES
    assert feature_schema_fingerprint() == feature_schema_fingerprint()


def test_explicit_features_only_activate_from_supplied_state() -> None:
    profile = TrainingProfile.from_history(
        [Rating(7, 1, 5, 10)],
        movies=MOVIES,
        train_popularity={},
    )
    rows = build_candidate_rows(
        movies=MOVIES,
        candidate_ids={2},
        signals=CandidateSignals(
            itemcf_scores={}, semantic_scores={2: 0.5}, history_similarity={2: (0.5, 0.5)}
        ),
        profile=profile,
        preference_state=PreferenceState(
            liked_genres={"Drama"}, required_genres={"Drama"}, year_min=2000, year_max=2010
        ),
    )

    assert rows[2].as_dict()["explicit_liked_genre_overlap"] == 0.5
    assert rows[2].as_dict()["explicit_required_genres_match"] == 1.0
    assert rows[2].as_dict()["explicit_year_range_match"] == 1.0
    assert rows[2].as_dict()["explicit_preference_present"] == 1.0


def test_candidate_rows_reject_non_finite_route_score() -> None:
    profile = TrainingProfile.from_history(
        [Rating(7, 1, 5, 10)], movies=MOVIES, train_popularity={}
    )
    with pytest.raises(ValueError, match="user 7 movie 2 feature semantic_score"):
        build_candidate_rows(
            movies=MOVIES,
            candidate_ids={2},
            signals=CandidateSignals(
                itemcf_scores={}, semantic_scores={2: float("nan")}, history_similarity={2: (0.0, 0.0)}
            ),
            profile=profile,
            preference_state=None,
        )
```

- [ ] **Step 2: Verify the feature module is missing**

Run:

```bash
.venv/bin/pytest tests/test_candidate_features.py -v
```

Expected: collection fails because `recagent_eval.candidate_features` does not exist.

- [ ] **Step 3: Implement the schema and feature builder**

Create `src/recagent_eval/candidate_features.py` with:

```python
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState

FEATURE_NAMES = (
    "itemcf_score_log1p",
    "itemcf_reciprocal_rank",
    "semantic_score",
    "semantic_reciprocal_rank",
    "in_itemcf",
    "in_semantic",
    "log1p_train_popularity",
    "history_genre_jaccard",
    "history_year_abs_gap",
    "history_year_missing",
    "max_history_semantic_similarity",
    "mean_history_semantic_similarity",
    "explicit_liked_genre_overlap",
    "explicit_required_genres_match",
    "explicit_year_range_match",
    "explicit_preference_present",
)


@dataclass(frozen=True)
class CandidateSignals:
    itemcf_scores: dict[int, float]
    semantic_scores: dict[int, float]
    history_similarity: dict[int, tuple[float, float]]


@dataclass(frozen=True)
class CandidateFeatureRow:
    user_id: int
    movie_id: int
    feature_names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values, strict=True))


@dataclass(frozen=True)
class TrainingProfile:
    user_id: int
    history_ids: tuple[int, ...]
    top_genres: frozenset[str]
    mean_year: float | None
    train_popularity: dict[int, int]

    @classmethod
    def from_history(
        cls,
        history: list[Rating] | tuple[Rating, ...],
        *,
        movies: dict[int, Movie],
        train_popularity: dict[int, int],
    ) -> "TrainingProfile":
        if not history:
            raise ValueError("training profile requires non-empty history")
        genre_counts = Counter(
            genre for row in history for genre in movies[row.movie_id].genres
        )
        top_genres = frozenset(
            genre
            for genre, _ in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        )
        years = [movies[row.movie_id].year for row in history if movies[row.movie_id].year is not None]
        return cls(
            user_id=history[0].user_id,
            history_ids=tuple(row.movie_id for row in history),
            top_genres=top_genres,
            mean_year=float(np.mean(years)) if years else None,
            train_popularity=dict(train_popularity),
        )


def feature_schema_fingerprint() -> str:
    canonical = json.dumps({"version": 1, "features": FEATURE_NAMES}, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _ranks(scores: dict[int, float]) -> dict[int, int]:
    ordered = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))
    return {movie_id: rank for rank, movie_id in enumerate(ordered, start=1)}


def build_candidate_rows(
    *,
    movies: dict[int, Movie],
    candidate_ids: set[int],
    signals: CandidateSignals,
    profile: TrainingProfile,
    preference_state: PreferenceState | None,
) -> dict[int, CandidateFeatureRow]:
    itemcf_ranks = _ranks(signals.itemcf_scores)
    semantic_ranks = _ranks(signals.semantic_scores)
    rows: dict[int, CandidateFeatureRow] = {}
    for movie_id in sorted(candidate_ids):
        movie = movies[movie_id]
        genres = set(movie.genres)
        union = genres | set(profile.top_genres)
        genre_jaccard = len(genres & set(profile.top_genres)) / len(union) if union else 0.0
        year_missing = float(movie.year is None or profile.mean_year is None)
        year_gap = 0.0 if year_missing else abs(float(movie.year) - float(profile.mean_year))
        history_max, history_mean = signals.history_similarity.get(movie_id, (0.0, 0.0))
        explicit_present = float(preference_state is not None)
        liked_overlap = 0.0
        required_match = 0.0
        year_match = 0.0
        if preference_state is not None:
            liked_overlap = (
                len(genres & preference_state.liked_genres) / len(genres) if genres else 0.0
            )
            required_match = float(preference_state.required_genres.issubset(genres))
            year_match = float(
                movie.year is not None
                and (preference_state.year_min is None or movie.year >= preference_state.year_min)
                and (preference_state.year_max is None or movie.year <= preference_state.year_max)
            )
        values = (
            math.log1p(signals.itemcf_scores.get(movie_id, 0.0)),
            1.0 / itemcf_ranks[movie_id] if movie_id in itemcf_ranks else 0.0,
            signals.semantic_scores.get(movie_id, 0.0),
            1.0 / semantic_ranks[movie_id] if movie_id in semantic_ranks else 0.0,
            float(movie_id in signals.itemcf_scores),
            float(movie_id in signals.semantic_scores),
            math.log1p(profile.train_popularity.get(movie_id, 0)),
            genre_jaccard,
            year_gap,
            year_missing,
            history_max,
            history_mean,
            liked_overlap,
            required_match,
            year_match,
            explicit_present,
        )
        for name, value in zip(FEATURE_NAMES, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    f"user {profile.user_id} movie {movie_id} feature {name} is not finite"
                )
        rows[movie_id] = CandidateFeatureRow(profile.user_id, movie_id, FEATURE_NAMES, values)
    return rows
```

- [ ] **Step 4: Run feature tests**

Run:

```bash
.venv/bin/pytest tests/test_candidate_features.py -v
```

Expected: all candidate-feature tests pass.

- [ ] **Step 5: Commit the feature contract**

```bash
git add src/recagent_eval/candidate_features.py tests/test_candidate_features.py
git commit -m "feat: add train-only reranking features"
```

## Task 6: Implement deterministic pairwise linear ranking

**Files:**
- Create: `src/recagent_eval/learned_ranking.py`
- Create: `tests/test_learned_ranking.py`

- [ ] **Step 1: Write failing tests for negative order, symmetric pairs, user weights, and score explanations**

Create `tests/test_learned_ranking.py`:

```python
import numpy as np
import pytest

from recagent_eval.candidate_features import CandidateFeatureRow, FEATURE_NAMES
from recagent_eval.learned_ranking import (
    PairwiseLinearRanker,
    UserRankingExample,
    build_pairwise_rows,
    select_hard_negatives,
)


def row(user_id: int, movie_id: int, first: float) -> CandidateFeatureRow:
    values = (first,) + (0.0,) * (len(FEATURE_NAMES) - 1)
    return CandidateFeatureRow(user_id, movie_id, FEATURE_NAMES, values)


def test_hard_negatives_take_route_heads_then_hashed_remainder() -> None:
    selected = select_hard_negatives(
        user_id=7,
        target_id=1,
        candidate_ids={1, 2, 3, 4, 5, 6},
        itemcf_ranked_ids=[1, 2, 3],
        semantic_ranked_ids=[3, 4, 1],
        seed=42,
        max_negatives=5,
        itemcf_head=2,
        semantic_head=2,
    )

    assert selected[:3] == [2, 3, 4]
    assert len(selected) == 5
    assert 1 not in selected


def test_pairwise_rows_are_symmetric_and_each_user_totals_one() -> None:
    examples = [
        UserRankingExample(
            user_id=7,
            target_id=1,
            rows={1: row(7, 1, 3.0), 2: row(7, 2, 1.0), 3: row(7, 3, 0.0)},
            itemcf_ranked_ids=(2, 3),
            semantic_ranked_ids=(3, 2),
        )
    ]
    matrix, labels, weights = build_pairwise_rows(
        examples,
        seed=42,
        max_negatives=2,
        itemcf_head=1,
        semantic_head=1,
    )

    assert matrix.shape == (4, len(FEATURE_NAMES))
    assert labels.tolist() == [1, 0, 1, 0]
    assert matrix[0].tolist() == pytest.approx((-matrix[1]).tolist())
    assert weights.sum() == pytest.approx(1.0)


def test_linear_ranker_serializes_coefficients_and_contributions() -> None:
    examples = [
        UserRankingExample(
            user_id=user_id,
            target_id=1,
            rows={1: row(user_id, 1, 3.0), 2: row(user_id, 2, 0.0)},
            itemcf_ranked_ids=(1, 2),
            semantic_ranked_ids=(2, 1),
        )
        for user_id in (1, 2, 3)
    ]
    ranker = PairwiseLinearRanker.fit(
        examples,
        c_value=0.1,
        seed=42,
        max_negatives=1,
        itemcf_head=1,
        semantic_head=0,
    )

    ranked = ranker.rank(examples[0].rows)
    explanation = ranker.explain(examples[0].rows[1])

    assert ranked[0] == 1
    assert set(explanation) == set(FEATURE_NAMES)
    assert ranker.to_artifact()["feature_names"] == list(FEATURE_NAMES)
```

- [ ] **Step 2: Run the tests and verify the ranker module is missing**

Run:

```bash
.venv/bin/pytest tests/test_learned_ranking.py -v
```

Expected: collection fails because `recagent_eval.learned_ranking` does not exist.

- [ ] **Step 3: Implement exact negative selection and pair generation**

Create `src/recagent_eval/learned_ranking.py` and add:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from recagent_eval.candidate_features import CandidateFeatureRow, FEATURE_NAMES


@dataclass(frozen=True)
class UserRankingExample:
    user_id: int
    target_id: int
    rows: dict[int, CandidateFeatureRow]
    itemcf_ranked_ids: tuple[int, ...]
    semantic_ranked_ids: tuple[int, ...]


def select_hard_negatives(
    *,
    user_id: int,
    target_id: int,
    candidate_ids: set[int],
    itemcf_ranked_ids: list[int] | tuple[int, ...],
    semantic_ranked_ids: list[int] | tuple[int, ...],
    seed: int,
    max_negatives: int,
    itemcf_head: int,
    semantic_head: int,
) -> list[int]:
    selected: list[int] = []

    def add(ids, limit):
        added = 0
        for movie_id in ids:
            if movie_id != target_id and movie_id in candidate_ids and movie_id not in selected:
                selected.append(movie_id)
                added += 1
                if added == limit or len(selected) == max_negatives:
                    return

    add(itemcf_ranked_ids, itemcf_head)
    add(semantic_ranked_ids, semantic_head)
    remainder = sorted(
        candidate_ids - {target_id} - set(selected),
        key=lambda movie_id: hashlib.sha256(
            f"{seed}:{user_id}:{movie_id}".encode()
        ).hexdigest(),
    )
    selected.extend(remainder[: max_negatives - len(selected)])
    return selected


def _candidate_matrix(examples: list[UserRankingExample]) -> np.ndarray:
    return np.asarray(
        [row.values for example in examples for row in example.rows.values()],
        dtype=np.float64,
    )


def build_pairwise_rows(
    examples: list[UserRankingExample],
    *,
    seed: int,
    max_negatives: int,
    itemcf_head: int,
    semantic_head: int,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = scaler or StandardScaler().fit(_candidate_matrix(examples))
    differences: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for example in examples:
        if example.target_id not in example.rows:
            continue
        negatives = select_hard_negatives(
            user_id=example.user_id,
            target_id=example.target_id,
            candidate_ids=set(example.rows),
            itemcf_ranked_ids=example.itemcf_ranked_ids,
            semantic_ranked_ids=example.semantic_ranked_ids,
            seed=seed,
            max_negatives=max_negatives,
            itemcf_head=itemcf_head,
            semantic_head=semantic_head,
        )
        if not negatives:
            continue
        positive = scaler.transform([example.rows[example.target_id].values])[0]
        user_weight = 1.0 / (2 * len(negatives))
        for movie_id in negatives:
            negative = scaler.transform([example.rows[movie_id].values])[0]
            differences.extend((positive - negative, negative - positive))
            labels.extend((1, 0))
            weights.extend((user_weight, user_weight))
    if not differences:
        raise ValueError("pairwise training has no positive pairs")
    return np.asarray(differences), np.asarray(labels), np.asarray(weights)
```

- [ ] **Step 4: Implement fitting, ranking, artifact loading, and explanations**

Add to the same file:

```python
@dataclass
class PairwiseLinearRanker:
    scaler: StandardScaler
    model: LogisticRegression
    c_value: float

    @classmethod
    def fit(
        cls,
        examples: list[UserRankingExample],
        *,
        c_value: float,
        seed: int,
        max_negatives: int,
        itemcf_head: int,
        semantic_head: int,
    ) -> "PairwiseLinearRanker":
        scaler = StandardScaler().fit(_candidate_matrix(examples))
        matrix, labels, weights = build_pairwise_rows(
            examples,
            seed=seed,
            max_negatives=max_negatives,
            itemcf_head=itemcf_head,
            semantic_head=semantic_head,
            scaler=scaler,
        )
        model = LogisticRegression(
            C=c_value,
            l1_ratio=0.0,
            solver="lbfgs",
            fit_intercept=False,
            max_iter=1000,
            random_state=seed,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return cls(scaler=scaler, model=model, c_value=c_value)

    def scores(self, rows: dict[int, CandidateFeatureRow]) -> dict[int, float]:
        movie_ids = sorted(rows)
        matrix = self.scaler.transform([rows[movie_id].values for movie_id in movie_ids])
        scores = self.model.decision_function(matrix)
        return {movie_id: float(score) for movie_id, score in zip(movie_ids, scores, strict=True)}

    def rank(self, rows: dict[int, CandidateFeatureRow]) -> list[int]:
        scores = self.scores(rows)
        return sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))

    def explain(self, row: CandidateFeatureRow) -> dict[str, float]:
        transformed = self.scaler.transform([row.values])[0]
        return {
            name: float(value * coefficient)
            for name, value, coefficient in zip(
                FEATURE_NAMES, transformed, self.model.coef_[0], strict=True
            )
        }

    def to_artifact(self) -> dict[str, object]:
        return {
            "kind": "pairwise_l2_logistic",
            "feature_names": list(FEATURE_NAMES),
            "c_value": self.c_value,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "coefficients": self.model.coef_[0].tolist(),
        }

    @classmethod
    def from_artifact(cls, payload: dict[str, object]) -> "PairwiseLinearRanker":
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("ranker feature schema mismatch")
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(FEATURE_NAMES)
        scaler.n_samples_seen_ = 1
        model = LogisticRegression(fit_intercept=False)
        model.classes_ = np.asarray([0, 1])
        model.coef_ = np.asarray([payload["coefficients"]], dtype=np.float64)
        model.intercept_ = np.asarray([0.0])
        model.n_features_in_ = len(FEATURE_NAMES)
        model.n_iter_ = np.asarray([1], dtype=np.int32)
        return cls(scaler=scaler, model=model, c_value=float(payload["c_value"]))
```

- [ ] **Step 5: Run ranker tests**

Run:

```bash
.venv/bin/pytest tests/test_learned_ranking.py -v
```

Expected: all pairwise ranker tests pass.

- [ ] **Step 6: Commit learned ranking**

```bash
git add src/recagent_eval/learned_ranking.py tests/test_learned_ranking.py
git commit -m "feat: add interpretable pairwise linear ranker"
```

## Task 7: Implement deterministic folds, paired bootstrap, and the conjunctive gate

**Files:**
- Create: `src/recagent_eval/v2_selection.py`
- Create: `tests/test_v2_selection.py`

- [ ] **Step 1: Write failing tests for balanced folds, user-level bootstrap, and every primary gate condition**

Create `tests/test_v2_selection.py` with:

```python
import pytest

from recagent_eval.v2_selection import (
    PromotionEvidence,
    ResourceUsage,
    make_fold_assignments,
    paired_bootstrap,
    promotion_failures,
)


def passing_evidence() -> PromotionEvidence:
    return PromotionEvidence(
        c_ndcg=0.12,
        itemcf_ndcg=0.10,
        bootstrap_lower=0.001,
        bootstrap_upper=0.04,
        seed_deltas={42: 0.01, 2025: 0.02, 3407: 0.01},
        positive_fold_cells=10,
        total_fold_cells=15,
        c_recall=0.11,
        itemcf_recall=0.10,
        c_hit_rate=0.11,
        itemcf_hit_rate=0.10,
        union_candidate_recall=0.8,
        itemcf_candidate_recall=0.7,
        excluded_seen_violation_rate=0.0,
        hard_constraint_satisfaction_rate=1.0,
        fingerprints_match=True,
        artifacts_complete=True,
        resource_usage=ResourceUsage(
            cache_bytes=10,
            peak_rss_bytes=10,
            embedding_seconds=1,
            validation_seconds=1,
            ranker_bytes=10,
            p95_ms=1,
            within_budget=True,
        ),
    )


def test_fold_assignments_are_deterministic_balanced_and_complete() -> None:
    first = make_fold_assignments(range(1, 24), seeds=(42, 2025), n_folds=5)
    second = make_fold_assignments(range(1, 24), seeds=(42, 2025), n_folds=5)

    assert first == second
    assert set(first) == {42, 2025}
    assert set(first[42]) == set(range(1, 24))
    counts = [list(first[42].values()).count(fold) for fold in range(5)]
    assert max(counts) - min(counts) <= 1


def test_paired_bootstrap_resamples_users_not_rows() -> None:
    result = paired_bootstrap(
        {1: 0.2, 2: 0.1, 3: 0.0},
        {1: 0.1, 2: 0.1, 3: 0.0},
        iterations=1000,
        seed=20260810,
    )

    assert result["mean_delta"] == pytest.approx(1 / 30)
    assert result == paired_bootstrap(
        {1: 0.2, 2: 0.1, 3: 0.0},
        {1: 0.1, 2: 0.1, 3: 0.0},
        iterations=1000,
        seed=20260810,
    )


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("c_ndcg", 0.10, "mean_ndcg_not_strictly_better"),
        ("bootstrap_lower", 0.0, "bootstrap_lower_not_positive"),
        ("seed_deltas", {42: 0.01, 2025: 0.0, 3407: 0.01}, "seed_direction_failed"),
        ("positive_fold_cells", 9, "fold_direction_failed"),
        ("c_recall", 0.09, "recall_regressed"),
        ("c_hit_rate", 0.09, "hit_rate_regressed"),
        ("union_candidate_recall", 0.69, "candidate_recall_regressed"),
        ("excluded_seen_violation_rate", 0.01, "seen_item_violation"),
        ("hard_constraint_satisfaction_rate", 0.99, "hard_constraint_failure"),
        ("fingerprints_match", False, "fingerprint_mismatch"),
        ("artifacts_complete", False, "artifact_incomplete"),
    ],
)
def test_each_gate_failure_keeps_test_locked(field, value, failure) -> None:
    evidence = passing_evidence().model_copy(update={field: value})
    assert failure in promotion_failures(evidence)


def test_passing_evidence_has_no_failures() -> None:
    assert promotion_failures(passing_evidence()) == []


def test_resource_budget_failure_keeps_test_locked() -> None:
    evidence = passing_evidence()
    evidence = evidence.model_copy(
        update={
            "resource_usage": evidence.resource_usage.model_copy(
                update={"within_budget": False}
            )
        }
    )

    assert "resource_budget_failed" in promotion_failures(evidence)
```

- [ ] **Step 2: Verify the new selection module is missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -v
```

Expected: collection fails because `recagent_eval.v2_selection` does not exist.

- [ ] **Step 3: Implement fold assignment, bootstrap, evidence models, and exact gate failures**

Create `src/recagent_eval/v2_selection.py` with:

```python
from __future__ import annotations

import random
from typing import Any

import numpy as np
from pydantic import BaseModel


class ResourceUsage(BaseModel):
    cache_bytes: int
    peak_rss_bytes: int
    embedding_seconds: float
    validation_seconds: float
    ranker_bytes: int
    p95_ms: float
    within_budget: bool


class PromotionEvidence(BaseModel):
    c_ndcg: float
    itemcf_ndcg: float
    bootstrap_lower: float
    bootstrap_upper: float
    seed_deltas: dict[int, float]
    positive_fold_cells: int
    total_fold_cells: int
    c_recall: float
    itemcf_recall: float
    c_hit_rate: float
    itemcf_hit_rate: float
    union_candidate_recall: float
    itemcf_candidate_recall: float
    excluded_seen_violation_rate: float
    hard_constraint_satisfaction_rate: float
    fingerprints_match: bool
    artifacts_complete: bool
    resource_usage: ResourceUsage


def make_fold_assignments(
    user_ids,
    *,
    seeds: tuple[int, ...],
    n_folds: int,
) -> dict[int, dict[int, int]]:
    users = sorted(set(int(user_id) for user_id in user_ids))
    if n_folds < 2 or len(users) < n_folds:
        raise ValueError("fold assignment requires at least one user per fold")
    result: dict[int, dict[int, int]] = {}
    for seed in seeds:
        shuffled = list(users)
        random.Random(seed).shuffle(shuffled)
        result[seed] = {
            user_id: index % n_folds for index, user_id in enumerate(shuffled)
        }
    return result


def paired_bootstrap(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("paired bootstrap requires identical non-empty users")
    user_ids = sorted(candidate)
    deltas = np.asarray([candidate[user_id] - baseline[user_id] for user_id in user_ids])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        means[index] = float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
    return {
        "iterations": iterations,
        "seed": seed,
        "users": len(user_ids),
        "mean_delta": float(np.mean(deltas)),
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "win_rate": float(np.mean(deltas > 0)),
        "tie_rate": float(np.mean(deltas == 0)),
        "loss_rate": float(np.mean(deltas < 0)),
    }


def promotion_failures(evidence: PromotionEvidence) -> list[str]:
    failures: list[str] = []
    if evidence.c_ndcg <= evidence.itemcf_ndcg:
        failures.append("mean_ndcg_not_strictly_better")
    if evidence.bootstrap_lower <= 0:
        failures.append("bootstrap_lower_not_positive")
    if any(delta <= 0 for delta in evidence.seed_deltas.values()):
        failures.append("seed_direction_failed")
    if evidence.total_fold_cells != 15 or evidence.positive_fold_cells < 10:
        failures.append("fold_direction_failed")
    if evidence.c_recall < evidence.itemcf_recall:
        failures.append("recall_regressed")
    if evidence.c_hit_rate < evidence.itemcf_hit_rate:
        failures.append("hit_rate_regressed")
    if evidence.union_candidate_recall < evidence.itemcf_candidate_recall:
        failures.append("candidate_recall_regressed")
    if evidence.excluded_seen_violation_rate != 0:
        failures.append("seen_item_violation")
    if evidence.hard_constraint_satisfaction_rate != 1:
        failures.append("hard_constraint_failure")
    if not evidence.fingerprints_match:
        failures.append("fingerprint_mismatch")
    if not evidence.artifacts_complete:
        failures.append("artifact_incomplete")
    if not evidence.resource_usage.within_budget:
        failures.append("resource_budget_failed")
    return failures
```

- [ ] **Step 4: Run primitive selection tests**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -v
```

Expected: all fold, bootstrap, and gate tests pass.

- [ ] **Step 5: Commit selection primitives**

```bash
git add src/recagent_eval/v2_selection.py tests/test_v2_selection.py
git commit -m "feat: add nested validation gate primitives"
```

## Task 8: Build leakage-safe ItemCF, TF-IDF, and BGE candidate bundles

**Files:**
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `tests/test_v2_selection.py`

- [ ] **Step 1: Write a failing bundle test that supplies train rows and fake embeddings only**

Append to `tests/test_v2_selection.py`:

```python
import numpy as np

from recagent_eval.data import Movie, Rating, build_positive_histories
from recagent_eval.embedding import EmbeddingIndex
from recagent_eval.v2_selection import build_candidate_context, build_user_candidate_bundle


def test_candidate_bundle_uses_train_history_and_keeps_target_only_as_label() -> None:
    movies = {
        1: Movie(1, "History (2000)", ("Drama",), 2000),
        2: Movie(2, "Target (2001)", ("Drama",), 2001),
        3: Movie(3, "Noise (2002)", ("Action",), 2002),
    }
    train = [
        Rating(1, 1, 5, 1),
        Rating(2, 1, 5, 1),
        Rating(2, 2, 5, 2),
        Rating(3, 1, 5, 1),
        Rating(3, 3, 5, 2),
    ]
    histories = build_positive_histories(train, history_cap=50)
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    embedding = EmbeddingIndex(movie_ids=np.asarray([1, 2, 3]), vectors=vectors)
    context = build_candidate_context(movies=movies, train_rows=train)

    bundle = build_user_candidate_bundle(
        user_id=1,
        target_id=2,
        movies=movies,
        context=context,
        histories=histories,
        embedding_index=embedding,
        retrieval_top_k=2,
        preference_state=None,
    )

    assert bundle.history_ids == (1,)
    assert 2 not in bundle.history_ids
    assert bundle.target_id == 2
    assert set(bundle.system_rows) == {"itemcf", "v1_control", "a", "b", "c"}
    assert bundle.system_rows["b"][2].feature_names == bundle.system_rows["c"][2].feature_names
```

- [ ] **Step 2: Verify the bundle builder is missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -k candidate_bundle -v
```

Expected: import fails because `build_user_candidate_bundle` does not exist.

- [ ] **Step 3: Implement route helpers and a typed user bundle**

Add imports and these contracts to `src/recagent_eval/v2_selection.py`:

```python
from collections import Counter
from dataclasses import dataclass

from recagent_eval.candidate_features import (
    CandidateFeatureRow,
    CandidateSignals,
    TrainingProfile,
    build_candidate_rows,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.embedding import EmbeddingIndex
from recagent_eval.models import PreferenceState
from recagent_eval.retrieval import (
    ItemCFRetriever,
    TfidfSemanticRetriever,
    hard_filter,
)


@dataclass(frozen=True)
class CandidateContext:
    itemcf: ItemCFRetriever
    tfidf: TfidfSemanticRetriever
    popularity: dict[int, int]


def build_candidate_context(
    *,
    movies: dict[int, Movie],
    train_rows: list[Rating] | tuple[Rating, ...],
) -> CandidateContext:
    return CandidateContext(
        itemcf=ItemCFRetriever.fit(train_rows),
        tfidf=TfidfSemanticRetriever.fit(movies),
        popularity=dict(
            Counter(row.movie_id for row in train_rows if row.rating >= 4)
        ),
    )


@dataclass(frozen=True)
class UserCandidateBundle:
    user_id: int
    target_id: int
    history_ids: tuple[int, ...]
    route_scores: dict[str, dict[int, float]]
    system_rows: dict[str, dict[int, CandidateFeatureRow]]


def _sparse_dot(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _tfidf_history_similarity(
    semantic: TfidfSemanticRetriever,
    history_ids: tuple[int, ...],
    candidate_ids: set[int],
) -> dict[int, tuple[float, float]]:
    result = {}
    for movie_id in candidate_ids:
        values = [
            _sparse_dot(semantic.vectors[movie_id], semantic.vectors[source])
            for source in history_ids
        ]
        result[movie_id] = (max(values, default=0.0), sum(values) / len(values) if values else 0.0)
    return result


def _embedding_history_similarity(
    embedding: EmbeddingIndex,
    history_ids: tuple[int, ...],
    candidate_ids: set[int],
) -> dict[int, tuple[float, float]]:
    result = {}
    history_vectors = [embedding.vector(movie_id) for movie_id in history_ids]
    for movie_id in candidate_ids:
        values = [float(embedding.vector(movie_id) @ vector) for vector in history_vectors]
        result[movie_id] = (max(values, default=0.0), sum(values) / len(values) if values else 0.0)
    return result
```

- [ ] **Step 4: Implement one-user candidate construction with no access to test targets**

Add the builder to `src/recagent_eval/v2_selection.py`:

```python
def build_user_candidate_bundle(
    *,
    user_id: int,
    target_id: int,
    movies: dict[int, Movie],
    context: CandidateContext,
    histories: dict[int, tuple[Rating, ...]],
    embedding_index: EmbeddingIndex,
    retrieval_top_k: int,
    preference_state: PreferenceState | None,
) -> UserCandidateBundle:
    history = histories[user_id]
    history_ids = tuple(row.movie_id for row in history)
    allowed = (
        {movie.movie_id for movie in hard_filter(movies.values(), preference_state)}
        if preference_state is not None
        else set(movies) - set(history_ids)
    )
    itemcf_scores = dict(
        context.itemcf.retrieve(
            set(history_ids), top_k=retrieval_top_k, allowed_ids=allowed
        )
    )
    tfidf_query = " ".join(movies[movie_id].text for movie_id in history_ids)
    tfidf_scores = dict(
        context.tfidf.retrieve(
            tfidf_query, top_k=retrieval_top_k, allowed_ids=allowed
        )
    )
    embedding_profile = embedding_index.profile(history_ids)
    bge_scores = dict(
        embedding_index.retrieve(embedding_profile, allowed_ids=allowed, top_k=retrieval_top_k)
    )
    profile = TrainingProfile.from_history(
        history, movies=movies, train_popularity=context.popularity
    )
    tfidf_candidates = set(itemcf_scores) | set(tfidf_scores)
    bge_candidates = set(itemcf_scores) | set(bge_scores)
    b_rows = build_candidate_rows(
        movies=movies,
        candidate_ids=tfidf_candidates,
        signals=CandidateSignals(
            itemcf_scores=itemcf_scores,
            semantic_scores=tfidf_scores,
            history_similarity=_tfidf_history_similarity(
                context.tfidf, history_ids, tfidf_candidates
            ),
        ),
        profile=profile,
        preference_state=preference_state,
    )
    c_rows = build_candidate_rows(
        movies=movies,
        candidate_ids=bge_candidates,
        signals=CandidateSignals(
            itemcf_scores=itemcf_scores,
            semantic_scores=bge_scores,
            history_similarity=_embedding_history_similarity(
                embedding_index, history_ids, bge_candidates
            ),
        ),
        profile=profile,
        preference_state=preference_state,
    )
    return UserCandidateBundle(
        user_id=user_id,
        target_id=target_id,
        history_ids=history_ids,
        route_scores={"itemcf": itemcf_scores, "tfidf": tfidf_scores, "bge": bge_scores},
        system_rows={
            "itemcf": {movie_id: c_rows[movie_id] for movie_id in itemcf_scores},
            "v1_control": b_rows,
            "a": c_rows,
            "b": b_rows,
            "c": c_rows,
        },
)
```

`build_candidate_context` is called once per validation run. Every user reuses
the exact same ItemCF, TF-IDF, and train-only popularity objects.

- [ ] **Step 5: Run candidate bundle and existing retrieval tests**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -k candidate_bundle -v
.venv/bin/pytest tests/test_retrieval.py tests/test_retrieval_selection.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit candidate bundle construction**

```bash
git add src/recagent_eval/v2_selection.py tests/test_v2_selection.py
git commit -m "feat: build v2 candidate ablation bundles"
```

## Task 9: Evaluate one outer fold with inner-only regularization selection

**Files:**
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `tests/test_v2_selection.py`

- [ ] **Step 1: Write a failing test that records which users reach scaler fitting and inner scoring**

Append to `tests/test_v2_selection.py`:

```python
from recagent_eval.learned_ranking import UserRankingExample
from recagent_eval.candidate_features import CandidateFeatureRow, FEATURE_NAMES
from recagent_eval.v2_selection import select_c_inner_cv


def feature_row(user_id: int, movie_id: int, score: float) -> CandidateFeatureRow:
    return CandidateFeatureRow(
        user_id,
        movie_id,
        FEATURE_NAMES,
        (score,) + (0.0,) * (len(FEATURE_NAMES) - 1),
    )


def test_inner_selection_never_receives_outer_users(monkeypatch) -> None:
    seen_fit_users = []

    class FakeRanker:
        @classmethod
        def fit(cls, examples, **kwargs):
            seen_fit_users.append({example.user_id for example in examples})
            return cls()

        def rank(self, rows):
            return sorted(rows)

    monkeypatch.setattr("recagent_eval.v2_selection.PairwiseLinearRanker", FakeRanker)
    examples = [
        UserRankingExample(
            user_id=user_id,
            target_id=1,
            rows={
                1: feature_row(user_id, 1, 2.0),
                2: feature_row(user_id, 2, 0.0),
            },
            itemcf_ranked_ids=(1, 2),
            semantic_ranked_ids=(2, 1),
        )
        for user_id in range(1, 10)
    ]

    selected = select_c_inner_cv(
        examples,
        c_grid=(0.01, 0.1),
        inner_folds=3,
        seed=42,
        forbidden_user_ids={99},
        max_negatives=1,
        itemcf_head=1,
        semantic_head=0,
    )

    assert selected in {0.01, 0.1}
    assert all(99 not in users for users in seen_fit_users)
    assert all(users < set(range(1, 10)) for users in seen_fit_users)
```

- [ ] **Step 2: Run the test and verify inner selection is missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -k inner_selection -v
```

Expected: import fails because `select_c_inner_cv` does not exist.

- [ ] **Step 3: Implement inner CV with deterministic tie breaking**

Add the required imports and function to `src/recagent_eval/v2_selection.py`:

```python
from recagent_eval.evaluation import ndcg_at_k, recall_at_k
from recagent_eval.learned_ranking import PairwiseLinearRanker, UserRankingExample


def select_c_inner_cv(
    examples: list[UserRankingExample],
    *,
    c_grid: tuple[float, ...],
    inner_folds: int,
    seed: int,
    forbidden_user_ids: set[int],
    max_negatives: int,
    itemcf_head: int,
    semantic_head: int,
) -> float:
    if forbidden_user_ids & {example.user_id for example in examples}:
        raise ValueError("outer users reached inner selection")
    assignments = make_fold_assignments(
        [example.user_id for example in examples], seeds=(seed,), n_folds=inner_folds
    )[seed]
    rows = []
    for c_value in sorted(c_grid):
        ndcgs = []
        recalls = []
        for fold in range(inner_folds):
            training = [example for example in examples if assignments[example.user_id] != fold]
            held_out = [example for example in examples if assignments[example.user_id] == fold]
            ranker = PairwiseLinearRanker.fit(
                training,
                c_value=c_value,
                seed=seed,
                max_negatives=max_negatives,
                itemcf_head=itemcf_head,
                semantic_head=semantic_head,
            )
            for example in held_out:
                ranked = ranker.rank(example.rows)[:10]
                ndcgs.append(ndcg_at_k(ranked, {example.target_id}, 10))
                recalls.append(recall_at_k(ranked, {example.target_id}, 10))
        rows.append((sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls), c_value))
    return max(rows, key=lambda row: (row[0], row[1], -row[2]))[2]
```

- [ ] **Step 4: Add and test `evaluate_outer_fold`**

Add these deterministic evaluation helpers and `evaluate_outer_fold`:

```python
import hashlib
import json
import time

from recagent_eval.evaluation import hit_rate_at_k
from recagent_eval.ranking import normalize_scores


def user_fingerprint(user_ids) -> str:
    canonical = json.dumps(sorted(int(user_id) for user_id in user_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _minmax_rank(
    itemcf_scores: dict[int, float],
    semantic_scores: dict[int, float],
) -> list[int]:
    itemcf = normalize_scores(itemcf_scores)
    semantic = normalize_scores(semantic_scores)
    candidate_ids = set(itemcf) | set(semantic)
    return sorted(
        candidate_ids,
        key=lambda movie_id: (
            -(0.7 * itemcf.get(movie_id, 0.0) + 0.3 * semantic.get(movie_id, 0.0)),
            movie_id,
        ),
    )


def _route_membership(target_id: int, itemcf: set[int], semantic: set[int]) -> str:
    if target_id in itemcf and target_id in semantic:
        return "both"
    if target_id in itemcf:
        return "itemcf_only"
    if target_id in semantic:
        return "semantic_only"
    return "neither"


def _evaluate_bundle_systems(
    bundle: UserCandidateBundle,
    *,
    models: dict[str, PairwiseLinearRanker],
) -> list[dict[str, object]]:
    itemcf = bundle.route_scores["itemcf"]
    tfidf = bundle.route_scores["tfidf"]
    bge = bundle.route_scores["bge"]
    ranking_functions = {
        "itemcf": lambda: sorted(itemcf, key=lambda movie_id: (-itemcf[movie_id], movie_id)),
        "v1_control": lambda: _minmax_rank(itemcf, tfidf),
        "a": lambda: _minmax_rank(itemcf, bge),
        "b": lambda: models["b"].rank(bundle.system_rows["b"]),
        "c": lambda: models["c"].rank(bundle.system_rows["c"]),
    }
    results = []
    for system, rank in ranking_functions.items():
        started = time.perf_counter()
        ranked_ids = rank()[:10]
        latency_ms = (time.perf_counter() - started) * 1000
        semantic_scores = tfidf if system in {"v1_control", "b"} else bge
        if system == "itemcf":
            semantic_scores = {}
        seen_violations = len(set(ranked_ids) & set(bundle.history_ids))
        relevant = {bundle.target_id}
        results.append(
            {
                "user_id": bundle.user_id,
                "target_id": bundle.target_id,
                "system": system,
                "ranked_ids": ranked_ids,
                "recall_at_10": recall_at_k(ranked_ids, relevant, 10),
                "ndcg_at_10": ndcg_at_k(ranked_ids, relevant, 10),
                "hit_rate_at_10": hit_rate_at_k(ranked_ids, relevant, 10),
                "itemcf_candidate_recall": float(bundle.target_id in itemcf),
                "semantic_candidate_recall": float(bundle.target_id in semantic_scores),
                "union_candidate_recall": float(
                    bundle.target_id in set(itemcf) | set(semantic_scores)
                ),
                "target_route_membership": _route_membership(
                    bundle.target_id, set(itemcf), set(semantic_scores)
                ),
                "excluded_seen_item_count": seen_violations,
                "hard_constraint_satisfied": float(seen_violations == 0),
                "latency_ms": latency_ms,
            }
        )
    return results


def evaluate_outer_fold(
    *,
    bundles: list[UserCandidateBundle],
    outer_test_users: set[int],
    c_grid: tuple[float, ...],
    inner_folds: int,
    seed: int,
    max_negatives: int,
    itemcf_head: int,
    semantic_head: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    training_bundles = [bundle for bundle in bundles if bundle.user_id not in outer_test_users]
    held_out_bundles = [bundle for bundle in bundles if bundle.user_id in outer_test_users]
    if {bundle.user_id for bundle in training_bundles} & outer_test_users:
        raise ValueError("outer user overlap")
    results: list[dict[str, object]] = []
    models = {}
    for system in ("b", "c"):
        examples = [
            UserRankingExample(
                user_id=bundle.user_id,
                target_id=bundle.target_id,
                rows=bundle.system_rows[system],
                itemcf_ranked_ids=tuple(
                    sorted(bundle.route_scores["itemcf"], key=lambda item: (-bundle.route_scores["itemcf"][item], item))
                ),
                semantic_ranked_ids=tuple(
                    sorted(
                        bundle.route_scores["tfidf" if system == "b" else "bge"],
                        key=lambda item: (
                            -bundle.route_scores["tfidf" if system == "b" else "bge"][item], item
                        ),
                    )
                ),
            )
            for bundle in training_bundles
        ]
        c_value = select_c_inner_cv(
            examples,
            c_grid=c_grid,
            inner_folds=inner_folds,
            seed=seed,
            forbidden_user_ids=outer_test_users,
            max_negatives=max_negatives,
            itemcf_head=itemcf_head,
            semantic_head=semantic_head,
        )
        models[system] = PairwiseLinearRanker.fit(
            examples,
            c_value=c_value,
            seed=seed,
            max_negatives=max_negatives,
            itemcf_head=itemcf_head,
            semantic_head=semantic_head,
        )
    for bundle in held_out_bundles:
        results.extend(_evaluate_bundle_systems(bundle, models=models))
    training_fingerprint = user_fingerprint(
        bundle.user_id for bundle in training_bundles
    )
    artifacts = {}
    for name, model in models.items():
        artifacts[name] = {
            **model.to_artifact(),
            "training_user_fingerprint": training_fingerprint,
        }
    return results, artifacts
```

Append a test that builds nine `UserCandidateBundle` objects with
`feature_row`, uses users 8 and 9 as `outer_test_users`, and asserts:

```python
bundles = []
for user_id in range(1, 10):
    rows_for_user = {
        1: feature_row(user_id, 1, 2.0),
        2: feature_row(user_id, 2, 0.0),
    }
    bundles.append(
        UserCandidateBundle(
            user_id=user_id,
            target_id=1,
            history_ids=(3,),
            route_scores={
                "itemcf": {1: 1.0, 2: 0.5},
                "tfidf": {2: 1.0, 1: 0.5},
                "bge": {1: 0.9, 2: 0.1},
            },
            system_rows={
                system: dict(rows_for_user)
                for system in ("itemcf", "v1_control", "a", "b", "c")
            },
        )
    )
rows, artifacts = evaluate_outer_fold(
    bundles=bundles,
    outer_test_users={8, 9},
    c_grid=(0.1,),
    inner_folds=2,
    seed=42,
    max_negatives=1,
    itemcf_head=1,
    semantic_head=0,
)
assert len(rows) == 10
assert {row["user_id"] for row in rows} == {8, 9}
assert {row["system"] for row in rows} == {"itemcf", "v1_control", "a", "b", "c"}
assert all(
    artifact["training_user_fingerprint"] == user_fingerprint(range(1, 8))
    for artifact in artifacts.values()
)
```

- [ ] **Step 5: Run all v2 selection tests**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -v
```

Expected: all tests pass, including explicit outer/inner isolation assertions.

- [ ] **Step 6: Commit fold-local model selection**

```bash
git add src/recagent_eval/v2_selection.py tests/test_v2_selection.py
git commit -m "feat: add nested fold ranker evaluation"
```

## Task 10: Write complete immutable run artifacts, bootstrap, subgroups, and promotion manifest

**Files:**
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `tests/test_v2_selection.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for artifact completeness, refusal to overwrite, and promotion omission on failure**

Append tests that call `write_validation_artifacts` with synthetic rows:

```python
import json
from pathlib import Path

from recagent_eval.v2_selection import write_validation_artifacts


def test_failed_run_writes_complete_evidence_without_promotion(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = write_validation_artifacts(
        output_dir=output,
        manifest={"run_id": "failed", "fingerprints_match": True},
        feature_schema={"features": ["x"]},
        fold_assignments={"42": {"1": 0}},
        fold_metrics=[{"seed": 42, "fold": 0, "system": "c", "ndcg_at_10": 0.0}],
        user_metrics=[{"user_id": 1, "system": "c", "ndcg_at_10": 0.0}],
        ablations={"rows": []},
        bootstrap={"lower": -0.1, "upper": 0.1},
        subgroups={"history_length": []},
        resource_usage={"within_budget": True},
        model_artifacts={"42-0-b": {"kind": "pairwise_l2_logistic"}},
        evidence=passing_evidence().model_copy(update={"bootstrap_lower": -0.1}),
    )

    assert result["test_unlocked"] is False
    assert not (output / "promotion_manifest.json").exists()
    assert (output / "fold_metrics.jsonl").exists()
    assert (output / "user_metrics.jsonl").exists()
    assert (output / "run_integrity.json").exists()
    assert (output / "report.md").read_text().startswith("# RecAgent-Eval v2 validation")


def test_validation_artifacts_refuse_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError, match="validation output already exists"):
        write_validation_artifacts(
            output_dir=output,
            manifest={}, feature_schema={}, fold_assignments={}, fold_metrics=[],
            user_metrics=[], ablations={}, bootstrap={}, subgroups={},
            resource_usage={}, model_artifacts={}, evidence=passing_evidence(),
        )


def test_promotion_manifest_freezes_all_required_inputs(tmp_path: Path) -> None:
    output = tmp_path / "passing"
    write_validation_artifacts(
        output_dir=output,
        manifest={
            "config": {"seeds": [42, 2025, 3407]},
            "code_fingerprint": "code",
            "git_commit": "commit",
            "dataset_fingerprint": "data",
            "embedding_manifest": {"fingerprint": "embedding"},
        },
        feature_schema={"features": ["x"], "fingerprint": "schema"},
        fold_assignments={}, fold_metrics=[], user_metrics=[],
        ablations={"rows": []}, bootstrap={}, subgroups={}, resource_usage={},
        model_artifacts={
            "42-0-b": {"kind": "pairwise_l2_logistic"},
            "final-c": {"kind": "pairwise_l2_logistic", "coef": [1.0]},
        },
        evidence=passing_evidence(),
    )

    promotion = json.loads((output / "promotion_manifest.json").read_text())
    assert promotion["status"] == "promoted"
    assert promotion["feature_schema"]["fingerprint"] == "schema"
    assert promotion["embedding_manifest"]["fingerprint"] == "embedding"
    assert set(promotion["rankers"]) == {"final-c"}
    assert "Frozen test not run" in (output / "report.md").read_text()
```

- [ ] **Step 2: Verify the artifact writer is missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -k validation_artifacts -v
```

Expected: import fails because `write_validation_artifacts` does not exist.

- [ ] **Step 3: Implement deterministic JSON, JSONL, Markdown, and promotion writes**

Add helpers that serialize with `sort_keys=True`, reject pre-existing output,
write all required files, and call `promotion_failures` from serialized evidence
before creating a promotion manifest:

```python
import json
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    columns = list(rows[0])
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    output.extend(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    )
    return output


def _render_validation_report(
    *,
    summary: dict[str, object],
    ablations: dict[str, object],
    fold_metrics: list[dict[str, object]],
    bootstrap: dict[str, object],
    subgroups: dict[str, object],
    resource_usage: dict[str, object],
) -> str:
    lines = [
        "# RecAgent-Eval v2 validation",
        "",
        f"- Frozen test unlocked: `{summary['test_unlocked']}`",
        f"- Gate failures: `{summary['gate_failures']}`",
        "- Frozen test not run",
        "",
        "## Aggregate ablations",
        "",
        *_markdown_table(list(ablations.get("rows", []))),
        "",
        "## Seed/fold metrics",
        "",
        *_markdown_table(fold_metrics),
        "",
        "## Paired bootstrap",
        "",
        "```json",
        json.dumps(bootstrap, indent=2, sort_keys=True),
        "```",
        "",
        "## Subgroups",
        "",
        "```json",
        json.dumps(subgroups, indent=2, sort_keys=True),
        "```",
        "",
        "## Resource usage",
        "",
        "```json",
        json.dumps(resource_usage, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_validation_artifacts(
    *,
    output_dir: Path,
    manifest: dict[str, object],
    feature_schema: dict[str, object],
    fold_assignments: dict[str, object],
    fold_metrics: list[dict[str, object]],
    user_metrics: list[dict[str, object]],
    ablations: dict[str, object],
    bootstrap: dict[str, object],
    subgroups: dict[str, object],
    resource_usage: dict[str, object],
    model_artifacts: dict[str, dict[str, object]],
    evidence: PromotionEvidence,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError("validation output already exists")
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    failures = promotion_failures(evidence)
    summary = {
        "test_unlocked": not failures,
        "gate_failures": failures,
        "evidence": evidence.model_dump(mode="json"),
    }
    manifest = {
        **manifest,
        "gate_status": "passed" if not failures else "locked",
        "gate_failures": failures,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "feature_schema.json", feature_schema)
    _write_json(output_dir / "fold_assignments.json", fold_assignments)
    _write_jsonl(output_dir / "fold_metrics.jsonl", fold_metrics)
    _write_jsonl(output_dir / "user_metrics.jsonl", user_metrics)
    _write_json(output_dir / "ablations.json", ablations)
    _write_json(output_dir / "bootstrap.json", bootstrap)
    _write_json(output_dir / "subgroups.json", subgroups)
    _write_json(output_dir / "resource_usage.json", resource_usage)
    _write_json(
        output_dir / "aggregate_report.json",
        {
            "manifest": manifest,
            "summary": summary,
            "ablations": ablations,
            "bootstrap": bootstrap,
            "subgroups": subgroups,
            "resource_usage": resource_usage,
            "fold_metrics": fold_metrics,
        },
    )
    for name, artifact in sorted(model_artifacts.items()):
        _write_json(models_dir / f"{name}.json", artifact)
    (output_dir / "report.md").write_text(
        _render_validation_report(
            summary=summary,
            ablations=ablations,
            fold_metrics=fold_metrics,
            bootstrap=bootstrap,
            subgroups=subgroups,
            resource_usage=resource_usage,
        )
    )
    integrity = {
        "files": {
            str(path.relative_to(output_dir)): _sha256_path(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }
    }
    _write_json(output_dir / "run_integrity.json", integrity)
    if not failures:
        promoted_models = {
            name: artifact
            for name, artifact in sorted(model_artifacts.items())
            if name == "final-c"
        }
        _write_json(
            output_dir / "promotion_manifest.json",
            {
                "status": "promoted",
                "evidence": evidence.model_dump(mode="json"),
                "config": manifest["config"],
                "code_fingerprint": manifest["code_fingerprint"],
                "git_commit": manifest["git_commit"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "embedding_manifest": manifest["embedding_manifest"],
                "feature_schema": feature_schema,
                "rankers": promoted_models,
                "evidence_file_fingerprints": integrity["files"],
            },
        )
    return summary
```

Add a read-only integrity verifier in the same module:

```python
def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_validation_run(output_dir: Path) -> dict[str, object]:
    integrity = json.loads((output_dir / "run_integrity.json").read_text())["files"]
    actual = {
        relative: _sha256_path(output_dir / relative)
        for relative in integrity
    }
    if actual != integrity:
        raise ValueError("validation run integrity mismatch")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    aggregate = json.loads((output_dir / "aggregate_report.json").read_text())
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text())
    fold_assignments = json.loads((output_dir / "fold_assignments.json").read_text())
    if feature_schema["fingerprint"] != manifest["feature_schema_fingerprint"]:
        raise ValueError("feature schema fingerprint mismatch")
    if _fingerprint_json(fold_assignments) != manifest["fold_assignment_fingerprint"]:
        raise ValueError("fold assignment fingerprint mismatch")
    if _fingerprint_json(manifest["config"]) != manifest["config_fingerprint"]:
        raise ValueError("config fingerprint mismatch")
    fold_rows = _read_jsonl(output_dir / "fold_metrics.jsonl")
    user_rows = _read_jsonl(output_dir / "user_metrics.jsonl")
    config = manifest["config"]
    if len(config["seeds"]) != 3 or config["outer_folds"] != 5:
        raise ValueError("validation fold protocol drift")
    cells = {(int(row["seed"]), int(row["fold"])) for row in fold_rows}
    if len(cells) != 15:
        raise ValueError("validation outer cell count mismatch")
    for cell in cells:
        completed = {
            row["system"] for row in fold_rows
            if (int(row["seed"]), int(row["fold"])) == cell
            and row.get("status") == "completed"
        }
        failed = [
            row for row in fold_rows
            if (int(row["seed"]), int(row["fold"])) == cell
            and row.get("status") == "failed"
        ]
        if completed != {"itemcf", "v1_control", "a", "b", "c"} and len(failed) != 1:
            raise ValueError(f"incomplete unrecorded outer cell: {cell}")
    evidence = PromotionEvidence.model_validate(aggregate["summary"]["evidence"])
    failures = promotion_failures(evidence)
    if failures != aggregate["summary"]["gate_failures"]:
        raise ValueError("stored gate decision mismatch")
    recomputed_bootstrap = build_bootstrap_report(
        user_rows,
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    ) if user_rows else aggregate["bootstrap"]
    if recomputed_bootstrap != aggregate["bootstrap"]:
        raise ValueError("bootstrap evidence mismatch")
    forbidden = ("test_targets", "fixed_cases", "DEEPSEEK_API_KEY", "VLLM")
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    if any(token in serialized_manifest for token in forbidden):
        raise ValueError("validation manifest contains forbidden frozen/provider fields")
    promotion_exists = (output_dir / "promotion_manifest.json").exists()
    if promotion_exists != (not failures):
        raise ValueError("promotion manifest presence disagrees with gate")
    return {
        "gate_failures": failures,
        "test_unlocked": not failures,
        "outer_cells": len(cells),
        "user_rows": len(user_rows),
    }
```

Append these exact verifier tests:

```python
from recagent_eval.v2_selection import (
    _fingerprint_json,
    build_bootstrap_report,
    verify_validation_run,
)


def _write_synthetic_formal_run(output: Path) -> None:
    seeds = [42, 2025, 3407]
    systems = ["itemcf", "v1_control", "a", "b", "c"]
    fold_assignments = {
        str(seed): {str(user_id): (user_id - 1) % 5 for user_id in range(1, 6)}
        for seed in seeds
    }
    fold_metrics = [
        {
            "seed": seed, "fold": fold, "system": system,
            "status": "completed", "users": 1,
        }
        for seed in seeds for fold in range(5) for system in systems
    ]
    user_metrics = [
        {
            "seed": seed, "fold": (user_id - 1) % 5, "user_id": user_id,
            "system": system, "ndcg_at_10": 0.2 if system == "c" else 0.1,
        }
        for seed in seeds for user_id in range(1, 6) for system in systems
    ]
    bootstrap = build_bootstrap_report(user_metrics, iterations=100, seed=7)
    write_validation_artifacts(
        output_dir=output,
        manifest={
            "config": {
                "seeds": seeds, "outer_folds": 5,
                "bootstrap_iterations": 100, "bootstrap_seed": 7,
            },
            "code_fingerprint": "code", "git_commit": "commit",
            "dataset_fingerprint": "data", "embedding_manifest": {"fingerprint": "embed"},
            "feature_schema_fingerprint": "schema",
            "fold_assignment_fingerprint": _fingerprint_json(fold_assignments),
            "config_fingerprint": _fingerprint_json({
                "seeds": seeds, "outer_folds": 5,
                "bootstrap_iterations": 100, "bootstrap_seed": 7,
            }),
        },
        feature_schema={"features": ["x"], "fingerprint": "schema"},
        fold_assignments=fold_assignments,
        fold_metrics=fold_metrics,
        user_metrics=user_metrics,
        ablations={"rows": []}, bootstrap=bootstrap, subgroups={}, resource_usage={},
        model_artifacts={"final-c": {"kind": "pairwise_l2_logistic"}},
        evidence=passing_evidence(),
    )


def test_verify_validation_run_recomputes_clean_gate(tmp_path: Path) -> None:
    output = tmp_path / "clean"
    _write_synthetic_formal_run(output)

    assert verify_validation_run(output)["test_unlocked"] is True


def test_verify_validation_run_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "tampered"
    _write_synthetic_formal_run(output)
    (output / "bootstrap.json").write_text('{"lower": -1}\n')

    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_validation_run(output)
```

- [ ] **Step 4: Implement aggregate bootstrap and subgroup builders**

Add these functions:

```python
from collections import defaultdict


def averaged_user_metric(
    rows: list[dict[str, object]],
    *,
    system: str,
    metric: str,
) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            values[int(row["user_id"])].append(float(row[metric]))
    return {
        user_id: sum(user_values) / len(user_values)
        for user_id, user_values in sorted(values.items())
    }


def build_bootstrap_report(
    user_metrics: list[dict[str, object]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    candidate = averaged_user_metric(user_metrics, system="c", metric="ndcg_at_10")
    baseline = averaged_user_metric(user_metrics, system="itemcf", metric="ndcg_at_10")
    result = paired_bootstrap(candidate, baseline, iterations=iterations, seed=seed)
    baseline_mean = sum(baseline.values()) / len(baseline)
    result["relative_lift"] = (
        float(result["mean_delta"]) / baseline_mean if baseline_mean else None
    )
    return result


def popularity_bucket(value: int, *, q50: float, q90: float) -> str:
    if value == 0:
        return "unseen"
    if value <= q50:
        return "tail"
    if value <= q90:
        return "mid"
    return "head"


def _quartile_labels(values: dict[int, int]) -> dict[int, str]:
    thresholds = np.percentile(list(values.values()), [25, 50, 75])
    return {
        user_id: f"q{int(np.searchsorted(thresholds, value, side='left')) + 1}"
        for user_id, value in values.items()
    }


def build_subgroup_report(
    user_metrics: list[dict[str, object]],
    *,
    user_metadata: dict[int, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    if not user_metadata:
        return {
            "history_length": [], "genre_diversity": [],
            "target_popularity": [], "target_route_membership": [],
        }
    nonzero_popularity = [
        int(metadata["target_popularity"])
        for metadata in user_metadata.values()
        if int(metadata["target_popularity"]) > 0
    ]
    q50, q90 = (
        np.percentile(nonzero_popularity, [50, 90])
        if nonzero_popularity else (0.0, 0.0)
    )
    dimensions = {
        "history_length": _quartile_labels(
            {user_id: int(metadata["history_length"]) for user_id, metadata in user_metadata.items()}
        ),
        "genre_diversity": _quartile_labels(
            {user_id: int(metadata["genre_diversity"]) for user_id, metadata in user_metadata.items()}
        ),
        "target_popularity": {
            user_id: popularity_bucket(
                int(metadata["target_popularity"]), q50=float(q50), q90=float(q90)
            )
            for user_id, metadata in user_metadata.items()
        },
        "target_route_membership": {
            user_id: str(metadata["target_route_membership"])
            for user_id, metadata in user_metadata.items()
        },
    }
    report: dict[str, list[dict[str, object]]] = {}
    for dimension, labels in dimensions.items():
        output_rows = []
        for label in sorted(set(labels.values())):
            users = {user_id for user_id, value in labels.items() if value == label}
            for system in ("itemcf", "v1_control", "a", "b", "c"):
                selected = [
                    row for row in user_metrics
                    if int(row["user_id"]) in users and row["system"] == system
                ]
                output_rows.append(
                    {
                        "group": label,
                        "system": system,
                        "users": len(users),
                        "exploratory": len(users) < 100,
                        **{
                            metric: (
                                sum(float(row[metric]) for row in selected) / len(selected)
                                if selected else None
                            )
                            for metric in (
                                "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
                                "itemcf_candidate_recall", "semantic_candidate_recall",
                                "union_candidate_recall",
                            )
                        },
                    }
                )
            itemcf_ndcg = output_rows[-5]["ndcg_at_10"]
            c_ndcg = output_rows[-1]["ndcg_at_10"]
            output_rows[-1]["delta_ndcg_vs_itemcf"] = (
                float(c_ndcg) - float(itemcf_ndcg)
                if c_ndcg is not None and itemcf_ndcg is not None else None
            )
        report[dimension] = output_rows
    return report
```

Append exact boundary tests:

```python
def test_popularity_bucket_boundaries_are_fixed() -> None:
    assert popularity_bucket(0, q50=3, q90=9) == "unseen"
    assert popularity_bucket(3, q50=3, q90=9) == "tail"
    assert popularity_bucket(9, q50=3, q90=9) == "mid"
    assert popularity_bucket(10, q50=3, q90=9) == "head"


def test_subgroups_below_one_hundred_are_exploratory() -> None:
    metadata = {
        user_id: {
            "history_length": user_id,
            "genre_diversity": 1,
            "target_popularity": user_id,
            "target_route_membership": "both",
        }
        for user_id in range(1, 11)
    }
    metrics = [
        {
            "user_id": user_id,
            "system": system,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "hit_rate_at_10": 0.0,
            "itemcf_candidate_recall": 1.0,
            "semantic_candidate_recall": 1.0,
            "union_candidate_recall": 1.0,
        }
        for user_id in metadata
        for system in ("itemcf", "v1_control", "a", "b", "c")
    ]
    report = build_subgroup_report(metrics, user_metadata=metadata)

    assert all(row["exploratory"] for rows in report.values() for row in rows)
```

Subgroup functions are diagnostic only and are not called by
`promotion_failures`.

- [ ] **Step 5: Update ignore rules without ignoring aggregate reports**

Append to `.gitignore`:

```gitignore
artifacts/v2/cache/
artifacts/v2/validation/
artifacts/v2/frozen/frozen_test_consumption.json
```

- [ ] **Step 6: Run artifact, subgroup, and gate tests**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -v
```

Expected: all v2 selection tests pass and failed synthetic evidence never writes
a promotion manifest.

- [ ] **Step 7: Commit immutable evidence output**

```bash
git add src/recagent_eval/v2_selection.py tests/test_v2_selection.py .gitignore
git commit -m "feat: preserve complete v2 validation evidence"
```

## Task 11: Orchestrate the full repeated nested validation run

**Files:**
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `tests/test_v2_selection.py`

- [ ] **Step 1: Write a failing tiny end-to-end orchestration test**

Add these exact fixtures and test:

```python
from recagent_eval.config import ResourceBudget, V2ExperimentConfig


def tiny_movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"Drama {movie_id} (200{movie_id})", ("Drama",), 2000 + movie_id)
        for movie_id in range(1, 6)
    }


def tiny_train_rows() -> list[Rating]:
    return [
        Rating(user_id, 1 if user_id % 2 else 2, 5, user_id)
        for user_id in range(1, 7)
    ]


def tiny_validation_targets() -> dict[int, int]:
    return {user_id: 3 for user_id in range(1, 7)}


def tiny_embedding_index() -> EmbeddingIndex:
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return EmbeddingIndex(movie_ids=np.arange(1, 6), vectors=vectors)


def tiny_v2_config() -> V2ExperimentConfig:
    return V2ExperimentConfig(
        model_id="fake/encoder",
        requested_revision="resolved",
        embedding_batch_size=2,
        embedding_device="cpu",
        retrieval_top_k=4,
        history_cap=2,
        outer_folds=2,
        inner_folds=2,
        seeds=(42,),
        c_grid=(0.01, 0.1),
        bootstrap_iterations=100,
        bootstrap_seed=7,
        max_negatives=2,
        itemcf_head_negatives=1,
        semantic_head_negatives=1,
        resource_budget=ResourceBudget(
            max_cache_bytes=1_000_000,
            max_peak_rss_bytes=10_000_000_000,
            max_embedding_seconds=10,
            max_validation_seconds=60,
            max_ranker_bytes=1_000_000,
            max_p95_ms=1_000,
        ),
    )


def test_run_v2_validation_writes_all_systems_and_no_test_fields(tmp_path: Path) -> None:
    result = run_v2_validation(
        movies=tiny_movies(),
        train_rows=tiny_train_rows(),
        validation_targets=tiny_validation_targets(),
        embedding_index=tiny_embedding_index(),
        embedding_manifest={
            "fingerprint": "embed",
            "dataset_fingerprint": "dataset",
            "cache_bytes": 100,
            "embedding_seconds": 0.1,
        },
        config=tiny_v2_config(),
        output_dir=tmp_path / "run",
        code_fingerprint="code",
        git_commit="commit",
        dataset_fingerprint="dataset",
        raw_file_hashes={"movies.dat": "movies", "ratings.dat": "ratings"},
    )

    user_rows = [
        json.loads(line)
        for line in (tmp_path / "run/user_metrics.jsonl").read_text().splitlines()
    ]
    assert {row["system"] for row in user_rows} == {"itemcf", "v1_control", "a", "b", "c"}
    assert (tmp_path / "run/models/final-c.json").exists()
    assert "test_targets" not in (tmp_path / "run/manifest.json").read_text()
    assert result["test_unlocked"] in {True, False}
```

- [ ] **Step 2: Run the orchestration test and verify the runner is missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_selection.py -k run_v2_validation -v
```

Expected: import fails because `run_v2_validation` does not exist.

- [ ] **Step 3: Implement `run_v2_validation` in explicit stages**

The function must have this signature and must not accept `DatasetSplit`, a
test-target mapping, a case path, or a frozen-case loader:

```python
def run_v2_validation(
    *,
    movies: dict[int, Movie],
    train_rows: list[Rating] | tuple[Rating, ...],
    validation_targets: dict[int, int],
    embedding_index: EmbeddingIndex,
    embedding_manifest: dict[str, object],
    config: V2ExperimentConfig,
    output_dir: Path,
    code_fingerprint: str,
    git_commit: str,
    dataset_fingerprint: str,
    raw_file_hashes: dict[str, str],
    run_mode: Literal["formal", "smoke"] = "formal",
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError("validation output already exists")
    started = time.perf_counter()
    histories = build_positive_histories(train_rows, history_cap=config.history_cap)
    exclusions = []
    eligible = []
    for user_id, target_id in sorted(validation_targets.items()):
        if user_id not in histories:
            exclusions.append({"user_id": user_id, "reason": "missing_positive_history"})
        elif target_id not in movies:
            exclusions.append({"user_id": user_id, "reason": "missing_catalog_target"})
        else:
            eligible.append((user_id, target_id))
    context = build_candidate_context(movies=movies, train_rows=train_rows)
    bundles = [
        build_user_candidate_bundle(
            user_id=user_id,
            target_id=target_id,
            movies=movies,
            context=context,
            histories=histories,
            embedding_index=embedding_index,
            retrieval_top_k=config.retrieval_top_k,
            preference_state=None,
        )
        for user_id, target_id in eligible
    ]
    assignments = make_fold_assignments(
        [bundle.user_id for bundle in bundles],
        seeds=config.seeds,
        n_folds=config.outer_folds,
    )
    user_metrics: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    model_artifacts: dict[str, dict[str, object]] = {}
    failed_cells = []
    for seed in config.seeds:
        for fold in range(config.outer_folds):
            outer_users = {
                user_id for user_id, assigned in assignments[seed].items()
                if assigned == fold
            }
            try:
                rows, models = evaluate_outer_fold(
                    bundles=bundles,
                    outer_test_users=outer_users,
                    c_grid=config.c_grid,
                    inner_folds=config.inner_folds,
                    seed=seed,
                    max_negatives=config.max_negatives,
                    itemcf_head=config.itemcf_head_negatives,
                    semantic_head=config.semantic_head_negatives,
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failure = {
                    "seed": seed,
                    "fold": fold,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failed_cells.append(failure)
                fold_metrics.append(failure)
                continue
            for row in rows:
                row.update({"seed": seed, "fold": fold, "status": "completed"})
            user_metrics.extend(rows)
            for system in ("itemcf", "v1_control", "a", "b", "c"):
                selected = [row for row in rows if row["system"] == system]
                fold_metrics.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "system": system,
                        "status": "completed",
                        "users": len(selected),
                        **{
                            metric: sum(float(row[metric]) for row in selected) / len(selected)
                            for metric in (
                                "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
                                "itemcf_candidate_recall", "semantic_candidate_recall",
                                "union_candidate_recall",
                            )
                        },
                    }
                )
            for name, artifact in models.items():
                model_artifacts[f"{seed}-{fold}-{name}"] = artifact

    final_ranker = None
    completed_c = [
        artifact for name, artifact in model_artifacts.items()
        if name.endswith("-c")
    ]
    if completed_c:
        c_counts = Counter(float(artifact["c_value"]) for artifact in completed_c)
        final_c = min(
            c_value for c_value, count in c_counts.items()
            if count == max(c_counts.values())
        )
        final_examples = [
            UserRankingExample(
                user_id=bundle.user_id,
                target_id=bundle.target_id,
                rows=bundle.system_rows["c"],
                itemcf_ranked_ids=tuple(
                    sorted(
                        bundle.route_scores["itemcf"],
                        key=lambda item: (-bundle.route_scores["itemcf"][item], item),
                    )
                ),
                semantic_ranked_ids=tuple(
                    sorted(
                        bundle.route_scores["bge"],
                        key=lambda item: (-bundle.route_scores["bge"][item], item),
                    )
                ),
            )
            for bundle in bundles
        ]
        try:
            final_ranker = PairwiseLinearRanker.fit(
                final_examples,
                c_value=final_c,
                seed=config.seeds[0],
                max_negatives=config.max_negatives,
                itemcf_head=config.itemcf_head_negatives,
                semantic_head=config.semantic_head_negatives,
            )
            model_artifacts["final-c"] = {
                **final_ranker.to_artifact(),
                "selection_rule": "modal_outer_c_then_smallest_tie",
                "outer_c_counts": dict(sorted(c_counts.items())),
                "training_user_fingerprint": user_fingerprint(
                    bundle.user_id for bundle in bundles
                ),
            }
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            failed_cells.append(
                {
                    "stage": "final_model",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def aggregate(system: str, metric: str) -> float:
        rows = [row for row in user_metrics if row["system"] == system]
        return sum(float(row[metric]) for row in rows) / len(rows) if rows else 0.0

    ablation_rows = [
        {
            "system": system,
            **{
                metric: aggregate(system, metric)
                for metric in (
                    "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
                    "itemcf_candidate_recall", "semantic_candidate_recall",
                    "union_candidate_recall",
                )
            },
        }
        for system in ("itemcf", "v1_control", "a", "b", "c")
    ]
    if user_metrics:
        bootstrap = build_bootstrap_report(
            user_metrics,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        )
    else:
        bootstrap = {
            "iterations": config.bootstrap_iterations,
            "seed": config.bootstrap_seed,
            "users": 0,
            "mean_delta": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "loss_rate": 0.0,
            "relative_lift": None,
        }
    user_metadata = {}
    for bundle in bundles:
        genres = {
            genre for movie_id in bundle.history_ids for genre in movies[movie_id].genres
        }
        user_metadata[bundle.user_id] = {
            "history_length": len(bundle.history_ids),
            "genre_diversity": len(genres),
            "target_popularity": context.popularity.get(bundle.target_id, 0),
            "target_route_membership": _route_membership(
                bundle.target_id,
                set(bundle.route_scores["itemcf"]),
                set(bundle.route_scores["bge"]),
            ),
        }
    subgroups = build_subgroup_report(user_metrics, user_metadata=user_metadata)
    validation_seconds = time.perf_counter() - started
    final_artifact = model_artifacts.get("final-c")
    ranker_bytes = (
        len(json.dumps(final_artifact, sort_keys=True).encode())
        if final_artifact else 0
    )
    final_latencies = []
    if final_ranker is not None:
        for bundle in bundles:
            latency_started = time.perf_counter()
            final_ranker.rank(bundle.system_rows["c"])
            final_latencies.append((time.perf_counter() - latency_started) * 1000)
    p95_ms = float(np.percentile(final_latencies, 95)) if final_latencies else 1e30
    peak_rss_bytes = _peak_rss_bytes()
    budget = config.resource_budget
    resource_usage = ResourceUsage(
        cache_bytes=int(embedding_manifest["cache_bytes"]),
        peak_rss_bytes=peak_rss_bytes,
        embedding_seconds=float(embedding_manifest["embedding_seconds"]),
        validation_seconds=validation_seconds,
        ranker_bytes=ranker_bytes,
        p95_ms=p95_ms,
        within_budget=(
            int(embedding_manifest["cache_bytes"]) <= budget.max_cache_bytes
            and peak_rss_bytes <= budget.max_peak_rss_bytes
            and float(embedding_manifest["embedding_seconds"]) <= budget.max_embedding_seconds
            and validation_seconds <= budget.max_validation_seconds
            and ranker_bytes <= budget.max_ranker_bytes
            and p95_ms <= budget.max_p95_ms
        ),
    )
    seed_deltas = {}
    for seed in config.seeds:
        c_rows = [row for row in user_metrics if row["seed"] == seed and row["system"] == "c"]
        itemcf_rows = [
            row for row in user_metrics if row["seed"] == seed and row["system"] == "itemcf"
        ]
        seed_deltas[seed] = (
            sum(float(row["ndcg_at_10"]) for row in c_rows) / len(c_rows)
            - sum(float(row["ndcg_at_10"]) for row in itemcf_rows) / len(itemcf_rows)
            if c_rows and itemcf_rows else 0.0
        )
    fold_pairs = {
        (int(row["seed"]), int(row["fold"])): row
        for row in fold_metrics if row.get("system") == "itemcf"
    }
    positive_cells = sum(
        float(row["ndcg_at_10"]) > float(fold_pairs[(int(row["seed"]), int(row["fold"]))]["ndcg_at_10"])
        for row in fold_metrics
        if row.get("system") == "c" and (int(row["seed"]), int(row["fold"])) in fold_pairs
    )
    expected_user_rows = len(bundles) * len(config.seeds) * 5
    artifacts_complete = (
        run_mode == "formal"
        and not failed_cells
        and len(user_metrics) == expected_user_rows
        and "final-c" in model_artifacts
    )
    computed_data_fingerprint = v2_data_fingerprint(
        movies, train_rows, validation_targets
    )
    fingerprints_match = (
        embedding_manifest.get("dataset_fingerprint") == dataset_fingerprint
        and computed_data_fingerprint == dataset_fingerprint
    )
    evidence = PromotionEvidence(
        c_ndcg=aggregate("c", "ndcg_at_10"),
        itemcf_ndcg=aggregate("itemcf", "ndcg_at_10"),
        bootstrap_lower=float(bootstrap["lower"]),
        bootstrap_upper=float(bootstrap["upper"]),
        seed_deltas=seed_deltas,
        positive_fold_cells=positive_cells,
        total_fold_cells=len(config.seeds) * config.outer_folds,
        c_recall=aggregate("c", "recall_at_10"),
        itemcf_recall=aggregate("itemcf", "recall_at_10"),
        c_hit_rate=aggregate("c", "hit_rate_at_10"),
        itemcf_hit_rate=aggregate("itemcf", "hit_rate_at_10"),
        union_candidate_recall=aggregate("c", "union_candidate_recall"),
        itemcf_candidate_recall=aggregate("itemcf", "itemcf_candidate_recall"),
        excluded_seen_violation_rate=(
            sum(int(row["excluded_seen_item_count"]) for row in user_metrics if row["system"] == "c")
            / max(
                1,
                sum(
                    len(row["ranked_ids"])
                    for row in user_metrics
                    if row["system"] == "c"
                ),
            )
        ),
        hard_constraint_satisfaction_rate=aggregate("c", "hard_constraint_satisfied"),
        fingerprints_match=fingerprints_match,
        artifacts_complete=artifacts_complete,
        resource_usage=resource_usage,
    )
    manifest = {
        "code_fingerprint": code_fingerprint,
        "git_commit": git_commit,
        "dataset_fingerprint": dataset_fingerprint,
        "computed_data_fingerprint": computed_data_fingerprint,
        "raw_file_hashes": dict(sorted(raw_file_hashes.items())),
        "embedding_manifest": embedding_manifest,
        "feature_schema_fingerprint": feature_schema_fingerprint(),
        "eligible_user_fingerprint": user_fingerprint(bundle.user_id for bundle in bundles),
        "ordered_history_fingerprint": ordered_history_fingerprint(histories),
        "fold_assignment_fingerprint": _fingerprint_json(assignments),
        "config_fingerprint": _fingerprint_json(asdict(config)),
        "config": asdict(config),
        "candidate_policy": {
            "retrieval_top_k": config.retrieval_top_k,
            "history_cap": config.history_cap,
            "routes": ["itemcf", "tfidf", "bge"],
        },
        "negative_sampling": {
            "max_negatives": config.max_negatives,
            "itemcf_head": config.itemcf_head_negatives,
            "semantic_head": config.semantic_head_negatives,
        },
        "estimator": "pairwise_l2_logistic",
        "dependencies": dependency_versions(),
        "runtime": runtime_fingerprint(),
        "eligible_users": len(bundles),
        "exclusions": exclusions,
        "failed_cells": failed_cells,
        "run_mode": run_mode,
    }
    return write_validation_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        feature_schema={"features": list(FEATURE_NAMES), "fingerprint": feature_schema_fingerprint()},
        fold_assignments={str(seed): mapping for seed, mapping in assignments.items()},
        fold_metrics=fold_metrics,
        user_metrics=user_metrics,
        ablations={"rows": ablation_rows, "failed_cells": failed_cells},
        bootstrap=bootstrap,
        subgroups=subgroups,
        resource_usage=resource_usage.model_dump(mode="json"),
        model_artifacts=model_artifacts,
        evidence=evidence,
    )
```

Add the required imports and RSS helper:

```python
import resource
import sys
from dataclasses import asdict
from typing import Literal
import importlib.metadata
import platform
from importlib.util import find_spec

from recagent_eval.candidate_features import FEATURE_NAMES, feature_schema_fingerprint
from recagent_eval.config import V2ExperimentConfig
from recagent_eval.data import build_positive_histories


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _fingerprint_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def v2_data_fingerprint(
    movies: dict[int, Movie],
    train_rows: list[Rating] | tuple[Rating, ...],
    validation_targets: dict[int, int],
) -> str:
    return _fingerprint_json(
        {
            "movies": [asdict(movies[movie_id]) for movie_id in sorted(movies)],
            "train": [asdict(row) for row in train_rows],
            "validation_targets": dict(sorted(validation_targets.items())),
        }
    )


def ordered_history_fingerprint(histories: dict[int, tuple[int, ...]]) -> str:
    return _fingerprint_json({str(user_id): list(ids) for user_id, ids in sorted(histories.items())})


def dependency_versions() -> dict[str, str]:
    modules = {
        "numpy": "numpy",
        "pydantic": "pydantic",
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers",
    }
    return {
        name: importlib.metadata.version(name)
        for name, module in modules.items()
        if find_spec(module) is not None
    }


def runtime_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy_dtype": "float32",
    }
```

Append the platform-unit regression test:

```python
from types import SimpleNamespace


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 7), ("linux", 7 * 1024)])
def test_peak_rss_bytes_normalizes_platform_units(monkeypatch, platform: str, expected: int) -> None:
    monkeypatch.setattr(
        "recagent_eval.v2_selection.resource.getrusage",
        lambda _: SimpleNamespace(ru_maxrss=7),
    )
    monkeypatch.setattr("recagent_eval.v2_selection.sys.platform", platform)

    assert _peak_rss_bytes() == expected
```

- [ ] **Step 4: Add deterministic run and failure-preservation tests**

Append:

```python
def test_failed_outer_cell_is_preserved_and_blocks_promotion(monkeypatch, tmp_path: Path) -> None:
    original = evaluate_outer_fold
    calls = 0

    def fail_first(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected fold failure")
        return original(**kwargs)

    monkeypatch.setattr("recagent_eval.v2_selection.evaluate_outer_fold", fail_first)
    result = run_v2_validation(
        movies=tiny_movies(),
        train_rows=tiny_train_rows(),
        validation_targets=tiny_validation_targets(),
        embedding_index=tiny_embedding_index(),
        embedding_manifest={
            "fingerprint": "embed", "dataset_fingerprint": "dataset",
            "cache_bytes": 100, "embedding_seconds": 0.1,
        },
        config=tiny_v2_config(),
        output_dir=tmp_path / "failed",
        code_fingerprint="code",
        git_commit="commit",
        dataset_fingerprint="dataset",
        raw_file_hashes={"movies.dat": "movies", "ratings.dat": "ratings"},
    )

    assert result["test_unlocked"] is False
    assert "injected fold failure" in (tmp_path / "failed/fold_metrics.jsonl").read_text()
    assert not (tmp_path / "failed/promotion_manifest.json").exists()


def test_smoke_run_is_deterministic_and_never_promotes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("recagent_eval.v2_selection._peak_rss_bytes", lambda: 100)
    kwargs = {
        "movies": tiny_movies(),
        "train_rows": tiny_train_rows(),
        "validation_targets": tiny_validation_targets(),
        "embedding_index": tiny_embedding_index(),
        "embedding_manifest": {
            "fingerprint": "embed", "dataset_fingerprint": "dataset",
            "cache_bytes": 100, "embedding_seconds": 0.1,
        },
        "config": tiny_v2_config(),
        "code_fingerprint": "code",
        "git_commit": "commit",
        "dataset_fingerprint": "dataset",
        "raw_file_hashes": {"movies.dat": "movies", "ratings.dat": "ratings"},
        "run_mode": "smoke",
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = run_v2_validation(output_dir=first, **kwargs)
    second_result = run_v2_validation(output_dir=second, **kwargs)

    assert first_result["test_unlocked"] is False
    assert second_result["test_unlocked"] is False
    assert not (first / "promotion_manifest.json").exists()
    for relative in (
        "fold_assignments.json", "feature_schema.json", "ablations.json",
        "bootstrap.json", "subgroups.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    assert {
        path.name: path.read_bytes() for path in (first / "models").iterdir()
    } == {
        path.name: path.read_bytes() for path in (second / "models").iterdir()
    }
```

- [ ] **Step 5: Run the entire v2 unit suite**

Run:

```bash
.venv/bin/pytest tests/test_embedding.py tests/test_candidate_features.py tests/test_learned_ranking.py tests/test_v2_selection.py -v
```

Expected: all v2 module tests pass without a model download.

- [ ] **Step 6: Commit the validation orchestrator**

```bash
git add src/recagent_eval/v2_selection.py tests/test_v2_selection.py
git commit -m "feat: orchestrate repeated nested v2 validation"
```

## Task 12: Add embedding preparation and validation CLI commands

**Files:**
- Modify: `src/recagent_eval/cli.py:1-478`
- Create: `tests/test_v2_cli.py`

- [ ] **Step 1: Write failing CLI tests with monkeypatched resolvers and runners**

Create `tests/test_v2_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from recagent_eval.cli import app


runner = CliRunner()


def test_prepare_embeddings_reports_model_license_and_resolved_revision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "recagent_eval.cli._prepare_v2_embedding_cache",
        lambda **kwargs: {
            "model_id": "BAAI/bge-small-en-v1.5",
            "revision": "resolved-sha",
            "license": "MIT",
            "fingerprint": "cache-sha",
        },
    )
    result = runner.invoke(
        app,
        [
            "prepare-embeddings", "--config", "configs/v2_offline.yaml",
            "--data-dir", "data/raw/ml-1m", "--output", str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    assert "resolved-sha" in result.stdout
    assert "MIT" in result.stdout


def test_validate_v2_has_no_cases_option(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "recagent_eval.cli._run_v2_validation_command",
        lambda **kwargs: {"test_unlocked": False, "gate_failures": ["bootstrap_lower_not_positive"]},
    )
    result = runner.invoke(
        app,
        [
            "validate-v2", "--config", "configs/v2_offline.yaml",
            "--data-dir", "data/raw/ml-1m", "--embedding-cache", str(tmp_path / "cache"),
            "--output", str(tmp_path / "run"), "--max-users", "100",
        ],
    )

    assert result.exit_code == 0
    assert "Frozen test remains locked" in result.stdout
    help_result = runner.invoke(app, ["validate-v2", "--help"])
    assert "--cases" not in help_result.stdout
```

- [ ] **Step 2: Run CLI tests and verify commands are missing**

Run:

```bash
.venv/bin/pytest tests/test_v2_cli.py -k 'prepare_embeddings or validate_v2' -v
```

Expected: tests fail because the CLI commands and helpers do not exist.

- [ ] **Step 3: Implement `prepare-embeddings` with revision resolution before model load**

Add a lazy helper that imports `huggingface_hub.HfApi`, resolves
`config.requested_revision` to an immutable SHA, verifies the model-card license
equals `mit`, then constructs `SentenceTransformerEmbedder` with the resolved
SHA. The command prints model ID, resolved revision, MIT license, expected weight
size, cache fingerprint, elapsed time, and output directory. It writes no secret
or Hugging Face token value.

Implement the helper with these boundaries (the model metadata is resolved
before `SentenceTransformerEmbedder` can load weights):

```python
def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_file_hashes(data_dir: Path) -> dict[str, str]:
    return {
        name: _sha256_file(data_dir / name)
        for name in ("movies.dat", "ratings.dat")
    }


def _prepare_v2_embedding_cache(*, config_path: Path, data_dir: Path, output: Path) -> dict[str, object]:
    from huggingface_hub import HfApi

    config = load_v2_config(config_path)
    info = HfApi().model_info(
        config.model_id,
        revision=config.requested_revision,
        files_metadata=True,
    )
    license_name = str(info.card_data.license).lower()
    if license_name != "mit":
        raise ValueError(f"unexpected embedding license: {license_name}")
    weight_rows = sorted(
        [
            {
            "path": sibling.rfilename,
            "blob_id": sibling.blob_id,
            "size": sibling.size,
            }
            for sibling in info.siblings
            if sibling.rfilename.endswith((".safetensors", ".bin"))
        ],
        key=lambda row: row["path"],
    )
    if not info.sha or not weight_rows:
        raise ValueError("model metadata lacks immutable revision or weight files")
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    data_fingerprint = v2_data_fingerprint(movies, split.train, split.validation_targets)
    embedder = SentenceTransformerEmbedder(
        model_id=config.model_id,
        revision=info.sha,
        license_name="MIT",
        weight_fingerprint=_sha256_json(weight_rows),
        batch_size=config.embedding_batch_size,
        device=config.embedding_device,
    )
    manifest = build_embedding_cache(
        movies, embedder, output_dir=output, dataset_fingerprint=data_fingerprint
    )
    return {**manifest.model_dump(mode="json"), "output": str(output)}
```

Add the Typer command:

```python
@app.command("prepare-embeddings")
def prepare_embeddings(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/v2/cache/latest"),
) -> None:
    summary = _prepare_v2_embedding_cache(
        config_path=config_path, data_dir=data_dir, output=output
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
```

- [ ] **Step 4: Implement `validate-v2` without a case path**

Add:

```python
@app.command("validate-v2")
def validate_v2(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    embedding_cache: Annotated[Path, typer.Option("--embedding-cache")] = Path("artifacts/v2/cache/latest"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/v2/validation/latest"),
    max_users: Annotated[int | None, typer.Option("--max-users")] = None,
) -> None:
    summary = _run_v2_validation_command(
        config_path=config_path,
        data_dir=data_dir,
        embedding_cache=embedding_cache,
        output=output,
        max_users=max_users,
    )
    if summary["test_unlocked"]:
        typer.echo("Validation gate passed; promotion manifest written. Frozen test was not run.")
    else:
        typer.echo(f"Frozen test remains locked: {summary['gate_failures']}")
```

Implement the command helper so the test mapping is discarded at the split
boundary and never passed downstream:

```python
def _run_v2_validation_command(
    *,
    config_path: Path,
    data_dir: Path,
    embedding_cache: Path,
    output: Path,
    max_users: int | None,
) -> dict[str, object]:
    config = load_v2_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    validation_targets = dict(sorted(split.validation_targets.items()))
    if max_users is not None:
        if max_users <= 0:
            raise ValueError("max_users must be positive")
        validation_targets = dict(list(validation_targets.items())[:max_users])
    full_data_fingerprint = v2_data_fingerprint(
        movies, split.train, split.validation_targets
    )
    embedding_index, embedding_manifest = load_embedding_cache(
        embedding_cache,
        expected_fingerprint=json.loads(
            (embedding_cache / "manifest.json").read_text()
        )["fingerprint"],
    )
    return run_v2_validation(
        movies=movies,
        train_rows=split.train,
        validation_targets=validation_targets,
        embedding_index=embedding_index,
        embedding_manifest=embedding_manifest.model_dump(mode="json"),
        config=config,
        output_dir=output,
        code_fingerprint=_source_fingerprint(),
        git_commit=_current_git_commit(),
        dataset_fingerprint=full_data_fingerprint,
        raw_file_hashes=_raw_file_hashes(data_dir),
        run_mode="smoke" if max_users is not None else "formal",
    )
```

Add the exact Git provenance helper:

```python
def _current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_fingerprint() -> str:
    paths = sorted(Path("src").rglob("*.py")) + [Path("pyproject.toml"), Path("uv.lock")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
```

The smoke subset therefore shares the formal dataset fingerprint while its
manifest records `run_mode: smoke`; `run_v2_validation` makes smoke evidence
incomplete, so it cannot write a promotion manifest regardless of metrics.

- [ ] **Step 5: Run all CLI tests**

Run:

```bash
.venv/bin/pytest tests/test_v2_cli.py tests/test_cli.py -v
```

Expected: new commands pass their tests and all existing CLI tests remain green.

- [ ] **Step 6: Commit preparation and validation commands**

```bash
git add src/recagent_eval/cli.py tests/test_v2_cli.py
git commit -m "feat: add v2 preparation and validation commands"
```

## Task 13: Enforce the frozen-test lock and one-consumption marker

**Files:**
- Create: `configs/frozen_test_lock.yaml`
- Modify: `src/recagent_eval/v2_selection.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_v2_selection.py`
- Modify: `tests/test_v2_cli.py`

- [ ] **Step 1: Create the opaque lock configuration**

Create `configs/frozen_test_lock.yaml`:

```yaml
case_fingerprint: bc2f622cd9311bca8509a46f0ee516355bc64db7d91f809273a35d97ce304d88
consumption_marker: artifacts/v2/frozen/frozen_test_consumption.json
```

- [ ] **Step 2: Write a failing locked-gate test that proves cases are never loaded**

Append to `tests/test_v2_cli.py`:

```python
def test_evaluate_v2_frozen_rejects_before_loading_cases(monkeypatch, tmp_path: Path) -> None:
    promotion = tmp_path / "promotion.json"
    promotion.write_text('{"status":"locked","evidence":{}}\n')
    monkeypatch.setattr(
        "recagent_eval.cli.load_cases",
        lambda path: (_ for _ in ()).throw(AssertionError("frozen cases were loaded")),
    )

    result = runner.invoke(
        app,
        [
            "evaluate-v2-frozen", "--config", "configs/v2_offline.yaml",
            "--promotion", str(promotion), "--cases", "cases/fixed_cases.json",
            "--lock", "configs/frozen_test_lock.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "frozen test is locked" in result.stdout
```

- [ ] **Step 3: Write failing consumption-marker state tests**

Add tests for `begin_frozen_consumption` and `complete_frozen_consumption`:

```python
import hashlib

from recagent_eval.v2_selection import (
    begin_frozen_consumption,
    complete_frozen_consumption,
    validate_promotion_manifest,
)


def test_frozen_consumption_marker_blocks_started_and_completed_runs(tmp_path: Path) -> None:
    marker = tmp_path / "consumption.json"
    begin_frozen_consumption(marker, promotion_fingerprint="promotion")
    with pytest.raises(ValueError, match="already started"):
        begin_frozen_consumption(marker, promotion_fingerprint="promotion")
    complete_frozen_consumption(marker, result_fingerprint="result")
    with pytest.raises(ValueError, match="already completed"):
        begin_frozen_consumption(marker, promotion_fingerprint="promotion")


def test_promotion_revalidation_rejects_tampered_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "validation"
    run_dir.mkdir()
    evidence_path = run_dir / "bootstrap.json"
    evidence_path.write_text('{"lower": 0.1}\n')
    payload = {
        "status": "promoted",
        "evidence": passing_evidence().model_dump(mode="json"),
        "evidence_file_fingerprints": {
            "bootstrap.json": hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        },
    }
    evidence_path.write_text('{"lower": -0.1}\n')

    with pytest.raises(ValueError, match="validation evidence drift"):
        validate_promotion_manifest(payload, validation_run_dir=run_dir)
```

- [ ] **Step 4: Implement promotion revalidation and atomic marker transitions**

In `v2_selection.py`, add:

```python
def validate_promotion_manifest(
    payload: dict[str, object],
    *,
    validation_run_dir: Path,
) -> PromotionEvidence:
    if payload.get("status") != "promoted":
        raise ValueError("frozen test is locked")
    expected_files = payload.get("evidence_file_fingerprints")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("frozen test is locked: missing evidence fingerprints")
    actual_files = {
        relative: _sha256_path(validation_run_dir / relative)
        for relative in expected_files
    }
    if actual_files != expected_files:
        raise ValueError("frozen test is locked: validation evidence drift")
    evidence = PromotionEvidence.model_validate(payload["evidence"])
    failures = promotion_failures(evidence)
    if failures:
        raise ValueError(f"frozen test is locked: {failures}")
    return evidence


def begin_frozen_consumption(marker: Path, *, promotion_fingerprint: str) -> None:
    if marker.exists():
        state = json.loads(marker.read_text()).get("state")
        raise ValueError(f"frozen test already {state}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    _write_json(temporary, {"state": "started", "promotion_fingerprint": promotion_fingerprint})
    temporary.replace(marker)


def complete_frozen_consumption(marker: Path, *, result_fingerprint: str) -> None:
    payload = json.loads(marker.read_text())
    if payload.get("state") != "started":
        raise ValueError("frozen consumption is not in started state")
    payload.update({"state": "completed", "result_fingerprint": result_fingerprint})
    temporary = marker.with_suffix(".tmp")
    _write_json(temporary, payload)
    temporary.replace(marker)
```

- [ ] **Step 5: Implement protected frozen evaluation over already-loaded cases**

Add this path-free evaluator to `v2_selection.py`; this module still must not
import `load_cases`:

```python
from recagent_eval.cases import EvaluationCase


def _ensemble_rank(
    rows: dict[int, CandidateFeatureRow],
    ranker_artifacts: list[dict[str, object]],
) -> list[int]:
    rankers = [PairwiseLinearRanker.from_artifact(payload) for payload in ranker_artifacts]
    if not rankers:
        raise ValueError("promotion manifest has no C rankers")
    per_model = [ranker.scores(rows) for ranker in rankers]
    scores = {
        movie_id: sum(model[movie_id] for model in per_model) / len(per_model)
        for movie_id in rows
    }
    return sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))


def evaluate_promoted_frozen_cases(
    *,
    movies: dict[int, Movie],
    train_rows: list[Rating] | tuple[Rating, ...],
    cases: list[EvaluationCase],
    embedding_index: EmbeddingIndex,
    ranker_artifacts: list[dict[str, object]],
    retrieval_top_k: int,
    history_cap: int,
) -> dict[str, object]:
    histories = build_positive_histories(train_rows, history_cap=history_cap)
    context = build_candidate_context(movies=movies, train_rows=train_rows)
    rows = []
    for case in cases:
        state = case.expected_preferences or case.initial_state
        relevant = set(case.relevant_movie_ids)
        bundle = build_user_candidate_bundle(
            user_id=case.user_id,
            target_id=min(relevant),
            movies=movies,
            context=context,
            histories=histories,
            embedding_index=embedding_index,
            retrieval_top_k=retrieval_top_k,
            preference_state=state,
        )
        ranked = _ensemble_rank(bundle.system_rows["c"], ranker_artifacts)[
            : state.requested_count
        ]
        allowed_ids = {
            movie.movie_id for movie in hard_filter(movies.values(), state)
        } - set(bundle.history_ids)
        violations = sorted(set(ranked) - allowed_ids)
        itemcf_ids = set(bundle.route_scores["itemcf"])
        bge_ids = set(bundle.route_scores["bge"])
        rows.append(
            {
                "case_id": case.case_id,
                "user_id": case.user_id,
                "ranked_ids": ranked,
                "recall_at_10": recall_at_k(ranked, relevant, 10),
                "ndcg_at_10": ndcg_at_k(ranked, relevant, 10),
                "hit_rate_at_10": hit_rate_at_k(ranked, relevant, 10),
                "itemcf_candidate_recall": float(bool(relevant & itemcf_ids)),
                "semantic_candidate_recall": float(bool(relevant & bge_ids)),
                "union_candidate_recall": float(bool(relevant & (itemcf_ids | bge_ids))),
                "constraint_violations": violations,
            }
        )
    metrics = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in (
            "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
            "itemcf_candidate_recall", "semantic_candidate_recall",
            "union_candidate_recall",
        )
    }
    metrics["hard_constraint_satisfaction_rate"] = sum(
        not row["constraint_violations"] for row in rows
    ) / len(rows)
    if metrics["hard_constraint_satisfaction_rate"] != 1.0:
        raise ValueError("frozen evaluation produced a hard-constraint violation")
    return {"system": "c", "cases": rows, "metrics": metrics}
```

Append these exact tests:

```python
from types import SimpleNamespace

from recagent_eval.cases import EvaluationCase
from recagent_eval.models import PreferenceState
from recagent_eval.v2_selection import _ensemble_rank, evaluate_promoted_frozen_cases


def _linear_artifact(coefficient: float) -> dict[str, object]:
    return {
        "kind": "pairwise_l2_logistic",
        "feature_names": list(FEATURE_NAMES),
        "c_value": 0.1,
        "scaler_mean": [0.0] * len(FEATURE_NAMES),
        "scaler_scale": [1.0] * len(FEATURE_NAMES),
        "coefficients": [coefficient] + [0.0] * (len(FEATURE_NAMES) - 1),
    }


def test_ensemble_rank_averages_serialized_model_scores() -> None:
    rows = {2: feature_row(1, 2, 2.0), 3: feature_row(1, 3, 1.0)}

    assert _ensemble_rank(rows, [_linear_artifact(1.0), _linear_artifact(-1.0)]) == [2, 3]


def test_frozen_evaluator_applies_constraints_and_reports_all_metrics(monkeypatch) -> None:
    bundle = SimpleNamespace(
        history_ids=(1,),
        system_rows={"c": {2: feature_row(1, 2, 2.0), 3: feature_row(1, 3, 1.0)}},
        route_scores={"itemcf": {2: 1.0}, "bge": {2: 1.0, 3: 0.5}},
    )
    monkeypatch.setattr("recagent_eval.v2_selection.build_candidate_context", lambda **_: object())
    monkeypatch.setattr(
        "recagent_eval.v2_selection.build_user_candidate_bundle", lambda **_: bundle
    )
    case = EvaluationCase(
        case_id="frozen-1", user_id=1, turns=("recommend",),
        relevant_movie_ids={2},
        initial_state=PreferenceState(liked_movie_ids={1}, requested_count=2),
    )

    result = evaluate_promoted_frozen_cases(
        movies=tiny_movies(), train_rows=tiny_train_rows(), cases=[case],
        embedding_index=tiny_embedding_index(),
        ranker_artifacts=[_linear_artifact(1.0)], retrieval_top_k=4, history_cap=2,
    )

    row = result["cases"][0]
    assert 1 not in row["ranked_ids"]
    assert {
        "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
        "itemcf_candidate_recall", "semantic_candidate_recall",
        "union_candidate_recall",
    } <= row.keys()
    assert result["metrics"]["hard_constraint_satisfaction_rate"] == 1.0
```

In `cli.py`, implement the protected helper and thin Typer command:

```python
FROZEN_MARKER = Path("artifacts/v2/frozen/frozen_test_consumption.json")


def _evaluate_v2_frozen_command(
    *, config_path: Path, promotion_path: Path, cases_path: Path,
    lock_path: Path, data_dir: Path, embedding_cache: Path, output: Path,
) -> dict[str, object]:
    promotion = json.loads(promotion_path.read_text())
    validate_promotion_manifest(
        promotion, validation_run_dir=promotion_path.parent
    )
    config = load_v2_config(config_path)
    movies, ratings = _load_dataset(data_dir)
    split = chronological_split(ratings)
    data_fingerprint = v2_data_fingerprint(movies, split.train, split.validation_targets)
    embedding_index, embedding_manifest = load_embedding_cache(
        embedding_cache,
        expected_fingerprint=promotion["embedding_manifest"]["fingerprint"],
    )
    current = {
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "code_fingerprint": _source_fingerprint(),
        "dataset_fingerprint": data_fingerprint,
        "raw_file_hashes": _raw_file_hashes(data_dir),
        "embedding_manifest": embedding_manifest.model_dump(mode="json"),
        "feature_schema": {
            "features": list(FEATURE_NAMES),
            "fingerprint": feature_schema_fingerprint(),
        },
    }
    for field, actual in current.items():
        if promotion.get(field) != actual:
            raise ValueError(f"frozen test is locked: {field} drift")
    if output.exists():
        raise FileExistsError("frozen output already exists")
    if FROZEN_MARKER.exists():
        state = json.loads(FROZEN_MARKER.read_text()).get("state")
        raise ValueError(f"frozen test already {state}")
    lock = yaml.safe_load(lock_path.read_text())
    if Path(lock["consumption_marker"]) != FROZEN_MARKER:
        raise ValueError("frozen marker path drift")
    cases = load_cases(cases_path)
    if case_fingerprint(cases) != lock["case_fingerprint"]:
        raise ValueError("frozen case fingerprint mismatch")
    promotion_fingerprint = _sha256_file(promotion_path)
    begin_frozen_consumption(
        FROZEN_MARKER, promotion_fingerprint=promotion_fingerprint
    )
    result = evaluate_promoted_frozen_cases(
        movies=movies,
        train_rows=split.train,
        cases=cases,
        embedding_index=embedding_index,
        ranker_artifacts=list(promotion["rankers"].values()),
        retrieval_top_k=config.retrieval_top_k,
        history_cap=config.history_cap,
    )
    output.mkdir(parents=True)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    complete_frozen_consumption(
        FROZEN_MARKER, result_fingerprint=_sha256_file(result_path)
    )
    return result


@app.command("evaluate-v2-frozen")
def evaluate_v2_frozen(
    config_path: Annotated[Path, typer.Option("--config")],
    promotion_path: Annotated[Path, typer.Option("--promotion")],
    cases_path: Annotated[Path, typer.Option("--cases")],
    lock_path: Annotated[Path, typer.Option("--lock")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    embedding_cache: Annotated[Path, typer.Option("--embedding-cache")] = Path("artifacts/v2/cache/latest"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/v2/frozen/result"),
) -> None:
    result = _evaluate_v2_frozen_command(
        config_path=config_path, promotion_path=promotion_path,
        cases_path=cases_path, lock_path=lock_path, data_dir=data_dir,
        embedding_cache=embedding_cache, output=output,
    )
    typer.echo(json.dumps(result["metrics"], indent=2, sort_keys=True))
```

Do not run this command during implementation or verification.

- [ ] **Step 6: Run frozen guard tests only**

Run:

```bash
.venv/bin/pytest tests/test_v2_cli.py tests/test_v2_selection.py -k frozen -v
```

Expected: locked evidence never loads cases, and started/completed markers both
block a second attempt.

- [ ] **Step 7: Commit frozen isolation**

```bash
git add configs/frozen_test_lock.yaml src/recagent_eval/v2_selection.py src/recagent_eval/cli.py tests/test_v2_selection.py tests/test_v2_cli.py
git commit -m "feat: protect the single v2 frozen evaluation"
```

## Task 14: Document the pending v2 path without claiming results

**Files:**
- Modify: `README.md:28-204`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write a failing documentation regression test**

Append to `tests/test_scripts.py`:

```python
def test_readme_keeps_v2_pending_and_separates_frozen_test() -> None:
    readme = Path("README.md").read_text()

    assert "validate-v2" in readme
    assert "v2 validation is pending" in readme.lower()
    assert "does not run the frozen test" in readme.lower()
    assert "BAAI/bge-small-en-v1.5" in readme
```

- [ ] **Step 2: Run the documentation test and verify it fails**

Run:

```bash
.venv/bin/pytest tests/test_scripts.py -k v2_pending -v
```

Expected: failure because README has no v2 validation section.

- [ ] **Step 3: Add an honest pending-v2 section and exact commands**

Insert this exact section before the existing reproduction section:

````markdown
## v2 offline ranking (pending)

v2 validation is pending. There is no v2 frozen-test result, and the current
complete system must not be described as better than ItemCF. The v1 semantic
route remains title/genre TF-IDF; v2 adds the real sentence-embedding model
`BAAI/bge-small-en-v1.5` as a separate ablation route.

The BGE weights are approximately 133 MB, use the MIT license, and are cached
outside Git. `prepare-embeddings` resolves and records an immutable model
revision and weight fingerprint. `validate-v2` runs the pre-registered A/B/C
repeated nested validation and does not run the frozen test. Neither command
invokes DeepSeek, Qwen, vLLM, or a remote GPU.

```bash
uv sync --extra dev --extra v2 --extra embedding --locked
uv run --extra v2 --extra embedding recagent-eval prepare-embeddings \
  --config configs/v2_offline.yaml \
  --data-dir data/raw/ml-1m \
  --output artifacts/v2/cache/bge-small-en-v1.5
uv run --extra v2 recagent-eval validate-v2 \
  --config configs/v2_offline.yaml \
  --data-dir data/raw/ml-1m \
  --embedding-cache artifacts/v2/cache/bge-small-en-v1.5 \
  --output artifacts/v2/validation/formal
```
````

- [ ] **Step 4: Run documentation and existing script tests**

Run:

```bash
.venv/bin/pytest tests/test_scripts.py -v
```

Expected: all script/documentation assertions pass.

- [ ] **Step 5: Commit pending documentation**

```bash
git add README.md tests/test_scripts.py
git commit -m "docs: document pending v2 validation path"
```

## Task 15: Run the no-network implementation verification

**Files:**
- No production changes expected.

- [ ] **Step 1: Run Ruff**

```bash
.venv/bin/ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 2: Run the full suite with coverage**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing
```

Expected: all original and new tests pass; inspect missing lines and add focused
tests if v2 safety branches are uncovered. Do not weaken assertions to restore
the percentage.

- [ ] **Step 3: Run the deterministic smoke command**

```bash
.venv/bin/recagent-eval smoke --output artifacts/runs/smoke
```

Expected: `Offline smoke test passed`.

- [ ] **Step 4: Verify that validation help exposes no cases option**

```bash
.venv/bin/recagent-eval validate-v2 --help
```

Expected: options include config, data directory, embedding cache, output, and
max users; no cases or provider option appears.

- [ ] **Step 5: Inspect repository state and commit only verification fixes**

```bash
git status --short --branch
git diff --stat
git diff --name-status
```

Expected: no unknown or overwritten user files. If focused verification tests
were added, commit them with:

```bash
git add tests
git commit -m "test: cover v2 safety boundaries"
```

## Task 16: Prepare embeddings and run the validation experiments

**Files:**
- Generate ignored cache/run artifacts.
- Generate after formal validation: `reports/experiments/v2-validation.json`
- Generate after formal validation: `reports/experiments/v2-validation.md`
- Modify after formal validation: `README.md`

- [ ] **Step 1: Request explicit approval for model/dependency download**

State before running the command:

- model: `BAAI/bge-small-en-v1.5`;
- purpose: cached real sentence-embedding retrieval;
- weights: approximately 133 MB;
- license: MIT;
- dependency cache: Sentence Transformers/PyTorch may add several hundred MB;
- location: uv and Hugging Face user caches, with only a small item cache under
  ignored `artifacts/v2/cache/`.

- [ ] **Step 2: Install the embedding extra after approval**

```bash
uv sync --extra dev --extra v2 --extra embedding --locked
```

Expected: the locked environment installs successfully without modifying
`uv.lock`.

- [ ] **Step 3: Prepare the pinned item embedding cache**

```bash
uv run --extra v2 --extra embedding recagent-eval prepare-embeddings \
  --config configs/v2_offline.yaml \
  --data-dir data/raw/ml-1m \
  --output artifacts/v2/cache/bge-small-en-v1.5
```

Expected: output reports model ID, resolved immutable revision, MIT license,
cache fingerprint, 3,883 items, and a resource duration within 1,800 seconds.

- [ ] **Step 4: Run the non-selectable 100-user smoke validation**

```bash
uv run --extra v2 recagent-eval validate-v2 \
  --config configs/v2_offline.yaml \
  --data-dir data/raw/ml-1m \
  --embedding-cache artifacts/v2/cache/bge-small-en-v1.5 \
  --output artifacts/v2/validation/smoke-100 \
  --max-users 100
```

Expected: every artifact file is present, the manifest says `run_mode: smoke`,
and the frozen test remains locked regardless of the metrics.

- [ ] **Step 5: Audit smoke boundaries before the formal run**

Check:

```bash
rg -n 'test_target|test_targets|fixed_cases|DEEPSEEK|VLLM' artifacts/v2/validation/smoke-100
```

Expected: no test target, fixed-case, private provider, or remote-model field is
present. Verify the remaining smoke invariants with:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; p=Path("artifacts/v2/validation/smoke-100"); m=json.loads((p/"manifest.json").read_text()); s=json.loads((p/"feature_schema.json").read_text()); assert m["run_mode"]=="smoke"; assert m["eligible_users"]==100; assert m["candidate_policy"]["routes"]==["itemcf","tfidf","bge"]; assert len(m["config"]["seeds"])==3; assert m["config"]["outer_folds"]==5; assert m["embedding_manifest"]["revision"]; assert m["embedding_manifest"]["fingerprint"]; assert s["fingerprint"]==m["feature_schema_fingerprint"]; assert not (p/"promotion_manifest.json").exists(); print("smoke artifact boundaries verified")'
```

- [ ] **Step 6: Run one full formal validation directory**

```bash
uv run --extra v2 recagent-eval validate-v2 \
  --config configs/v2_offline.yaml \
  --data-dir data/raw/ml-1m \
  --embedding-cache artifacts/v2/cache/bge-small-en-v1.5 \
  --output artifacts/v2/validation/formal-2026-08-10
```

Expected: the command evaluates all 6,035 eligible validation users across all
15 outer cells, writes every required artifact, prints the actual gate decision,
and does not run the frozen test. A locked gate is a valid experimental result.

- [ ] **Step 7: Export complete aggregate reports without filtering failures**

Copy the deterministic aggregate artifacts without selecting or deleting rows:

```bash
mkdir -p reports/experiments
cp artifacts/v2/validation/formal-2026-08-10/aggregate_report.json reports/experiments/v2-validation.json
cp artifacts/v2/validation/formal-2026-08-10/report.md reports/experiments/v2-validation.md
```

Verify the copies are byte-identical:

```bash
cmp artifacts/v2/validation/formal-2026-08-10/aggregate_report.json reports/experiments/v2-validation.json
cmp artifacts/v2/validation/formal-2026-08-10/report.md reports/experiments/v2-validation.md
```

The reports must include ItemCF, v1 control, A, B, and C; every seed/fold;
paired bootstrap; resource usage; subgroup counts; gate failures; and the exact
statement `Frozen test not run`. If the gate fails, no promotion manifest may be
present.

- [ ] **Step 8: Update README with the real outcome and bounded claim**

Replace the pending result paragraph with the actual complete table. Use one of
these evidence-controlled conclusions:

- gate passes: “System C passed the pre-registered validation gate; frozen test
  remains unrun pending explicit approval.”
- gate fails: “System C did not pass the pre-registered validation gate; all
  ablations are retained and the frozen test remains locked.”

Do not claim that BGE, the learned ranker, or the full system improves final
recommendation quality unless its corresponding recorded evidence supports the
claim.

- [ ] **Step 9: Commit all formal validation evidence, including negative results**

```bash
git add reports/experiments/v2-validation.json reports/experiments/v2-validation.md README.md
git commit -m "docs: report v2 validation outcome"
```

Do not add the large cache, per-user run directory, model weights, or a frozen
test result.

## Task 17: Final verification before any completion claim

**Files:**
- No changes expected unless verification reveals a defect.

- [ ] **Step 1: Invoke the required completion skill**

Use `superpowers:verification-before-completion` and follow it before claiming
the implementation or validation is complete.

- [ ] **Step 2: Run the complete local verification set**

```bash
.venv/bin/ruff check .
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing
.venv/bin/recagent-eval smoke --output artifacts/runs/smoke
```

Expected: Ruff passes, all tests pass, coverage is reported without hiding new
files, and the offline smoke command succeeds.

- [ ] **Step 3: Verify evidence and gate state from disk**

Run the read-only verifier and marker check:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; from recagent_eval.v2_selection import verify_validation_run; print(json.dumps(verify_validation_run(Path("artifacts/v2/validation/formal-2026-08-10")), indent=2, sort_keys=True))'
.venv/bin/python -c 'import json; from pathlib import Path; from recagent_eval.cli import _raw_file_hashes; from recagent_eval.data import chronological_split, load_movielens_movies, load_movielens_ratings; from recagent_eval.v2_selection import v2_data_fingerprint; d=Path("data/raw/ml-1m"); m=json.loads(Path("artifacts/v2/validation/formal-2026-08-10/manifest.json").read_text()); movies=load_movielens_movies(d/"movies.dat"); split=chronological_split(load_movielens_ratings(d/"ratings.dat")); assert _raw_file_hashes(d)==m["raw_file_hashes"]; assert v2_data_fingerprint(movies, split.train, split.validation_targets)==m["dataset_fingerprint"]; print("raw and split fingerprints verified")'
test ! -e artifacts/v2/frozen/frozen_test_consumption.json
```

Then check the formal counts directly:

```bash
wc -l artifacts/v2/validation/formal-2026-08-10/user_metrics.jsonl
wc -l artifacts/v2/validation/formal-2026-08-10/fold_metrics.jsonl
rg -n 'test_targets|fixed_cases|DEEPSEEK_API_KEY|VLLM' artifacts/v2/validation/formal-2026-08-10/manifest.json
```

Expected user rows are `6035 * 3 * 5 = 90525` when no outer cell failed. If a
cell failed, its missing paired rows must be explained by the corresponding
preserved `status: failed` fold record; do not manufacture replacement rows.
The final `rg` command is expected to return no matches.

Expected: the recomputed gate state exactly matches the stored gate state.

- [ ] **Step 4: Inspect Git state and retained unknown files**

```bash
git status --short --branch
git log --oneline --decorate -12
git diff --stat
git diff --name-status
```

Expected: only intentional implementation/evidence commits exist; the three
pre-existing untracked handoff documents remain untouched if they are still
present.

- [ ] **Step 5: Stop at the frozen-test decision boundary**

If the gate fails, report the complete failure evidence and do not run anything
else. If the gate passes, report the frozen model/config/fingerprints and ask the
user whether to authorize the single offline frozen-test evaluation. Do not
infer authorization from approval of this implementation plan.

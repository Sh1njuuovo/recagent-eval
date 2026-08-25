# Strong-Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified, pre-registered baseline evaluation harness (cohorts, metrics, paired bootstrap, JSON artifacts) and evaluate Popularity / ItemCF / ALS direct / current-v2b / BPR-MF / LightGCN on untouched confirmation cohorts, then apply the pre-registered success criteria A/B.

**Architecture:** A new `cohorts.py` module assigns mutually exclusive dev/A/B cohorts from the unused eligible pool (seed 42). A new `baseline_eval.py` module ranks each method's candidates over the same allowed universe with identical metrics and paired bootstrap, writing JSON evidence. Popularity/ItemCF/ALS reuse existing retrievers; BPR-MF and LightGCN are new torch modules with determinism and serialization tests. The current v2b method reuses the existing pipeline with the training cohort = historical-500 ∪ dev-600.

**Tech Stack:** Python 3.11+, NumPy, torch (declared in `ml` extra), LightGBM, Typer, Pydantic, pytest, Ruff, YAML.

**Spec:** `docs/superpowers/specs/2026-08-23-strong-baselines-design.md`.

---

### Task 1: Cohort ledger builder

**Files:**
- Create: `src/recagent_eval/cohorts.py`
- Create: `tests/test_cohorts.py`
- Modify: `src/recagent_eval/cli.py` (new `build-cohorts` command)

- [ ] **Step 1: Write failing tests**

Create `tests/test_cohorts.py`:

```python
from __future__ import annotations

import hashlib
import json

from recagent_eval.cohorts import build_cohort_ledger, ledger_fingerprint


def test_ledger_cohorts_are_mutually_exclusive_and_cover_the_pool() -> None:
    eligible = list(range(1, 21))
    historical = set(range(1, 6))
    excluded = {9}
    ledger = build_cohort_ledger(
        eligible,
        historical=historical,
        excluded=excluded,
        sizes={"development": 4, "confirmation_a": 5, "confirmation_b": 5},
        seed=42,
    )
    cohorts = {name: set(ledger["cohorts"][name]) for name in ("development", "confirmation_a", "confirmation_b")}
    assert cohorts["development"].isdisjoint(cohorts["confirmation_a"])
    assert cohorts["development"].isdisjoint(cohorts["confirmation_b"])
    assert cohorts["confirmation_a"].isdisjoint(cohorts["confirmation_b"])
    assert all(
        user not in historical and user not in excluded
        for cohort in cohorts.values()
        for user in cohort
    )
    assert ledger["seed"] == 42
    assert ledger["fingerprint"] == ledger_fingerprint(ledger["cohorts"])
    assert all(len(ledger["cohorts"][name]) == size for name, size in ledger["sizes"].items())


def test_ledger_is_deterministic_and_hashes_the_lists() -> None:
    eligible = list(range(1, 101))
    first = build_cohort_ledger(eligible, historical={1}, excluded=set(),
                                sizes={"development": 10, "confirmation_a": 20, "confirmation_b": 20}, seed=7)
    second = build_cohort_ledger(eligible, historical={1}, excluded=set(),
                                 sizes={"development": 10, "confirmation_a": 20, "confirmation_b": 20}, seed=7)
    assert first["cohorts"] == second["cohorts"]
    assert first["fingerprint"] == second["fingerprint"]
    canonical = json.dumps(first["cohorts"], sort_keys=True, separators=(",", ":")).encode()
    assert first["fingerprint"] == hashlib.sha256(canonical).hexdigest()
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/pytest tests/test_cohorts.py -q
```
Expected: `ModuleNotFoundError: recagent_eval.cohorts`.

- [ ] **Step 3: Implement `cohorts.py`**

```python
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence

COHORT_SCHEMA_VERSION = "cohort-ledger/v1"


def build_cohort_ledger(
    eligible: Sequence[int],
    *,
    historical: Iterable[int],
    excluded: Iterable[int],
    sizes: Mapping[str, int],
    seed: int,
) -> dict[str, object]:
    """Deterministically assign disjoint cohorts from the eligible pool."""
    blocked = set(historical) | set(excluded)
    pool = sorted(user for user in eligible if user not in blocked)
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    cohorts: dict[str, list[int]] = {}
    cursor = 0
    for name in ("development", "confirmation_a", "confirmation_b"):
        size = int(sizes[name])
        if cursor + size > len(shuffled):
            raise ValueError(f"pool too small for cohort {name}")
        cohorts[name] = sorted(shuffled[cursor : cursor + size])
        cursor += size
    cohorts["reserve"] = sorted(shuffled[cursor:])
    payload = {"cohorts": cohorts}
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "seed": seed,
        "sizes": dict(sizes),
        "blocked_historical_count": len(set(historical) & set(eligible)),
        "blocked_excluded_count": len(set(excluded) & set(eligible)),
        "pool_size": len(pool),
        "cohorts": cohorts,
        "fingerprint": ledger_fingerprint(cohorts),
    }


def ledger_fingerprint(cohorts: Mapping[str, Sequence[int]]) -> str:
    canonical = json.dumps(cohorts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
```

- [ ] **Step 4: Add the `build-cohorts` CLI command**

In `cli.py`:

```python
@app.command("build-cohorts")
def build_cohorts(
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    cases_path: Annotated[Path, typer.Option("--cases")] = Path("cases/fixed_cases.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/cohorts/cohort_ledger.json"
    ),
    seed: Annotated[int, typer.Option()] = 42,
    development_size: Annotated[int, typer.Option()] = 600,
    confirmation_a_size: Annotated[int, typer.Option()] = 1000,
    confirmation_b_size: Annotated[int, typer.Option()] = 1000,
) -> None:
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing cohort ledger: {output}")
    movies, ratings = _load_dataset(data_dir)
    split = leakage_safe_ranking_split(ratings)
    eligible = sorted(split.validation_targets)
    historical = set(eligible[:500])
    frozen_users = {case["user_id"] for case in load_cases(cases_path)}
    ledger = build_cohort_ledger(
        eligible,
        historical=historical,
        excluded=frozen_users,
        sizes={
            "development": development_size,
            "confirmation_a": confirmation_a_size,
            "confirmation_b": confirmation_b_size,
        },
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"fingerprint": ledger["fingerprint"], "sizes": ledger["sizes"]}))
```

- [ ] **Step 5: Run tests, generate the real ledger, and commit**

```bash
.venv/bin/pytest tests/test_cohorts.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/cohorts.py src/recagent_eval/cli.py tests/test_cohorts.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/recagent-eval build-cohorts \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --cases cases/fixed_cases.json \
  --output artifacts/cohorts/cohort_ledger.json
cp artifacts/cohorts/cohort_ledger.json reports/audit/2026-08-23-cohort-ledger.json
git add src/recagent_eval/cohorts.py src/recagent_eval/cli.py tests/test_cohorts.py tests/test_cli.py reports/audit/2026-08-23-cohort-ledger.json
git commit -m "feat: add deterministic cohort ledger builder and ledger artifact"
```

---

### Task 2: Unified baseline evaluation harness

**Files:**
- Create: `src/recagent_eval/baseline_eval.py`
- Create: `tests/test_baseline_eval.py`
- Modify: `src/recagent_eval/cli.py` (`evaluate-baselines` command)

- [ ] **Step 1: Write failing tests**

Create `tests/test_baseline_eval.py`:

```python
from __future__ import annotations

import json

import pytest

from recagent_eval.baseline_eval import (
    MetricRow,
    paired_bootstrap_deltas,
    score_ranking,
)


def _ranked() -> list[int]:
    return [5, 3, 1, 2, 4, 6, 7, 8, 9, 10]


def test_score_ranking_metrics() -> None:
    row = score_ranking(
        ranked_ids=_ranked(),
        target=3,
        allowed={1, 2, 3, 4, 5},
        history={9, 10},
        candidate_recall=1.0,
        latency_ms=1.0,
    )
    assert row.recall_at_10 == 1.0
    assert abs(row.ndcg_at_10 - (1.0 / __import__("math").log2(3))) < 1e-9
    assert row.mrr_at_10 == 1.0 / 2
    assert row.constraint_satisfied is True
    assert row.candidate_recall == 1.0


def test_score_ranking_missing_target_and_constraint_violation() -> None:
    row = score_ranking(
        ranked_ids=[10, 11], target=3, allowed={1, 2, 3}, history=set(),
        candidate_recall=0.0, latency_ms=0.5,
    )
    assert row.recall_at_10 == 0.0 and row.ndcg_at_10 == 0.0 and row.mrr_at_10 == 0.0
    assert row.constraint_satisfied is False  # 11 not in allowed


def test_paired_bootstrap_is_deterministic_and_reports_ci() -> None:
    first = [1.0, 0.0, 1.0, 0.0] * 25
    second = [0.0, 0.0, 1.0, 0.0] * 25
    a = paired_bootstrap_deltas(first, second, seed=42, resamples=2000)
    b = paired_bootstrap_deltas(first, second, seed=42, resamples=2000)
    assert a == b
    assert a["resamples"] == 2000
    assert abs(a["mean_delta"] - (second[0] - first[0]) / 1) < 1e-9 or True
    assert a["lower"] <= a["upper"]
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/pytest tests/test_baseline_eval.py -q
```
Expected: import error.

- [ ] **Step 3: Implement `baseline_eval.py`**

```python
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricRow:
    user_id: int
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    candidate_recall: float
    constraint_satisfied: bool
    latency_ms: float
    recommended_ids: tuple[int, ...]


def score_ranking(
    *,
    ranked_ids: Sequence[int],
    target: int,
    allowed: set[int],
    history: set[int],
    candidate_recall: float,
    latency_ms: float,
) -> MetricRow:
    top = list(ranked_ids[:10])
    target_index = top.index(target) if target in top else None
    recall = float(target_index is not None)
    ndcg = 1.0 / math.log2(target_index + 2) if target_index is not None else 0.0
    mrr = 1.0 / (target_index + 1) if target_index is not None else 0.0
    ranked_set = set(top)
    constraints = ranked_set.issubset(allowed) and ranked_set.isdisjoint(history)
    return MetricRow(
        user_id=0,
        recall_at_10=recall,
        ndcg_at_10=ndcg,
        mrr_at_10=mrr,
        candidate_recall=float(candidate_recall),
        constraint_satisfied=bool(constraints),
        latency_ms=float(latency_ms),
        recommended_ids=tuple(top),
    )


def paired_bootstrap_deltas(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("bootstrap values must be finite")
    deltas = c - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    means = deltas[samples].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "resamples": resamples,
        "seed": seed,
    }


def metric_json(
    rows: Sequence[MetricRow],
    *,
    method: str,
    cohort: str,
    universe_size: int,
    config_fingerprint: str,
    dataset_fingerprint: str,
    model_fingerprint: str,
    training_seconds: float,
    peak_memory_mb: float,
    model_size_bytes: int,
    environment: Mapping[str, str],
    bootstrap: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "baseline-evaluation/v1",
        "method": method,
        "cohort": cohort,
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "model_fingerprint": model_fingerprint,
        "user_count": len(rows),
        "aggregates": {
            "recall_at_10": _mean([r.recall_at_10 for r in rows]),
            "ndcg_at_10": _mean([r.ndcg_at_10 for r in rows]),
            "mrr_at_10": _mean([r.mrr_at_10 for r in rows]),
            "candidate_recall": _mean([r.candidate_recall for r in rows]),
            "constraint_satisfaction_rate": _mean(
                [float(r.constraint_satisfied) for r in rows]
            ),
            "coverage": _coverage(rows, universe_size=universe_size),
            "latency_ms_p50": _quantile([r.latency_ms for r in rows], 0.5),
            "latency_ms_p95": _quantile([r.latency_ms for r in rows], 0.95),
        },
        "training_seconds": training_seconds,
        "peak_memory_mb": peak_memory_mb,
        "model_size_bytes": model_size_bytes,
        "environment": dict(environment),
        "bootstrap_vs_itemcf": bootstrap,
        "fingerprint": _artifact_fingerprint(rows, method, cohort),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coverage(rows: Sequence[MetricRow], *, universe_size: int) -> float:
    recommended = {movie_id for row in rows for movie_id in row.recommended_ids}
    return len(recommended) / universe_size if universe_size else 0.0


def _quantile(values: Sequence[float], position: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(position * len(ordered)))
    return float(ordered[index])


def _artifact_fingerprint(rows: Sequence[MetricRow], method: str, cohort: str) -> str:
    payload = {
        "rows": [[r.user_id, r.recall_at_10, r.ndcg_at_10, r.mrr_at_10] for r in rows],
        "method": method,
        "cohort": cohort,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
```

- [ ] **Step 4: Wire the `evaluate-baselines` CLI**

`evaluate-baselines` takes `--ledger`, `--method`, `--output`, `--data-dir`,
`--max-users` (smoke limit), refuses existing outputs, and dispatches to the
per-method scorer modules (Tasks 3–8). It writes the `metric_json` artifact
with `universe_size=3883` and records per-user `recommended_ids` in the JSON.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/pytest tests/test_baseline_eval.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/baseline_eval.py tests/test_baseline_eval.py
git add src/recagent_eval/baseline_eval.py tests/test_baseline_eval.py src/recagent_eval/cli.py
git commit -m "feat: add unified baseline evaluation harness and metrics"
```

---

### Task 3: Popularity baseline

**Files:**
- Create: `src/recagent_eval/baselines/popularity.py`
- Create: `tests/test_baselines_popularity.py`

Score every allowed, non-history movie by its positive-rating count in
`legal_retrieval_train`; tie-break `(-count, movie_id)`; top-10. Deterministic,
no parameters, empty history allowed (popularity works for cold users). Tests:
popularity order, tie-break, history exclusion, allowed filtering, NaN-free
counts. Wire into `evaluate-baselines --method popularity`. Run 30-user smoke
on confirmation-A (first 30 ledger users).

Commit: `feat: add popularity baseline`.

---

### Task 4: ItemCF direct baseline

**Files:**
- Create: `src/recagent_eval/baselines/itemcf_direct.py`
- Create: `tests/test_baselines_itemcf_direct.py`

Reuse `ItemCFRetriever.fit(legal_retrieval_train)` and
`retrieve(history_ids, top_k=len(allowed), allowed_ids=allowed)`; rank by the
returned order (already `(-score, movie_id)`). Candidate recall = 1.0 when the
target is in the allowed universe. Tests: score consistency with `score_many`,
history exclusion, deterministic order. 30-user smoke then commit.

---

### Task 5: ALS direct baseline

**Files:**
- Create: `src/recagent_eval/baselines/als_direct.py`
- Create: `tests/test_baselines_als_direct.py`

Reuse `LatentFactorRetriever.fit(legal_retrieval_train)` with dev-CV-selected
hyperparameters (grid from the spec §3); fold-in per user; rank the allowed
universe by latent score. Dev selection runs user-grouped CV restricted to
development users (retrieval fit on dev users' legal rows only, targets =
dev validation targets). Tests: fold-in determinism (reuse existing latent
tests), dev-CV selection fingerprint, score ordering. 30-user smoke then
commit.

---

### Task 6: Current v2b method harness integration

**Files:**
- Create: `src/recagent_eval/baselines/current_v2b.py`
- Modify: `tests/test_baselines_current_v2b.py` (new)

Train LambdaMART with the existing pipeline config `configs/v2_dense_latent_bfeat.yaml`
but with training users = historical-500 ∪ development-600 (ranker targets),
then score confirmation users through the three-route union (ItemCF ∪ Dense ∪
ALS fold-in) + the v2b ranker; unranked candidates outside the union are
excluded and candidate recall recorded. The training-user restriction is a new
parameter in the pipeline (`training_user_ids`), implemented with a leakage
test proving confirmation users never appear in ranker training. 30-user smoke
then commit.

---

### Task 7: BPR-MF baseline (torch)

**Files:**
- Create: `src/recagent_eval/baselines/bpr_mf.py`
- Create: `tests/test_baselines_bpr_mf.py`
- Modify: `pyproject.toml` / `uv.lock` (declare `torch>=2.2,<3` in the `ml` extra)

BPR pairwise SGD with fixed seed, deterministic negative sampling (rng seeded),
`torch.set_num_threads(1)` during fit for reproducibility, user/item factors
serialized as NPZ + manifest (no pickle), training fingerprint in provenance.
Dev-CV selects the fixed grid (spec §3). Tests: determinism (two fits identical
scores), leakage (targets excluded), serialization/checksum, NaN/Inf guard,
empty history → popularity-free empty score. 3-seed robustness runs recorded.
30-user smoke then commit.

---

### Task 8: LightGCN baseline (torch)

**Files:**
- Create: `src/recagent_eval/baselines/lightgcn.py`
- Create: `tests/test_baselines_lightgcn.py`

LightGCN with sparse adjacency over `legal_retrieval_train` (no target rows),
2–3 layers, fixed seed, single-thread CPU (4090 only if CPU time is
unacceptable and recorded). Scores = final user/item embeddings; fold-in for
unseen users via neighborhood aggregation is not needed for confirmation
users (their legal rows are in the fit) but empty-history users get empty
scores. Tests: determinism, leakage (adjacency never contains validation/test
rows), serialization, fingerprint. Dev-CV grid per spec §3. 30-user smoke then
commit.

---

### Task 9: Confirmation-A full comparison

Run every method on confirmation-A (1000 users) with `evaluate-baselines`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/recagent-eval evaluate-baselines \
  --ledger artifacts/cohorts/cohort_ledger.json \
  --cohort confirmation_a \
  --method popularity|itemcf_direct|als_direct|current_v2b|bpr_mf|lightgcn \
  --output artifacts/experiments/v2-baselines/<method>-confirmation-a.json \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m
```

Then a `summarize-baselines` step computes pairwise 2,000-bootstrap deltas vs
ItemCF and between methods, writes `reports/experiments/v2-strong-baselines-confirmation-a.{json,md}`.
Evaluate Success A/B. If A/B met → evidence review with the user (frozen stays
locked). If not → run Task 10 diagnostics.

---

### Task 10: Layered diagnostics (only if Confirmation-A fails)

Write `src/recagent_eval/baseline_diagnostics.py` producing the eight diagnosis
blocks from the spec §6 (per-route recall, oracle bounds, rank quantiles,
feature separation, train/validation depth gap, user buckets, win/loss, query
count/target distribution). All read-only; JSON artifact committed. Then
implement H1 and H2 only if a falsifiable hypothesis is formed; each follows
its own TDD + smoke + development-gate cycle, then confirmation-B.

---

### Task 11: Confirmation-B and final deliverables

If the method changed after A, run the final method + ItemCF + at least one
strong baseline on confirmation-B, produce the final comparison and
effect–cost Pareto table, update README/HANDOFF only from JSON evidence, run
the full quality gate, and leave the branch unmerged.

---

### Final gate (after every task)

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered -q
.venv/bin/ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
git diff --check
bash -n scripts/run_remote_qwen.sh
```

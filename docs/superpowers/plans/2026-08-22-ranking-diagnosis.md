# v2 Ranking Diagnosis and Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the Top-10 ranking bottleneck and evaluate a leakage-safe percentile-calibrated LambdaMART variant while keeping the `semantic.top_k=1500` candidate policy and frozen gate unchanged.

**Architecture:** Add a read-only diagnostic builder over the existing validation query path, then thread an explicit score-calibration mode through candidate feature construction and fingerprints. Reuse the existing grouped CV, bootstrap evidence, artifact bundle, and fail-closed frozen gate.

**Tech Stack:** Python 3.11+, Typer, Pydantic, NumPy, LightGBM, pytest, Ruff, YAML.

---

### Task 1: Add failing tests for calibration and diagnostics

**Files:**
- Modify: `tests/test_v2_ranking.py`
- Create: `tests/test_ranker_diagnostics.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add tests for: route percentile values with deterministic ties; empty routes
returning zero; raw mode preserving current feature values; calibration mode
changing the feature/config fingerprint; diagnostic separation excluding
target-missing users from feature denominators; and the CLI refusing to
overwrite an existing diagnostic artifact.

- [ ] **Step 2: Run focused tests and verify the expected failures**

Run:

```bash
.venv/bin/pytest tests/test_v2_ranking.py tests/test_ranker_diagnostics.py tests/test_config.py -q
```

Expected: collection or assertion failures because the calibration setting and
diagnostic module/command do not exist yet.

### Task 2: Thread explicit score calibration through feature rows

**Files:**
- Modify: `src/recagent_eval/candidate_features.py`
- Modify: `src/recagent_eval/runner.py`
- Modify: `src/recagent_eval/config.py`
- Modify: `src/recagent_eval/lambdamart_pipeline.py`
- Modify: `src/recagent_eval/learned_ranking.py`
- Modify: `tests/test_v2_ranking.py`
- Modify: `tests/test_lambdamart_pipeline.py`

- [ ] **Step 1: Add the configuration field and validation**

Add `score_calibration: str = "raw"` to `ExperimentConfig`; accept only
`raw` or `percentile` under `ranker.score_calibration`; include it in the
candidate-policy and LambdaMART config fingerprints; keep old YAML behavior
unchanged.

- [ ] **Step 2: Implement deterministic route percentile transforms**

Add a helper that maps each finite route score to `(rank_count - rank + 1) /
rank_count`, ties sharing the same rank by score and movie ID order. Missing
route members remain zero. Apply it only to the `itemcf_score` and
`dense_score` feature positions when mode is `percentile`; leave rank,
membership, popularity, history, year, and preference features unchanged.

- [ ] **Step 3: Pass the mode through every legal query builder**

Thread `config.score_calibration` through training, fold, validation replay,
and demo feature construction. Store the selected mode in artifact provenance
and reject mismatched artifacts through the existing fingerprint checks.

- [ ] **Step 4: Run focused tests and refactor only after green**

Run:

```bash
.venv/bin/pytest tests/test_v2_ranking.py tests/test_lambdamart_pipeline.py tests/test_config.py -q
.venv/bin/ruff check src/recagent_eval/candidate_features.py src/recagent_eval/config.py src/recagent_eval/lambdamart_pipeline.py src/recagent_eval/learned_ranking.py
```

Expected: all focused tests pass with raw-mode backward compatibility intact.

### Task 3: Implement the read-only ranker diagnostic artifact

**Files:**
- Create: `src/recagent_eval/ranker_diagnostics.py`
- Modify: `src/recagent_eval/cli.py`
- Create: `tests/test_ranker_diagnostics.py`

- [ ] **Step 1: Implement pure diagnostic aggregation**

Build per-feature target-vs-negative means and AUC-like pairwise separation
only for users whose target is in the candidate union. Report route candidate
recall, target rank quantiles, conditional Top-10 hit/NDCG, score percentiles,
route overlap, and all relevant fingerprints. Keep aggregation deterministic
by sorting user and movie IDs.

- [ ] **Step 2: Add `diagnose-ranker` CLI behavior**

Load the registered cases only for their fingerprint, load the dense cache,
rebuild legal validation queries with `configs/v2_dense_recall1500.yaml`, and
write a new JSON artifact. Refuse existing output paths and never create a
frozen authorization marker.

- [ ] **Step 3: Run diagnostic unit and CLI tests**

Run:

```bash
.venv/bin/pytest tests/test_ranker_diagnostics.py tests/test_cli.py -q
```

Expected: all diagnostics tests pass and overwrite protection is exercised.

### Task 4: Run validation experiments and record evidence

**Files:**
- Create: `configs/v2_dense_recall1500_percentile.yaml`
- Create: `reports/experiments/v2-ranker-diagnostics.json`
- Create: `reports/experiments/v2-ranker-diagnostics.md`
- Create: `reports/experiments/v2-dense-lambdamart-recall1500-percentile.json`
- Create: `reports/experiments/v2-dense-lambdamart-recall1500-percentile.md`

- [ ] **Step 1: Run the diagnostic command**

Use the real MovieLens data and dense cache with `max-users=500`; save output
under a fresh `artifacts/experiments/v2-ranker-diagnostics/` directory.

- [ ] **Step 2: Run a 30-user percentile stability training**

Publish model/evidence/bundle under fresh `v2-recall-1500-percentile-30/`
paths. A crash or failed gate is recorded; it never unlocks frozen evaluation.

- [ ] **Step 3: Run the 500-user percentile validation**

Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and fresh
`v2-recall-1500-percentile/` artifact paths. Copy all metrics and fingerprints
from the JSON evidence into the Markdown report.

- [ ] **Step 4: Compare against the existing raw/top1500 evidence**

Declare promotion only if NDCG exceeds ItemCF, bootstrap lower bound is
positive, and constraints are exactly 100%. Otherwise preserve the negative
result and keep frozen evaluation locked.

### Task 5: Full verification and documentation reconciliation

**Files:**
- Modify: `README.md`
- Modify: `docs/HANDOFF-2026-08-22.md`
- Modify: interview/report files only when JSON evidence supports a change

- [ ] **Step 1: Run the complete quality gate**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered -q
.venv/bin/ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
git diff --check
bash -n scripts/run_remote_qwen.sh
```

- [ ] **Step 2: Reconcile current counts and claims**

Update stale test-count badges and ranking conclusions only from checked JSON;
do not claim frozen or Qwen results that were not produced.

- [ ] **Step 3: Inspect status and leave the branch unmerged**

Run `git status --short --branch`, verify no credentials, taste fields, model
weights, or accidental large caches are tracked, and leave merge/PR choice to
the user.

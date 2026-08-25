# RecAgent-Eval v2 Strong-Baseline Evaluation Design (2026-08-23)

Status: **pre-registered before any baseline run**. Fixes the evaluation
protocol, cohorts, baselines, hyperparameter selection, confirmation rules,
and success criteria. Frozen test stays locked; no LLM API involvement in
offline ranking; resume/promo material is out of scope.

## 1. Objective

Answer, with untouched confirmation cohorts, whether the current best learned
method (`v2b + all negatives`) is a credible recommendation result:

1. Does it stably beat ItemCF on fresh users?
2. Does it beat at least one strong collaborative baseline (ALS direct or
   BPR-MF)?
3. Against LightGCN (preferred modern baseline), does it lead in effect or
   form a quantified effect–cost Pareto advantage?
4. If it fails: attribute the limitation to candidate recall, training
   samples, ranking objective, feature representation, or evaluation variance.

## 2. Unified protocol (identical for every method)

### 2.1 Cohorts (fixed seed 42, mutually exclusive, recorded in a ledger)

- Pool: eligible users (3 targets) minus the historical-500 selection cohort,
  minus the 44 frozen-case users not already in historical-500 → 5491 users.
- Assignment: `random.Random(42).shuffle(pool)` then consecutive slices.
- development (600): hyperparameter selection via user-grouped CV on
  validation targets. Once used for selection, dev is permanently development
  evidence.
- confirmation-A (1000): first untouched comparison. If any algorithm changes
  after A is read, A downgrades to development evidence.
- confirmation-B (1000): never read until the final method is frozen; only B
  can certify post-A changes.
- reserve (2891): untouched this phase.
- frozen-test (50 cases): locked, never read.

All cohort lists, their SHA-256 fingerprints, and the draw code are recorded
in `artifacts/cohorts/cohort_ledger.json` (gitignored) and a committed copy
under `reports/audit/`.

### 2.2 Data boundaries for every run

- Legal history, train target, validation target, frozen target per user:
  exactly as `leakage_safe_ranking_split` defines (see Phase 0 audit §5).
- Confirmation evaluation labels: **validation targets only**.
- Retrieval fits: `legal_retrieval_train` (excludes every user's validation
  and test target movie IDs; includes ranker targets) — same boundary the
  existing pipeline uses.
- Ranker (LambdaMART v2b) training: ranker-target queries of the **training
  cohort = historical-500 ∪ development-600**. Confirmation-A/B users are
  never in ranker training.
- Baseline hyperparameter selection: user-grouped CV restricted to
  development users; confirmation users never appear in any tuning fit.

### 2.3 Item universe, candidates, and ranking

- Item universe: all 3,883 movies.
- Allowed universe per user: `hard_filter(movies, state) - history` (same
  filter and history exclusion for every method).
- Candidate set per method:
  - Popularity / ItemCF / ALS direct / BPR-MF / LightGCN: the full allowed
    universe scored by the model.
  - LambdaMART v2b: its fixed three-route union
    (ItemCF top-500 ∪ Dense top-1500 ∪ ALS top-500), scored by the ranker;
    candidates outside the union are unranked (candidate recall reported).
- Top-K: K=10 everywhere; tie-break `(-score, movie_id)` for every method.
- Seeds: 42 for all data assignment and model initialization; LightGBM
  deterministic, ALS deterministic, BPR/LightGCN fixed seed with
  `torch.manual_seed`; determinism verified by double-fit tests.

### 2.4 Metrics (same implementation for every method)

- Recall@10, NDCG@10, MRR@10, coverage (fraction of distinct items
  recommended across the cohort), constraint satisfaction (all recommendations
  inside the allowed universe and disjoint from history).
- Candidate/oracle recall (per-method candidate set).
- p50/p95 inference latency per user, training wall time, peak memory (RSS
  measured via `resource`/`psutil`-free wrapper), model size on disk, and
  dependency/hardware environment recorded in the JSON artifact.
- Paired statistics: user-level paired deltas, 2,000 paired bootstrap
  resamples with seed 42, saving point estimate, delta, and 95% CI for every
  method vs ItemCF and vs every other method.
- Non-deterministic models (BPR/LightGCN) report 3 seeds; results never pick
  the best seed — the reported value is seed 42 primary plus 2 robustness
  seeds recorded separately.

## 3. Baselines and parameter grids (pre-registered, finite)

All hyperparameter selection happens only on development users with
user-grouped CV (same splitter semantics as the existing grouped CV). The
grids are small and fixed now; no grid grows after results are seen.

| Method | Implementation | Fixed grid (dev-CV selected) |
| --- | --- | --- |
| Popularity | global positive-count ranking | none (no parameters) |
| ItemCF | existing `ItemCFRetriever` | `top_k ∈ {500}` (fixed; score ordering of full universe) |
| ALS direct | existing `LatentFactorRetriever` fold-in | `rank ∈ {20, 40}`, `iterations ∈ {10, 12}`, `alpha ∈ {20, 40}`, `lambda_reg ∈ {0.05, 0.1}` |
| Current method | LambdaMART v2b + all negatives (existing pipeline) | existing 16-param grid (unchanged), dev-CV selection on training cohort |
| BPR-MF | torch SGD pairwise (new module) | `rank ∈ {16, 32}`, `lr ∈ {1e-3, 5e-3}`, `reg ∈ {1e-4, 1e-3}`, `epochs ∈ {10, 20}` |
| LightGCN | torch graph CF (new module) | `rank ∈ {32, 64}`, `layers ∈ {2, 3}`, `lr ∈ {1e-3, 5e-3}`, `reg ∈ {1e-4, 1e-3}` |

LightGCN choice rationale (fixed before running): the audit shows the protocol
is chronological *leakage* (train only on interactions before the target), not
a requirement that the model consume order. LightGCN trained on
`legal_retrieval_train` satisfies the no-future-leakage constraint and is the
specified modern graph baseline. SASRec is the fallback only if LightGCN's
adjacency construction is found to violate the protocol during implementation
(must be re-justified and fixed in the plan before any run).

BPR-MF and LightGCN are implemented with torch (already in the locked venv),
declared explicitly in the `ml` extra; no `implicit`/`dgl` dependency.

## 4. Confirmation rules

1. Run every method on confirmation-A. Report all pairwise bootstrap results.
2. If the current method is changed as a result of A, A becomes development
   evidence; the changed method's final claim must come from confirmation-B.
3. If no algorithm change is needed, B is used to certify the A conclusions.
4. Frozen test remains locked until the user authorizes consumption.
5. No re-selecting users/seeds/metrics/thresholds after results are seen.

## 5. Success criteria (pre-registered, from the task brief)

**Success A (effect win)**: on an untouched confirmation cohort, the current
method has NDCG@10 > ItemCF with paired-bootstrap 95% CI lower bound > 0,
beats at least one of ALS direct / BPR-MF with the same CI rule, Recall@10
does not regress materially, and constraints stay 100%.

**Success B (Pareto win)**: if not ahead of the strongest neural baseline,
NDCG@10 within 5% relative of it, plus a quantified advantage in training
time, inference latency, model size, or CPU-only runnability; constraints
100%; explainable feature contributions and the Agent toolchain remain usable.

Only A or B qualifies the project as an "algorithmically competitive" result.

## 6. Failure path (Confirmation-A still fails)

Produce the layered diagnosis (per-route recall, oracle bounds, target rank
quantiles, feature separation, train/validation depth gap, per-user buckets,
win/loss vs ALS/BPR, query-count and target-distribution checks). Then at most
two hypothesis-driven rounds:

- **H1 rolling temporal training samples**: multiple historical cutoffs per
  development user, each query reading only pre-cutoff interactions, target =
  next positive after cutoff, user-grouped CV, confirmation users never in
  training, all-negatives or an independently validated unbiased sampler
  (route-balanced hard negatives remain falsified), counts/cutoffs/weights and
  fingerprints fixed.
- **H2 simple direct fusion**: deterministic reciprocal-rank or low-dimensional
  linear fusion of ALS direct + ItemCF + v2b learned score, weights selected
  only on development users, stored as a formal method with config,
  fingerprint, and explanation fields.

Each hypothesis needs a falsification criterion, TDD, a 30-user smoke, a
development-cohort gate, and only then Confirmation-B. Two failed rounds end
algorithm expansion; negative results are preserved.

## 7. Engineering requirements

- TDD for every feature/bugfix (RED → GREEN), determinism tests, leakage
  tests, user-grouped split tests, serialization/checksum tests, fingerprint
  tests, NaN/Inf/empty-history tests, CLI refuse-overwrite tests.
- Coverage stays ≥ 90%; final gate: pytest, coverage, Ruff, uv lock --check,
  git diff --check, bash -n.
- New artifacts in new directories, refusing overwrite. Summary JSON +
  Markdown + configs + commands committed; large artifacts gitignored.
- No lowering thresholds, deleting tests, or changing metric definitions to
  force a pass; historical negative results never overwritten.

## 8. Out of scope this phase

- Frozen test consumption, resume/STAR/PPT/README promotion updates, LLM API
  use for offline ranking, Qwen smoke, Demo polish, dense-model swaps until
  the diagnostic path justifies them.

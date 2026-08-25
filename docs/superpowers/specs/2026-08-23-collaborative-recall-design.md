# RecAgent-Eval v2 Collaborative Recall Design (2026-08-23)

Status: **draft, awaiting user confirmation**
Branch: `feat/dense-recall-v2` (worktree `.worktrees/agent-search-v2`)

## 1. Objective

Make the 500-user validation gate produce a trustworthy positive result by
fixing the ranking-depth bottleneck: the validation target is in the candidate
union for 87.8% of users, but its median rank inside the union is ~172, so only
~7% of present users can reach Top-10.

The primary route this design chooses is **Plan A: a deterministic,
numpy-based collaborative latent-vector retriever (weighted ALS, with
`threadpoolctl` added as a direct dependency) as a third candidate source**,
followed by a small, pre-selected set of **Plan B discriminative features**
(latent score/rank features first; cross/recent/year features as a contingency
variant). **Plan C (larger dense encoders, LLM
profile rewrite, GPU) is explicitly out of scope** for this round, per the
approved boundaries in the task brief.

This document freezes the pre-registered candidate-stage gates and the
experiment matrix before any production code is written. No production code,
dependency change, or formal 500-user run happens before the user confirms this
design.

## 2. Evidence audit (read from artifacts, not from memory)

### 2.1 Repository state

- Worktree: `/Users/shinjuu/intern/recagent-eval/.worktrees/agent-search-v2`,
  branch `feat/dense-recall-v2`, `git status` clean before this uncommitted
  spec was created (the spec itself is the only untracked file), 3 commits
  ahead of `origin/feat/dense-recall-v2`.
- Verified gate: 250 tests pass (`exit=0`), line coverage 90.1%
  (3467 statements, 343 missed), `ruff check .` clean, `git diff --check`
  clean.
- Data lives in the main checkout:
  `/Users/shinjuu/intern/recagent-eval/data/raw/ml-1m`
  (3,883 movies, 1,000,209 ratings). All CLI runs pass `--data-dir` explicitly.
- Dense cache (gitignored): `artifacts/embeddings/movielens-minilm.npz` +
  `.json`, `all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, dataset fingerprint
  `c79c4d31b8f4b0c48440281a020de95ea5b878ba6373e4b1134ec206ae40c792`.

### 2.2 Data split (leakage-safe, unchanged by this design)

`leakage_safe_ranking_split` gives each eligible user three disjoint targets
(ranker, validation, test) from the three latest distinct positive movies,
sorted chronologically. Per user:

- `histories`: rows before the ranker target, excluding all three target
  movie IDs (used for ranker-training queries).
- `legal_retrieval_train`: rows before the validation target, excluding the
  validation and test movie IDs but **including the ranker target** (used to
  fit retrieval for validation rows and frozen replay).
- `ranker_targets` / `validation_targets` / `test_targets`: disjoint, never
  crossed.

### 2.3 Current candidate policy and feature schema (v1)

- ItemCF: `retrieval_top_k=500`, fit on the legal rows of the current context.
- Dense: `semantic.top_k=1500` (recall-1500 policy), query built by
  `build_semantic_profile("", state, movies, history_cap=50)`.
- Union = ItemCF ∪ Dense, then hard-filtered and minus history.
- Feature schema `candidate-features/v1`, 10 features:
  `itemcf_score`, `itemcf_reciprocal_rank`, `dense_score`,
  `dense_reciprocal_rank`, `log1p_popularity`, `history_genre_jaccard`,
  `history_year_match`, `preference_affinity`, `in_itemcf`, `in_dense`.
  Feature fingerprint `2a4a8822...`.
- LambdaMART: `LGBMRanker` (`n_jobs=1`, deterministic, single-threaded OMP
  guard), 16-parameter grid × whole-user 3-fold GroupKFold, final fit on all
  training queries, 2,000 paired bootstrap resamples, constraint
  recomputation, fail-closed frozen gate.

### 2.4 Experiment evidence (each row is a distinct config; never merged)

Fixed provenance: dataset fingerprint `0d2c756a...`, case fingerprint
`bc2f622c...`, seed 42, 2,000 resamples.

| Run | Config / policy | Union / ItemCF / Dense recall | LambdaMART vs ItemCF NDCG@10 | Bootstrap 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| v2-500 | `v2_dense_validation.yaml`, `semantic.top_k` unset (500) | 0.776 / 0.696 / 0.288 | 0.0327 vs 0.0334 | [−0.0146, 0.0129] | gate failed |
| recall-1500 | `v2_dense_recall1500.yaml`, `semantic.top_k=1500` | 0.878 / 0.696 / 0.612 | 0.0299 vs 0.0334 | [−0.0190, 0.0111] | gate failed |
| percentile | `v2_dense_recall1500_percentile.yaml`, `score_calibration=percentile` | 0.878 / 0.696 / 0.612 | 0.0207 vs 0.0334 | [−0.0266, 0.0004] | falsified, worse |

Recall-1500 model checksum `3e12b627...`; evidence fingerprint
`ed0e79d9...`; percentile evidence fingerprint `be72a00f...`.

### 2.5 Ranking diagnostics (recall-1500 policy, 500 users, 439 present)

Read from `reports/experiments/v2-ranker-diagnostics.json`:

- Target rank quantiles (rank within the ~2,000-item union, sorted by route
  score; route-absent members score 0):
  - ItemCF order: p25 = 53, **p50 = 172**, p75 = 430.
  - LambdaMART order: p25 = 57, p50 = 173, p75 = 374.
- Top-10 hit on present users: ItemCF 0.073, LambdaMART 0.075; NDCG@10 on
  present users: ItemCF 0.0380, LambdaMART 0.0341.
- Feature separation (target mean minus negative mean, present users):
  `itemcf_score` +8.80, `log1p_popularity` +2.08, `in_itemcf` +0.51, all
  others near zero; `dense_score` −0.05 and `in_dense` −0.15 are
  anti-predictive.

Interpretation: the bottleneck is the **quality/depth of the route
orderings**, not union coverage. The learned ranker only re-ranks within the
union; it cannot surface a target that both base routes rank at ~172. The
percentile experiment already proved that changing score calibration alone
destroys the only strongly separating feature.

### 2.6 Traceability findings (must be recorded)

1. The recall-1500 **evidence** config fingerprint `7b9373b4...` was computed
   by the pre-`e1efee8` fingerprint payload (before `score_calibration` was
   added). The current code computes `3c0abb8c...` for the same YAML. The
   **diagnostics** artifact (`3c0abb8c...`) matches current code. Consequence:
   the recall-1500 raw bundle is not replayable under current code even if the
   gate had passed. The negative result stays valid evidence, but its recorded
   config fingerprint must be labeled with the fingerprint version that
   produced it. New artifacts in this round are generated under current code.
2. The percentile **summary JSON** (`v2-dense-lambdamart-recall1500-percentile.json`)
   contains a cosmetic `score_calibration: "raw"` field while its config and
   evidence fingerprints prove percentile mode. The authoritative numbers are
   in `artifacts/experiments/v2-recall-1500-percentile/validation.json`. New
   summaries must not repeat this kind of mismatch.
3. Old v1 artifacts must keep parsing. Any schema/fingerprint change must be
   backward compatible at the default config (see §4.4). Task 0 (§10)
   implements the summary correction and the legacy fingerprint marking.

## 3. Option comparison

### Plan A — Collaborative latent-vector recall (ALS / BPR / SVD)

Add a third candidate source learned from the same legal interaction history:
ALS item factors, with every user scored by the standard weighted-ALS fold-in
from their own legal history; output top-k latent candidates plus latent
score/rank/membership features for LambdaMART.

- ALS (implicit feedback, Hu-Koren-Volinsky): numpy-based implementation with
  `threadpoolctl` declared directly; deterministic ranked rows in the same
  pinned environment, without a cross-platform bit-exact claim; strong on
  MovieLens-scale implicit feedback.
- BPR-MF: directly optimizes pairwise ranking (closest to our objective) but
  uses SGD with negative sampling; byte-level reproducibility requires careful
  seed/order discipline and is harder to defend than ALS.
- Plain SVD of the binary matrix: zeros dominate; needs normalization tricks;
  expected to behave like a weaker ItemCF. Not worth it.

Pros: different inductive bias than the co-occurrence ItemCF (global
low-rank structure, no popularity-squared denominator), so it can both (a)
recover targets ItemCF/Dense miss (complementary coverage) and (b) provide a
genuinely new separating score feature instead of a transform of the same two
weak scores. CPU-only, no external data, no license issues; fit cost is
measured in the E3 benchmark rather than asserted.

Cons: still collaborative-only (same information source, different model);
quality depends on hyperparameters; ALS is not guaranteed to beat ItemCF at
Top-10 depth, which is exactly why we pre-register a candidate-stage gate and
fail fast.

### Plan B — Sequence and cross features only

Add discriminative features (recent-history affinity, genre transition,
year-distance buckets, popularity buckets, route agreement/cross terms) to the
existing LambdaMART without a new retrieval route.

Pros: cheap, leakage-safe by construction, no new model.

Cons: with median rank ~172 in **both** base orderings, no feature can move
the target into the union's top 10 unless the base scores themselves improve;
the percentile experiment is direct evidence that ranker-input changes alone
do not lift NDCG. B is a complement, not a fix for ranking depth.

### Plan C — Stronger dense / LLM / GPU

Pros: could add semantic signal beyond title/genre/year.

Cons: MovieLens-1M metadata is minimal; `all-MiniLM-L6-v2` already has
negligible/anti-predictive separation here, so the bottleneck is not embedding
quality. External descriptions raise licensing/caching/fairness questions.
LLM randomness breaks offline replay. A 4090 does not change the fact that the
target sits at rank ~172 in the score-based orderings. Rejected for this
round.

### Recommendation

**A first (ALS latent route), then a small B set.** Concretely:

1. ALS implicit-feedback latent retriever with standard fold-in scoring
   (numpy-only, deterministic, item factors persisted), `top_k=500`, output
   `latent_score` / `latent_reciprocal_rank` / `in_latent` features
   (schema v2, 13 features).
2. If the candidate-stage gates pass but the formal gate still fails, evaluate
   one B-feature variant (schema v2b, +3 features: `itemcf_latent_cross`,
   `recent_itemcf_score`, `year_recency`) as a single contingency, then stop.
3. C is not run in this round.

Why A is most likely to fix "median rank 172": the rank is a property of the
route score orderings. ALS produces a score function with a genuinely
different bias (global factors, no co-occurrence/popularity denominator), so a
material fraction of users should see the held-out positive move up hundreds
of positions; even a modest median-rank improvement plus a new separating
feature is exactly the input LambdaMART was missing. The pre-registered gates
in §9 make the bet falsifiable at the candidate stage, before ranker training.

## 4. Recommended design

### 4.1 Latent route: `LatentFactorRetriever` (weighted ALS with fold-in)

New module `src/recagent_eval/latent_retrieval.py` (keeps `retrieval.py` from
growing further):

```python
@dataclass(frozen=True)
class LatentFactorRetriever:
    item_factors: np.ndarray            # (n_items, rank), float32
    item_ids: np.ndarray                # sorted int64
    rank: int
    iterations: int
    alpha: float
    lambda_reg: float
    seed: int
    training_fingerprint: str           # hash of (params, training rows, seed)

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
    ) -> LatentFactorRetriever: ...

    def retrieve(
        self,
        history: set[int],
        *,
        top_k: int = 100,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]: ...

    def save(self, path: Path) -> None: ...   # NPZ + JSON manifest, safe fd/atomic
    @classmethod
    def load(
        cls, path: Path, *, expected_fingerprint: str | None = None
    ) -> LatentFactorRetriever: ...
```

Fit (training):

- Implicit matrix `r_ui = 1` for `rating >= 4`, confidence
  `c_ui = 1 + alpha * r_ui`; alternating least squares for `iterations`
  rounds, user and item order strictly sorted by ID; initial factors from
  `np.random.default_rng(seed)`; per-round least squares solved with
  `np.linalg.solve`.
- The training loop produces item factors `Y` (transient user factors `X` are
  **never persisted and never used for scoring**).
- Fit runs inside `threadpoolctl.threadpool_limits(1)` (declared direct
  dependency, see §4.4); `np.set_num_threads` is not used.

Scoring (all contexts, including grouped-CV validation users):

- Standard weighted-ALS fold-in from the user's **legal history**:
  ```text
  x_u = (YᵀY + Yᵀ(C_u − I)Y + λI)⁻¹ · YᵀC_u p_u
  score(u, i) = x_uᵀ y_i
  ```
  with `p_u = 1` on observed positives and `C_u = diag(1 + α·r_ui)`. The fixed
  `YᵀY` term is precomputed; the per-user delta is
  `Σ_{i∈history} α·(y_i y_iᵀ)`.
- CV validation users are fold-in scored from their own legal history only;
  no user factor saved from the training fit is ever used for scoring.
- History movies absent from `item_ids` (fold-training unseen items) are
  ignored; if no valid history remains, `retrieve` returns `[]` (empty route →
  zero latent features), matching the existing route convention.
- Scores are raw dot products, finite-checked, deterministic tie-break
  `(-score, movie_id)`, `top_k` validated, `allowed_ids` masked. No popularity
  fallback.

Persistence:

- The **final** item factors used for the 500-user validation rows (and the
  frozen replay) are fit on `legal_retrieval_train` and published as NPZ + JSON
  manifest using the project's existing safe-fd/atomic-replace style
  (mirroring `DenseSemanticRetriever.save`): schema version, params, training
  data fingerprint, sorted item IDs, shape, dtype, checksum, created_at,
  runtime metadata. No pickle.
- Grouped-CV fold models are temporary in-memory fits and are never persisted.
- The latent artifact checksum is bound into the ranker artifact, the
  evidence, and the bundle manifest (§4.4).
- Hyperparameters are **fixed** for this round: `rank=20, iterations=12,
  alpha=40, lambda_reg=0.1, seed=42`. No 30-user micro-grid; the 30-user smoke
  only verifies stability, wall time, artifact replay, and constraints. Any
  future ALS tuning may only use ranker targets through grouped CV, never the
  validation users.

### 4.2 Feature schema v2 (13 features)

`FEATURE_NAMES_V2 = FEATURE_NAMES + ("latent_score", "latent_reciprocal_rank",
"in_latent")`, schema version `candidate-features/v2`, new fingerprint.
`latent_score` keeps raw magnitude (the percentile experiment proved raw scale
matters); `latent_reciprocal_rank` mirrors the other routes; `in_latent` is the
membership bit. All three are computed from the latent route's top-500 list,
zero for absent movies, finite-checked like today.

Contingency schema v2b (only if §9 formal gate fails after a passing candidate
stage; never in the first 500-user main run):

- `itemcf_latent_cross = itemcf_score * latent_score` (interaction term; tree
  models learn multiplicative relations poorly).
- `recent_itemcf_score`: ItemCF score computed from the user's 10 most recent
  positive ratings only, via a new `ItemCFRetriever.score_many(history,
  movie_ids)` helper on the same legal fit (captures temporal drift).
- `year_recency`: `|candidate.year - median(year of recent positives)|` when
  both are present, else 0 (numeric, not the existing binary year match).

Each feature's data source and leakage boundary is documented in the code
comment and in §7. Feature count grows from 10 → 13 (→ 16 only in the
contingency variant), with every feature attributable to legal inputs.

### 4.3 Route-balanced hard-negative sampling (training only)

`build_training_matrix` already accepts `max_negatives` but slices by movie ID
order. The main run replaces that with a fixed, route-balanced policy so the
latent route's hardest negatives are not missed:

- Config: `ranker.max_negatives=200`, `ranker.negative_policy="route_balanced"`
  for the main run (E4). `negative_policy="all"` keeps today's behavior for
  compatibility and controls.
- Per query, negatives are built as:
  1. top 100 negatives by `itemcf_score` (desc, movie_id asc);
  2. top 100 negatives by `latent_score` (desc, movie_id asc);
  3. deduplicated, then topped up to 200 by descending
     `max(itemcf_reciprocal_rank, latent_reciprocal_rank)` (tie-break movie_id);
  4. final stable order: pre-score `max(itemcf_reciprocal_rank,
     latent_reciprocal_rank)` desc, then movie_id asc.
- The target (the single positive) is always kept. The sampler is a pure
  function of each query's own feature rows, so fold and final matrices use
  identical sampling with zero extra state.
- Policy name, quotas (100/100/200), and the top-up rule are folded into the
  config fingerprint and the matrix group fingerprint payload, so artifacts
  from different sampling policies cannot mix.
- Dense stays in the union and in the feature rows (its score feature remains
  available for the ranker to learn), but receives no fixed hard-negative
  quota.

### 4.4 Config, fingerprints, compatibility

- New YAML block:
  ```yaml
  latent:
    enabled: false              # default false
    rank: 20
    iterations: 12
    alpha: 40.0
    lambda_reg: 0.1
    top_k: 500
    seed: 42
    artifact_path: artifacts/experiments/<run>/latent.npz   # required when enabled
  ```
  plus `ranker.max_negatives`, `ranker.negative_policy` (`all` | `itemcf` |
  `itemcf_latent` | `route_balanced`), optional `ranker.negative_pre_score`
  (used only by the single-route policies; the main run uses
  `route_balanced`), and `ranker.feature_version` (`v1` default; `v2` / `v2b`
  when latent is on).
- Schemas are versioned and strictly dispatched by `schema_version`; v1 and v2
  artifacts/evidence/bundles can never mix:
  - `lambdamart-artifact/v1` + `lambdamart-validation/v1`: unchanged, produced
    when latent is disabled; old fingerprints preserved byte-for-byte.
  - `lambdamart-artifact/v2`: adds `feature_schema_version=candidate-features/v2`
    (or v2b), `latent_artifact_checksum`, and `latent_provenance` (params,
    `training_fingerprint`, top_k, artifact path).
  - `lambdamart-validation/v2`: adds the same latent provenance and latent
    artifact checksum.
  - `lambdamart-bundle/v2`: adds `latent_sha256` and the latent manifest
    checksum to the existing manifest fields; `publish_ranker_bundle` /
    `load_ranker_bundle` gain an optional latent member and validate schema
    version strictly. A v2 artifact with v1 evidence (or any other mix) fails
    closed.
- `candidate_policy_fingerprint` / `lambdamart_config_fingerprint`:
  when `latent.enabled=false` the payload is **byte-identical to today**
  (schema `union-candidate-policy/v1`), preserving default-config
  fingerprints; when enabled, the payload bumps to `/v2` and includes the
  latent block, latent `top_k`, feature version, and sampling policy.
- `ExperimentConfig` gains the latent fields with `latent_enabled=False`
  default, so `configs/v2_dense_validation.yaml` and
  `configs/v2_dense_recall1500.yaml` keep their exact current behavior.
- `threadpoolctl` is declared as a **direct dependency** (small, pure-Python)
  so the latent fit can run under `threadpoolctl.threadpool_limits(1)`;
  `np.set_num_threads` is not used (not a stable public API). The project
  `uv.lock` is updated with it; the CI lock check stays mandatory.
- The reproducibility claim is scoped: with fixed dependency versions in the
  same environment, ranked rows and fingerprints are identical; cross-platform
  bit-exactness is not promised, and runtime/dependency metadata is recorded in
  every manifest.

### 4.5 Determinism and replay

- Latent fit: fixed seed, sorted IDs, fixed iteration order,
  `threadpoolctl.threadpool_limits(1)` around the fit. Ranking tie-breaks are
  `(-score, movie_id)` everywhere (existing convention). A regression test fits
  twice and asserts identical scores within the test environment.
- The final latent artifact (item factors) is persisted and **loaded**, not
  refit, during frozen replay, so replay rows depend on exact stored bytes plus
  deterministic retrieval rather than on re-running the fit.
- The 30-user smoke publishes a bundle and the CLI replay path
  (`evaluate-ranker`'s `build_validation_rows` comparison) must reproduce the
  recorded rows exactly before any 500-user run is trusted.
- Fit/retrieve wall time is measured in the 30-user benchmark and recorded in
  the diagnostics artifact before the E1/E4 time budgets are finalized; no
  "seconds per fit" claim is made in advance.

## 5. Data flow

```text
config (latent.enabled + params, feature_version, sampling policy)
  -> legal rows for the current context (see §7)
  -> LatentFactorRetriever.fit(legal rows)          # deterministic, threadpool-limited
  -> [final context] publish latent NPZ + JSON manifest; checksum -> bundle v2
  -> fold-in retrieve(history, top_k=latent.top_k, allowed_ids)
  -> union = itemcf(top500) ∪ dense(top1500) ∪ latent(top500)
  -> feature rows (v1 | v2 | v2b, schema+fingerprint checked)
  -> grouped CV / route-balanced hard-negative matrix
  -> final fit + validation rows + 2,000 paired bootstrap
  -> candidate_policy/config/feature/latent fingerprints bound into bundle v2
  -> formal gate (unchanged) -> frozen replay loads the persisted latent artifact
```

The ItemCF baseline for the gate is computed inside the same validation-row
builder over the same union (non-ItemCF members score 0), exactly as today, so
"higher than the fixed ItemCF baseline" has the same meaning it always had.

## 6. Module interfaces

### 6.1 `latent_retrieval.py` (new)

- `LatentFactorRetriever` (§4.1): ALS fit producing item factors only, fold-in
  scoring, NPZ + JSON manifest save/load with checksum validation, no pickle.

### 6.2 `retrieval.py`

- Add `ItemCFRetriever.score_many(history, movie_ids) -> dict[int, float]`
  (only used by the v2b contingency; same legal fit).

### 6.3 `candidate_features.py`

- Keep v1 constants untouched.
- Add `FEATURE_NAMES_V2`, `FEATURE_SCHEMA_VERSION_V2`,
  `FEATURE_NAMES_V2B`, `FEATURE_SCHEMA_VERSION_V2B` and matching fingerprints.
- `build_candidate_feature_rows(..., latent_scores=None,
  feature_version="v1", recent_itemcf_scores=None)`; feature length and
  finiteness checks are schema-aware.

### 6.4 `config.py` / `runner.py`

- `ExperimentConfig` gains the §4.4 fields (defaults keep current behavior).
- `load_experiment_config` validates `latent.top_k > 0`, `latent.artifact_path`
  required when enabled, feature version ∈ {v1, v2, v2b},
  `max_negatives ≥ 0 or None`, `negative_policy` ∈ {all, itemcf,
  itemcf_latent, route_balanced}; rejects a v2 feature version without
  `latent.enabled`.

### 6.5 `learned_ranking.py`

- `build_training_matrix(..., max_negatives=None, negative_policy="all")`
  implements the route-balanced sampling (§4.3); `negative_policy="all"` keeps
  today's behavior exactly.
- `LearnedRanker` stores the artifact's feature version; `rank(...,
  latent_scores=None)` and `rank_feature_rows` are schema-aware; v1 callers
  are unchanged.
- `RankerArtifact.validate_contract` / `parse_ranker_artifact` dispatch on
  `schema_version`: v1 validates against the v1 constants exactly as today;
  v2 validates strictly against v2 fields including `latent_artifact_checksum`
  and `latent_provenance`. v1/v2 mixing fails closed.

### 6.6 `lambdamart_pipeline.py`

- `build_candidate_queries(..., latent: LatentFactorRetriever | None = None)`
  threads fold-in latent scores into feature rows; `build_fold_queries` uses
  temporary fold fits, `build_validation_rows` uses the persisted final latent
  artifact, and `train_lambdamart_pipeline` fits/publishes per context (§7).
- `candidate_policy_fingerprint` / `lambdamart_config_fingerprint` per §4.4.

### 6.7 `bundle.py`

- `publish_ranker_bundle` / `load_ranker_bundle` gain optional latent member
  support for `lambdamart-bundle/v2` (`latent_sha256` + latent manifest
  checksum) and validate schema version strictly.

### 6.8 `v2_selection.py`

- Evidence schema v2 adds latent provenance and the latent artifact checksum;
  gate math and fail-closed behavior unchanged; v1 evidence validation
  unchanged.

### 6.9 `latent_diagnostics.py` (new) / `ranker_diagnostics.py`

- New `diagnose-latent` CLI (read-only, validation users only, refuses to
  overwrite existing output): latent route recall@10/50/100/500 (unconditional
  and route-present), target coverage, latent-list rank quantiles (present
  users), latent-only coverage, overlap with ItemCF/Dense, fit/retrieve wall
  time, and fingerprints. Reuses `build_candidate_queries` with latent enabled.

### 6.10 `ranker_selection.py` (frozen path, v2 only)

- `evaluate_frozen_cases(..., latent_retriever=None)` gains an optional latent
  source that **loads the persisted artifact** so a v2 frozen run replays
  identically; the v1 default path is unchanged.

## 7. Anti-leakage design

| Context | Rows used to fit ItemCF / latent | Histories used for queries | Targets seen |
| --- | --- | --- | --- |
| ranker-training queries | `split.ranker_training_history` | `split.histories` | `ranker_targets` |
| CV fold train/val queries | fold-train subset of `ranker_training_history` (temporary fold fits) | per-fold `histories` | `ranker_targets` only |
| validation rows + diagnostics | `split.legal_retrieval_train` (final artifact fit, then persisted) | positive histories of legal rows | `validation_targets` only |
| frozen replay/consumption | loads the persisted final latent artifact (fit rows: `split.legal_retrieval_train`) | positive histories of legal rows | registered frozen cases only |

Additional rules:

- Hard negatives are sampled from each user's own legal candidate union; the
  sampler never reads validation/test targets or future rows.
- `allowed_ids` = hard-filter minus history (existing), applied identically to
  all three routes.
- Scoring is always **fold-in from the user's own legal history**; user
  factors saved by the training fit are never used for scoring, so a grouped-CV
  validation user can never benefit from a fit that saw their ranker target.
- Feature schema and candidate-policy fingerprints are versioned; v1 default
  artifacts cannot silently mix with v2 artifacts.
- Only the final latent item-factor artifact is serialized (NPZ + manifest,
  no pickle); fold models are temporary. The artifact checksum binds the
  ranker artifact, evidence, and bundle manifest, and frozen replay loads the
  exact stored bytes.
- No user subsetting, seed changing, difficult-user removal, metric
  redefinition, or gate adjustment is allowed; `max_users=500` (first 500
  sorted users), `seed=42`, fixed case fingerprint `bc2f622c...` throughout.

## 8. Experiment matrix

| ID | Stage | Users | Config / variant | Output / artifact | Decision gate |
| --- | --- | --- | --- | --- | --- |
| E0 | freeze | — | existing raw/percentile/ItemCF | none (never overwritten) | — |
| E1 | candidate-only | 500 | latent single-route diagnosis (ItemCF/Dense fixed, latent on) | `artifacts/experiments/v2-latent-diagnostics/diagnostics.json` + report | evidence only |
| E2 | candidate-only | 500 | three-route union diagnosis (same run or E1 section) | same artifact, `three_route` block | **G1–G4** |
| E3 | smoke | 30 | full pipeline: fixed ALS hyperparameters (§4.1), route-balanced hard negatives, latent artifact persisted | `v2-latent-hardneg-30/` bundle (incl. latent NPZ + manifest) | stability + wall-time benchmark + exact replay + constraints 100% |
| E4 | ranker | 500 | **main**: schema v2 latent features, raw calibration, `max_negatives=200`, `negative_policy=route_balanced` | `v2-latent-hardneg-500/` bundle v2 + report | **formal gate** |
| E5 | ranker | 500 | optional control: schema v2, `negative_policy=all` (compat) | `v2-latent-allneg-500/` | attribution only; run only if needed |
| E6 | ranker | 500 | contingency: schema v2b (+3 B features), route-balanced hard negatives | `v2-latent-bfeat-500/` | only if E4 fails and E2 passed |

E2 → E3 → E4 are sequential; E5/E6 are conditional and each requires a fresh
bundle path. No formal run before E1/E2 gates pass. E3 uses the fixed
hyperparameters and performs no ALS tuning on validation users. The percentile
variant is dead and is not rerun.

## 9. Success gates and stop conditions (pre-registered, fixed now)

Candidate-stage gates, evaluated on the 500-user three-route diagnosis. All
diagnostic artifacts output both unconditional and route-present conditional
metrics; the gate reads only the pre-registered fields below. Baselines from
current artifacts: ItemCF all-user recall@10 = 0.064; ItemCF present Top-10
hit = 0.073; median union rank = 172. G2/G4 thresholds are set as route-quality
targets for the latent route itself; the 172 baseline uses a different
denominator (union-present users) and is not directly comparable.

- **G1** latent recall@500 ≥ 0.55, denominator = all 500 validation users
  (Dense reaches 0.612 only at top-1500; at the same top-500 budget as ItemCF,
  a collaborative route should clear 0.55 to be worth a third route).
- **G2** among latent-present users (target inside the latent top-500 list),
  the target's rank within the latent route's own list: p50 ≤ 120 and
  p75 ≤ 300.
- **G3** three-route union recall ≥ 0.90 **and** latent-only target coverage
  (target in latent top-500, absent from ItemCF ∪ Dense) ≥ 10/500.
- **G4** latent recall@10 over all 500 users ≥ 0.08 (ItemCF all-user recall@10
  = 0.064; ≈25% relative lift).
- **G5** engineering gates: two identical latent fits produce identical scores
  (regression test); the final latent artifact persists and reloads with a
  checksum match; the 30-user smoke replays its own validation rows exactly and
  keeps constraint satisfaction 100%.

Formal gate (unchanged, E4/E6):

1. LambdaMART mean NDCG@10 > fixed ItemCF baseline (same union).
2. 2,000-sample paired-bootstrap 95% CI lower bound > 0.
3. Recomputed constraint satisfaction rate = 1.0.

Stop conditions:

- Any of G1–G4 fails → preserve the full negative result
  (`v2-latent-diagnostics` + report), never train the ranker, keep frozen
  locked, and hand off.
- E4 fails the formal gate but E2 passed → run E6 (one contingency only), then
  stop regardless of outcome.
- E4 passes → evidence review with the user before any frozen-marker
  consumption.
- Frozen test only after the gate evidence is audited and approved.

## 10. Files, tests, runtime

### Production files

- Modify: `src/recagent_eval/retrieval.py`, `candidate_features.py`,
  `config.py`, `runner.py`, `lambdamart_pipeline.py`, `learned_ranking.py`,
  `bundle.py`, `v2_selection.py`, `ranker_diagnostics.py`, `cli.py`,
  `pyproject.toml` + `uv.lock` (threadpoolctl dependency).
- Create: `src/recagent_eval/latent_retrieval.py`,
  `src/recagent_eval/latent_diagnostics.py`,
  `configs/v2_dense_latent.yaml` (E4), `configs/v2_dense_latent_allneg.yaml`
  (E5), `configs/v2_dense_latent_bfeat.yaml` (E6, written when triggered).
- Contingency only: `ranker_selection.py` (frozen latent threading).
- Task 0 (evidence hygiene, before feature work):
  - `reports/experiments/v2-dense-lambdamart-recall1500-percentile.json`:
    fix the `score_calibration: "raw"` summary typo to `"percentile"` with a
    `corrected_at` note; the original validation artifact and evidence stay
    untouched, and the Markdown report notes the correction.
  - `reports/experiments/v2-dense-lambdamart-recall1500.md` + HANDOFF notes:
    mark the raw evidence fingerprint
    `legacy/non-replayable-under-current-code` (its recorded config fingerprint
    `7b9373b4...` predates the `score_calibration` fingerprint payload added
    in `e1efee8`).

### Tests (TDD: RED first, then GREEN)

- `tests/test_latent_retrieval.py`: fit determinism (two fits, identical
  scores under the threadpool limit), fold-in scoring for CV-style users,
  unseen-item ignore and empty-route behavior, top-k/tie-break,
  allowed_ids/history exclusion, finite scores, rank/alpha/seed validation,
  training fingerprint, NPZ+manifest save/load with checksum validation.
- `tests/test_candidate_features_v2.py`: v1 byte-compat, v2/v2b row lengths
  and fingerprints, zero-filled absent latent features, NaN/Inf rejection.
- `tests/test_config.py`: latent block validation, default-config fingerprint
  unchanged (regression pin), feature-version rules, artifact_path required
  when latent enabled.
- `tests/test_lambdamart_pipeline.py`: route-balanced hard-negative quotas,
  top-up and stable ordering, sampling-policy fingerprint, latent threading
  through fold/validation builders, `negative_policy="all"` byte-compat.
- `tests/test_v2_selection.py`: v2 evidence with latent provenance and
  artifact checksum; v1 evidence still parses; gate unchanged; v1/v2 mixing
  fails closed.
- `tests/test_learned_ranking_errors.py`: v1 artifact still loads; v2 artifact
  schema dispatch and latent checksum check; v1/v2 parse mixing fails closed.
- `tests/test_safe_io_bundle.py`: `lambdamart-bundle/v2` latent member
  publication/load, checksum binding, schema-version strictness.
- `tests/test_cli.py`: `diagnose-latent` refuses overwrite; read-only on
  validation targets; frozen replay loads the persisted latent artifact.

### Runtime estimates (CPU, single-threaded LightGBM as today)

- Latent fit: measured in the E3 30-user benchmark and recorded in the
  diagnostics artifact; no number is promised in advance.
- E1/E2 candidate diagnosis (500 users): provisional ~10–30 min, finalized
  after E3.
- E3 30-user smoke: provisional ~5–15 min, includes the wall-time benchmark.
- E4 500-user main: provisional ~1–3 h (CV dominates; route-balanced hard
  negatives cut training rows ~8×, partially offset by the larger union).
- E5/E6: same order as E4 each.
- Full quality gate after code: pytest+coverage (~13 s current), Ruff, lock
  check, diff check, bash -n.

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| ALS route still does not move median rank | Pre-registered G1–G4 fail fast at candidate stage; negative result preserved; no ranker time spent |
| Determinism breaks replay | `threadpoolctl.threadpool_limits(1)` + fixed seed/order + two-fit regression test; the final item-factor artifact is persisted and loaded (not refit) at replay, with checksum binding; 30-user replay check before 500-user runs; reproducibility claim scoped to the same environment and fixed dependencies |
| Fold-in mismatch between training and validation users | Uniform fold-in scoring from each user's own legal history in every context; saved user factors are never used for scoring; dedicated fold-in regression test |
| Feature-schema bump breaks old artifacts | Versioned constants; v1 default byte-compatible; regression pins old fingerprints |
| Hard negatives change result attribution | Route-balanced policy is fixed and fingerprint-bound; optional `negative_policy="all"` control (E5) isolates its effect if attribution is needed |
| Fingerprint mismatch between runs (as seen in 2.6) | New artifacts record schema/fingerprint versions; E1–E6 all generated under current code; Task 0 marks the legacy fingerprint; summaries copy fields from JSON evidence |
| New dependency pressure | Minimal: `threadpoolctl` as a direct pure-Python dependency (ALS stays numpy-only); uv.lock updated and lock check kept mandatory; no ML-extra changes |

## 12. Non-goals (this round)

- No Plan C (dense encoder swap, LLM rewrite, GPU/4090, external metadata).
- No Qwen/vLLM remote smoke.
- No percentile re-run, no seed/user-subset/gate adjustment.
- No demo-path changes beyond keeping the `LearnedRanker` interface backward
  compatible.
- No resume-number updates until a gate passes and evidence is audited.

## 13. Deliverables after approval

1. Task 0: percentile summary typo correction (with `corrected_at` note) and
   legacy fingerprint marking for the recall-1500 raw evidence.
2. This spec committed and a `writing-plans` implementation plan under
   `docs/superpowers/plans/2026-08-23-collaborative-recall.md`.
3. TDD implementation in the worktree with the full quality gate
   (pytest+coverage, Ruff, uv lock --check, git diff --check, bash -n).
4. E1–E4 (and conditional E5/E6) with new artifact/report names under
   `reports/experiments/v2-latent-*.{json,md}`, including the persisted latent
   NPZ + manifest artifacts and schema-v2 bundles.
5. README/HANDOFF reconciliation only from checked JSON evidence.

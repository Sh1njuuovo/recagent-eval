# Core code walkthrough

## 1. Entry and configuration

`recagent_eval.cli` exposes data download, case generation, validation tuning,
evaluation, configuration inspection, and offline smoke commands.
`ExperimentConfig` controls the comparison without branching inside evaluation:

- baseline disables structured planning, memory, and semantic retrieval;
- structured-memory enables schema planning and state;
- full adds semantic candidates and frozen validation-selected weights.

## 2. Input and state

An episode receives a user utterance and `PreferenceState`. The state separates:

- soft signals: liked/disliked movies and genres;
- hard signals: required/excluded genres, year bounds, excluded movie IDs;
- presentation: requested count and ranking mode.

`PreferencePatch` merges new turns. Hard exclusions are monotonic: a later soft
preference replacement cannot silently clear them.

## 3. Planning and failure path

`RecommendationAgent.recommend` asks `LLMProvider` for one object containing a
preference patch and `ToolPlan`. Pydantic rejects unknown tools and invalid
order. An invalid response gets exactly one repair request. A second failure
uses a fixed safe plan and records `fallback_used`; it does not terminate the
batch.

The provider stores its API key in a private attribute, retries 408/409/429/5xx
and transport failures with exponential backoff, and returns typed errors
instead of logging secrets.

## 4. Retrieval and ranking

The current `v2b` execution path is:

1. `hard_filter` removes watched, disliked, and explicitly excluded IDs and
   enforces required/excluded genres and year ranges.
2. `ItemCFRetriever` produces a Top-500 collaborative route from legal positive
   history. Popularity remains its cold-start fallback.
3. `DenseSemanticRetriever` produces a Top-1500 semantic route from normalized
   `all-MiniLM-L6-v2` embeddings of MovieLens title/genre text.
4. `LatentFactorRetriever` produces a Top-500 weighted-ALS route. Evaluation users are
   scored by standard fold-in from their own legal history; fitted user factors
   are never reused across the split boundary.
5. The route union is represented by the fixed 16-feature `v2b` schema:
   raw score, reciprocal rank, route membership, popularity, genre/year/history
   signals, latent interactions, recent-ItemCF, and year recency.
6. The learned ranker returns the Top-10 with feature contributions, route
   provenance, and deterministic movie-ID tie-breaking.

The earlier TF-IDF/min-max hybrid is still available as a lightweight control
and API-key-free demo path. Its validation-selected weights `(0.7, 0.3, 0.0)`
are retained as historical evidence rather than presented as the final method.

## 5. Dense retrieval and leakage-safe LambdaMART (v2)

The v2 path is the current focus. `DenseSemanticRetriever` stores
`all-MiniLM-L6-v2` embeddings in a schema/fingerprint/revision-validated cache
and returns deterministic NumPy cosine retrieval through the same
`retrieve(query, top_k, allowed_ids)` contract as TF-IDF.

`LeakageSafeRankingSplit` orders each user's ratings by timestamp: earlier
interactions form legal history, then separate positives become ranking-train,
validation, and frozen targets. Whole users stay within one grouped-CV fold.
`train-ranker` builds the versioned candidate rows and publishes a
model/evidence bundle whose fingerprints cover dataset, rows, history, folds,
groups, candidate policy, feature order, config, metrics, and validation replay.

The final algorithm comparison uses a separately generated cohort ledger:
development, Confirmation-A, Confirmation-B, and reserve users are mutually
exclusive. Confirmation-A became debugging/replication evidence after baseline
corrections; the untouched 1,000-user Confirmation-B cohort is the sole
certification result. `baseline_eval.py`, `baseline_summary.py`, and the
`baselines/` package implement the common evaluation protocol for Popularity,
ItemCF, ALS, BPR-MF, LightGCN, and current_v2b.

Key files: `latent_retrieval.py` (weighted ALS persistence and fold-in),
`candidate_features.py` (versioned v1/v2/v2b schema), `learned_ranking.py`
(artifact and booster loading), `lambdamart_pipeline.py` (candidate construction),
`v2_selection.py` (grouped CV), `baseline_eval.py` / `baseline_summary.py`
(strong-baseline evidence), and `promotion.py` (single-use publication).

## 6. Native crash lesson: three OpenMP runtimes

The dense pipeline loads torch before LightGBM. On macOS arm64 this puts three
`libomp.dylib` copies (torch's, LightGBM's, scikit-learn's) in one process;
LightGBM's OpenMP worker threads then dereference a null suspension pointer in
`__kmp_suspend_initialize_thread` and the process dies with exit 139. The fix
pins `n_jobs=1` in the ranker factory, passes `num_threads=1` to raw Booster
prediction, and caps `OMP_NUM_THREADS` before Booster construction. Two
subprocess regression tests reproduce the crash boundary (torch imported, then
train/predict/load) and guard the fix.

## 7. Evaluation and reproducibility

`run_experiment` runs every turn, writes sanitized JSONL episodes, aggregate
metrics, and a manifest with Python/platform/config/data counts and a case
SHA-256. Set-like fields are recursively sorted before hashing. This fixed a
real cross-process reproducibility bug caused by Python hash randomization.

Tests target behavior at module boundaries; the only external mock is
`httpx.MockTransport`, which preserves the real provider request/response
shape.

The final promotion path adds a canonical manifest over the implementation
commit, training identity, model, validation evidence, latent factors, and
semantic cache. Label-free preflight verifies these bytes without opening the
frozen case file. Execution atomically transitions a marker through `started`
to `completed` or `failed`; every state permanently prevents a second run.

For the published result, the 1,000-user Confirmation-B comparison is the main
statistical evidence. The later 50-case one-shot run is a bounded generalization
supplement because that case fingerprint had appeared in an earlier DeepSeek
system experiment and the run did not contain matched ItemCF/ALS baselines.

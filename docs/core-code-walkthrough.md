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

The execution path is:

1. `hard_filter` removes watched/disliked/excluded IDs, invalid year ranges, and
   required/excluded genre violations.
2. `ItemCFRetriever` fits positive interactions from training rows only and
   uses cosine-normalized co-occurrence. Popularity is the cold-start fallback.
3. `TfidfSemanticRetriever` embeds title and genre tokens and excludes
   zero-similarity candidates.
4. `HybridRanker` min-max normalizes sources, adds preference affinity, and
   returns score decomposition for every movie.

Weights were searched on at most 500 validation users with step 0.1 and frozen
as `(0.7, 0.3, 0.0)`. The explicit preference score going to zero is an observed
validation result, not a manually chosen success story.

## 5. Dense retrieval and leakage-safe LambdaMART (v2)

The v2 path is the current focus. `DenseSemanticRetriever` stores
`all-MiniLM-L6-v2` embeddings in a schema/fingerprint/revision-validated cache
and returns deterministic NumPy cosine retrieval through the same
`retrieve(query, top_k, allowed_ids)` contract as TF-IDF.

`LeakageSafeRankingSplit` orders each user's ratings by timestamp: the
penultimate positive item is the validation target and the latest is the test
target; neither enters ItemCF training. `train-ranker` builds ten candidate
features per movie, runs whole-user GroupKFold three-fold CV over a fixed
sixteen-parameter grid, and publishes a model/evidence bundle whose fingerprints
cover dataset, rows, history, folds, groups, candidate policy, config, metrics,
cases, and validation replay. The frozen evaluation replays the validation rows
from the artifact and compares them byte-for-byte with the recorded evidence.

Key files: `learned_ranking.py` (schema, artifact, booster loading),
`lambdamart_pipeline.py` (candidate construction and orchestration),
`v2_selection.py` (grouped CV and evidence), and `bundle.py` (atomic single-use
publication).

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

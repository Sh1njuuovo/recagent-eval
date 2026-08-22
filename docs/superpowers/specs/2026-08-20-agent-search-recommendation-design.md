# RecAgent-Eval v2: Agent×Search/Recommendation Internship Project

## 1. Status and authority

This design records the plan approved on 2026-08-20. It supersedes the proposed
2026-08-10 v2 embedding/ranking design for future implementation. The
2026-08-10 documents remain in the repository as historical planning records;
they are not evidence that their proposed BGE or pairwise-linear pipeline was
implemented or evaluated.

The current v1 code, tests, configurations, and experiment reports remain
supported and unchanged. Any v2 result is publishable only after it exists as a
traceable artifact produced by the implementation below.

## 2. Objective and constraints

The project is a two-week, interview-oriented extension of RecAgent-Eval for
Agent × search/recommendation algorithm internships. It must demonstrate:

- structured LLM planning with deterministic recommendation tools;
- dense semantic candidate retrieval;
- learned Top-10 ranking with interpretable feature contributions;
- leakage-safe offline evaluation and an unbypassable frozen-test gate;
- a single-RTX-4090 Qwen deployment path;
- an explainable, per-session Gradio demo; and
- resume and interview claims grounded in checked-in evidence.

The default development path is local CPU plus an API provider. Server-side
model and demo services bind locally and are reached through SSH local port
forwarding. Multi-agent orchestration, fine-tuning, a vector database, and
distributed training are out of scope.

## 3. Evidence baseline and source policy

The implementation starts from the checked-in v1 repository and preserves all
historical experiment reports. Existing numerical claims may be repeated only
when their source artifact is named, for example:

- `reports/experiments/deepseek-constraint-aware.json` for the checked-in
  50-case Agent/retrieval comparison;
- `reports/experiments/offline-ranker-selection.md` and
  `artifacts/ranker_ablation.json` for the validation-only ranker study; and
- `reports/audit/audit.json` for the original InteRecAgent code audit.

Planned v2 metrics, Qwen results, latency, memory use, and learned-ranker gains
must remain labelled as acceptance criteria until generated. No numerical
result may be inferred from architecture choices or copied from an upstream
project.

Every future result must identify the config, dataset/case fingerprint, feature
schema, model/cache fingerprint, command, environment, and machine class that
produced it. Failure and non-promotion artifacts remain part of the evidence.

## 4. Architecture

```text
user text + session history
  -> structured preference state and tool plan (LLM or rule fallback)
  -> deterministic hard constraints
  -> ItemCF candidates + dense semantic candidates
  -> candidate feature rows in a fixed order
  -> selected ranker (existing ranker or LambdaMART)
  -> recommendations + score/feature explanations + trace
  -> validation evidence -> frozen gate -> at most one authorized test path
```

The LLM interprets language and plans tools. Hard filtering, retrieval, ranking,
metrics, and promotion decisions remain deterministic code paths. Existing
TF-IDF and ranker behavior stays available for old configurations.

## 5. Dense semantic retrieval

### 5.1 Public contract

Add a `SemanticRetriever` protocol with this observable method:

```python
retrieve(
    query: str,
    top_k: int,
    allowed_ids: set[int] | None,
) -> list[tuple[int, float]]
```

`DenseSemanticRetriever` implements the protocol and exposes `fit(...)`,
`load(...)`, and `save(...)` operations. Exact constructor and persistence
argument types may follow the repository's existing model style, but the public
operations and validation behavior are stable.

### 5.2 Model and index

The approved encoder is
`sentence-transformers/all-MiniLM-L6-v2`: 384-dimensional embeddings,
L2-normalized before scoring. Retrieval uses NumPy brute-force cosine
similarity over the small MovieLens catalog. Equal-score ties resolve by movie
ID so repeated runs are deterministic.

`allowed_ids` is applied before Top-K selection. Empty allowed sets return no
candidates. Invalid `top_k`, non-finite embeddings/scores, missing cache data,
dimension disagreement, or fingerprint disagreement fail with actionable
errors.

### 5.3 Cache provenance

The dense cache records a schema version, model name and resolved revision when
available, encoder/runtime metadata, item-text schema, ordered item IDs,
embedding shape/dtype/normalization, and hashes covering source items and array
bytes. `load(...)` validates every contract field before serving retrieval.

## 6. Leakage-safe data protocol

For each eligible user, chronological positive interactions are split as:

- all rows before the last three: legal feature/model training history;
- third-last positive interaction: ranker training target;
- penultimate positive interaction: validation target; and
- latest positive interaction: frozen test target.

Train, validation, and test targets must be disjoint. User histories and all
history-derived features use only legal training rows for the active stage.

Model selection uses three-fold grouped cross-validation with whole users as
groups. Fold assignment, eligibility counts, misses, and fingerprints are
serialized. Frozen-test targets are unavailable to ordinary training,
selection, CLI reporting, cache construction, and demo paths.

## 7. LambdaMART ranking

Extend `RankerKind` with `"lambdamart"`. The implementation uses LightGBM
LambdaMART with query groups defined per user. Candidate features have one
versioned, ordered schema covering:

- ItemCF score, rank, and route membership;
- dense score, rank, and route membership;
- legal-train popularity;
- history/candidate genre compatibility;
- history/candidate year compatibility;
- explicit preference compatibility and presence; and
- candidate source/intersection indicators.

No feature may read validation or frozen-test interactions. Missing values use
an explicitly documented representation. NaN or infinity is rejected before
training and scoring.

Model artifacts store the ranker kind, LightGBM/runtime versions, ordered
feature names, feature-schema version, training/fold/data fingerprints,
parameters, and model bytes/hash. Loading fails on missing model files, schema
or feature-order mismatch, incompatible versions, or fingerprint mismatch.
Empty candidate sets return an explicit empty result. Score ties use the
declared deterministic candidate-ID tie-break.

`ScoreBreakdown` gains `feature_contributions`. For LambdaMART, contributions
must be derived from the loaded model's prediction-contribution facility and
remain aligned with the saved feature order. They explain a score; they are not
causal claims.

## 8. Selection, uncertainty, and frozen gate

Offline comparisons keep identical eligible users, candidate depths, hard
constraints, and metric code. At minimum, reports compare the existing ItemCF
baseline with the dense + LambdaMART system and retain route-level candidate
recall so retrieval and ranking effects remain separable.

Validation uncertainty uses 2,000 paired bootstrap resamples at the user level.
The v2 frozen test unlocks only when a freshly recomputed promotion artifact
shows all of the following:

1. mean validation NDCG@10 is strictly greater than ItemCF;
2. the lower bound of the paired confidence interval for the NDCG@10
   difference is strictly greater than zero; and
3. hard-constraint satisfaction is exactly 100%.

The gate validates config, data, fold, feature, model, metric, and report
fingerprints and rejects manual booleans, edited summaries, stale artifacts,
missing cells, or already-consumed authorization. A failed gate remains a valid
negative result and does not authorize a frozen evaluation.

## 9. Provider and remote Qwen path

`OpenAICompatibleProvider` accepts
`extra_body: dict | None = None` and passes it through to compatible chat
completion requests without mutating caller data. Existing callers remain
unchanged when it is omitted.

The remote smoke path serves Qwen3-8B through vLLM on one RTX 4090, binds the
server endpoint locally, and accesses it through SSH local forwarding. Secrets
stay in environment variables and are never written to reports. The run uses
10 smoke cases and saves commands/configuration, environment, case fingerprint,
provider/model identity, per-case outcomes, aggregate metrics, and failures.
These are required future artifacts, not pre-claimed results.

Remote timeout, unavailable endpoint, malformed provider output, and missing
credentials produce bounded errors and activate the documented rule-based
fallback where the caller permits it.

## 10. Explainable Gradio demo

The demo exposes preferences, recommendations, route/ranker information,
feature contributions, and bounded failure messages. Each browser session owns
its conversation history and preference state; mutable session data is never
shared through module globals. Reset affects only the current session.

The demo supports:

- normal API/provider execution;
- remote Qwen through the forwarded endpoint;
- rule-based fallback on remote timeout or provider failure; and
- a no-key offline path suitable for local verification.

Recommendations still pass deterministic hard constraints under every mode.

## 11. Configuration and CLI compatibility

Configuration adds `semantic.kind`, `semantic.model_name`,
`semantic.cache_path`, and `semantic.device`. Missing new fields preserve old
behavior by selecting TF-IDF and the current ranker. New configuration, cache,
ranker, fold, and result artifacts carry schema versions and fingerprints and
reject incompatible input.

The CLI adds:

- `build-embeddings` for fitting or loading and validating the dense cache; and
- `train-ranker` for grouped-CV LambdaMART training and artifact creation.

`select-ranker` and `evaluate-ranker` accept dense retrieval and LambdaMART
while retaining their existing modes. Frozen evaluation remains protected by
the recomputed gate.

## 12. Test and acceptance contract

Automated tests cover:

- embedding normalization, cache fingerprint validation, allowed-ID filtering,
  deterministic ranking, and feature order;
- model serialization, missing models, schema/feature mismatch, NaN/Inf,
  empty candidates, and score ties;
- provider `extra_body` forwarding;
- train/validation/test disjointness and grouped CV;
- an unbypassable frozen gate;
- demo session isolation, remote-timeout fallback, and no-key offline mode; and
- CLI dense-build, ranker-train, gate, and remote-smoke paths.

Completion requires fresh evidence that:

- pytest and Ruff pass and line coverage is at least 90%;
- CLI smoke commands pass;
- local CPU baseline, modified pipeline, and demo are reproducible;
- the single-4090 Qwen 10-case smoke stores environment and metric artifacts;
- results trace to configs, fingerprints, and JSON evidence; and
- resume, interview pack, and demo text agree with real artifacts.

Until those checks and artifacts exist, documentation describes the work as a
plan or implementation-in-progress and does not claim v2 quality gains.

## 13. Delivery schedule

- Day 1: profile the repository/data and re-establish the v1 baseline.
- Days 2–3: implement and verify dense retrieval/cache.
- Days 4–6: implement features, grouped CV, LambdaMART, explanations, and gate.
- Day 7: run the authorized single-4090 remote smoke.
- Days 8–9: build and test the explainable session-isolated demo.
- Day 10: reconcile evidence, resume bullets, interview pack, and demo script.

The schedule is a planning budget. Evidence quality and frozen-test integrity
take precedence over forcing a positive result.

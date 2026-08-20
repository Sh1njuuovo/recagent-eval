# RecAgent-Eval v2: Agent×Search/Recommendation Internship Project Implementation Plan

> Approved 2026-08-20. Execute in an isolated feature worktree, use TDD for
> behavior changes, and preserve all v1 code paths and historical evidence.

**Goal:** Deliver a two-week, resume-ready extension with MiniLM dense recall,
LightGBM LambdaMART ranking, leakage-safe selection, a protected frozen gate,
single-4090 Qwen3-8B integration, and an explainable per-session demo.

**Authority:**
`docs/superpowers/specs/2026-08-20-agent-search-recommendation-design.md`.
The 2026-08-10 v2 documents are retained as historical proposals.

**Default environment:** Local CPU development plus an API provider. Remote
vLLM and demo services bind to loopback and are reached using SSH local port
forwarding. Do not run paid API, GPU, or frozen-test commands without explicit
authorization.

## Global execution rules

- Start each behavior task with a failing focused test and observe the expected
  failure before implementation.
- Keep old configs defaulting to TF-IDF and the current ranker.
- Preserve all checked-in experiment reports; write new runs to new paths.
- Never use the penultimate or latest target in legal training histories or
  features.
- Store schema versions and fingerprints in every new cache/model/result
  artifact and validate them on load.
- Record failures and rejected promotion gates as evidence.
- Do not write resume metrics until the corresponding result JSON exists.
- Before every task commit, run focused tests and Ruff on touched Python files.

## Task 1: Baseline, active project profile, and historical archive

**Files:**

- Create: `docs/superpowers/specs/2026-08-20-agent-search-recommendation-design.md`
- Create: `docs/superpowers/plans/2026-08-20-agent-search-recommendation.md`
- Create: `reports/archive/2026-08-20-pre-agent-search-v2/**`
- Modify: `reports/profile/jd.txt`
- Modify: `reports/profile/candidates.json`
- Remove from active profile: `reports/profile/taste.json`
- Regenerate: `reports/ranking/candidate_score.json`
- Regenerate: `reports/ranking/candidate_score.md`

Steps:

1. Verify the feature worktree and clean/understood status.
2. Copy the superseded proxy JD, candidates, taste, and taste-weighted ranking
   into the dated archive; leave experiment reports untouched.
3. Write the approved design and this implementation plan.
4. Update the active Agent × search/recommendation proxy JD.
5. Rebuild exactly three official GitHub candidates: RecAI/InteRecAgent,
   RecBole, and OpenOneRec. Use audited facts and explicit candidate-score
   fields; record source and risk notes. Do not add taste fields.
6. Run `candidate_score` without `--taste`, remove the inactive
   `score_breakdown.user_preference` compatibility field from the no-taste JSON,
   and assert that no active ranking/profile file contains Taste Fit,
   `taste_matches`, `taste_mismatches`, or `user_preference`.
7. Run full pytest and Ruff, inspect the diff, and commit the focused baseline
   profile change.

## Task 2: Freeze configuration and artifact contracts

**Files:**

- Modify: `src/recagent_eval/config.py`
- Modify: `src/recagent_eval/models.py`
- Modify: `configs/*.yaml` only where a new v2 example is required
- Test: `tests/test_config.py`
- Test: `tests/test_models.py`

Steps:

1. Add failing tests for the semantic configuration, old-config defaults,
   unknown values, invalid devices/paths, and schema-version failures.
2. Add `semantic.kind`, `semantic.model_name`, `semantic.cache_path`, and
   `semantic.device`; keep absent fields on TF-IDF/current-ranker behavior.
3. Add versioned fingerprint-bearing metadata models shared by dense cache,
   fold map, ranker, and result artifacts.
4. Extend `RankerKind` with `"lambdamart"` and `ScoreBreakdown` with ordered
   `feature_contributions` while retaining backward-compatible defaults.
5. Run config/model tests and serialization round trips.

## Task 3: Implement leakage-safe three-target data preparation

**Files:**

- Modify: `src/recagent_eval/data.py`
- Modify or create focused data-split tests under `tests/`

Steps:

1. Add failing tests proving chronological third-last train target,
   penultimate validation target, and latest frozen-test target are pairwise
   disjoint.
2. Test duplicate timestamps with a deterministic secondary key and users with
   insufficient positives.
3. Implement a new v2 split structure without changing the v1 split API.
4. Expose legal histories from explicitly supplied rows and test that no target
   can enter histories or history-derived features.
5. Fingerprint the ordered split inputs and serialized result.

## Task 4: Add the semantic retriever protocol and dense cache

**Files:**

- Create: `src/recagent_eval/embedding.py`
- Modify: `src/recagent_eval/retrieval.py`
- Create: `tests/test_embedding.py`
- Modify: `tests/test_retrieval.py`

Steps:

1. Add a failing protocol/fixture test for
   `SemanticRetriever.retrieve(query, top_k, allowed_ids)`.
2. Add deterministic fake-encoder tests for 384-d normalization, cosine order,
   movie-ID tie-breaks, Top-K bounds, allowed-ID filtering, and empty sets.
3. Implement `DenseSemanticRetriever.fit/load/save(...)` using NumPy brute
   force and an injected encoder interface.
4. Add cache round-trip and tamper tests for item-text, ordered IDs, array,
   model, dimension, dtype, normalization, schema, and fingerprints.
5. Add the Sentence Transformers MiniLM adapter behind the optional dependency;
   unit tests must not download model weights.

## Task 5: Add embedding build CLI

**Files:**

- Modify: `src/recagent_eval/cli.py`
- Create or modify CLI tests under `tests/`
- Modify: `pyproject.toml`, `uv.lock`, `.gitignore`, and README dependency notes

Steps:

1. Add failing CLI tests for `build-embeddings`, cache reuse, force/rebuild
   behavior, fingerprint mismatch, bad device, and missing optional dependency.
2. Add bounded optional Sentence Transformers dependencies and refresh the lock
   only after approved network access.
3. Implement `build-embeddings` with explicit dataset/config/cache inputs and a
   machine-readable manifest.
4. Run an offline fake-encoder CLI smoke. Record the real model revision only
   when weights are actually resolved.

## Task 6: Build the fixed candidate feature schema

**Files:**

- Create: `src/recagent_eval/candidate_features.py`
- Create: `tests/test_candidate_features.py`

Steps:

1. Define one ordered, versioned feature tuple covering ItemCF and dense
   scores/ranks/membership, legal-train popularity, genre/year compatibility,
   explicit preferences, and candidate-source indicators.
2. Add failing tests for exact feature order and values on a hand-computed
   fixture.
3. Prove history features use only legal training rows and explicit preference
   features use only supplied state.
4. Reject NaN/Inf with user, candidate, and feature context.
5. Serialize the schema and fingerprint with each candidate dataset.

## Task 7: Implement LambdaMART and explanations

**Files:**

- Create: `src/recagent_eval/learned_ranking.py`
- Modify: `src/recagent_eval/ranking.py`
- Create: `tests/test_learned_ranking.py`

Steps:

1. Add failing tests for query groups, labels, fixed feature order, prediction,
   deterministic tie-breaks, and empty candidates.
2. Add failure tests for a missing model, model hash mismatch, feature mismatch,
   NaN/Inf, and incompatible schema/version.
3. Implement the LightGBM LambdaMART adapter with injected/fake model support
   for lightweight unit tests.
4. Save parameters, runtime versions, group/fold/data/feature fingerprints,
   model bytes/hash, and ordered feature names.
5. Populate `ScoreBreakdown.feature_contributions` from model prediction
   contributions and verify order and sum semantics in tests.
6. Add LightGBM as a bounded optional dependency and refresh the lock with
   approved network access.

## Task 8: Train with three-fold grouped CV

**Files:**

- Create or extend a v2 selection module under `src/recagent_eval/`
- Modify: `src/recagent_eval/cli.py`
- Create: `tests/test_v2_selection.py`
- Create or modify CLI tests under `tests/`

Steps:

1. Add failing tests showing users are indivisible groups, fold maps are
   deterministic, and train/validation/test roles stay disjoint.
2. Implement three-fold grouped CV, fold-local model fitting, complete
   denominators, and route-level candidate diagnostics.
3. Add `train-ranker` with explicit config, cache, output, and seed arguments.
4. Extend `select-ranker` and `evaluate-ranker` for dense retrieval and
   LambdaMART without changing existing invocations.
5. Persist per-user and aggregate validation JSON plus all fingerprints and
   environment metadata.

## Task 9: Implement bootstrap and the frozen-test gate

**Files:**

- Modify the v2 selection module and CLI
- Create focused promotion/gate tests

Steps:

1. Add failing tests for 2,000 paired user-level bootstrap resamples and fixed
   seed reproducibility.
2. Implement mean NDCG@10 difference and confidence interval reporting.
3. Add failing gate tests for each required condition: positive mean delta,
   confidence lower bound above zero, and 100% hard constraints.
4. Add tamper/stale/missing-cell/fingerprint mismatch/manual-boolean/consumed
   authorization tests so the gate cannot be bypassed.
5. Keep frozen target loading behind the validated one-use authorization path.
   Do not run the frozen command during ordinary implementation.

## Task 10: Add provider `extra_body` and remote Qwen smoke path

**Files:**

- Modify: `src/recagent_eval/provider.py`
- Modify: `src/recagent_eval/cli.py`
- Modify or create remote scripts/configuration and tests

Steps:

1. Add failing tests that `OpenAICompatibleProvider(..., extra_body=None)`
   preserves old requests and a supplied dict is forwarded without mutation.
2. Add timeout, retry, malformed response, and secret-redaction tests.
3. Implement a bounded 10-case remote smoke command/config for Qwen3-8B served
   by vLLM on one RTX 4090 through SSH local forwarding.
4. Keep server endpoints loopback-only and credentials environment-only.
5. When explicitly authorized, save environment, exact command/config,
   model/provider identity, case fingerprint, per-case outcomes, aggregates,
   latency, and failures. Do not infer missing metrics.

## Task 11: Build the per-session explainable Gradio demo

**Files:**

- Modify: `src/recagent_eval/demo.py`
- Modify: `tests/test_demo.py`
- Update demo documentation after behavior is verified

Steps:

1. Add failing tests proving two sessions cannot read or reset each other's
   preference or conversation state.
2. Test recommendation explanations, feature contribution ordering, hard
   constraints, remote timeout fallback, invalid provider output, and no-key
   offline operation.
3. Implement state through Gradio session state or pure request-scoped objects;
   keep mutable state out of module globals.
4. Display provider/fallback mode, retrieval sources, ranker type, bounded
   explanations, and actionable errors without exposing secrets.
5. Smoke the local CPU demo and the forwarded remote mode when authorized.

## Task 12: Formal verification and evidence reconciliation

**Files:**

- Add new v2 result/evidence files only after running their commands
- Modify: README, demo script, resume/interview pack only from real artifacts

Steps:

1. Run the complete pytest suite with coverage and require at least 90% line
   coverage; run Ruff over source and tests.
2. Run all documented CLI smokes, the local CPU baseline, modified pipeline,
   and no-key demo path from a clean environment.
3. When explicitly authorized, run and archive the single-4090 Qwen 10-case
   smoke with environment and metrics.
4. Recompute the promotion gate from detailed validation artifacts. Run frozen
   evaluation only if every gate condition and fingerprint check passes and
   authorization is still unused.
5. Reconcile every README, resume, interview, and demo claim against a specific
   JSON field; downgrade absent or failed metrics to plans/limitations.
6. Inspect `git diff`, confirm all v1 experiment reports remain byte-for-byte
   available, run final verification, and prepare a focused integration commit.

## Final acceptance checklist

- [ ] Existing configs and rankers retain their behavior.
- [ ] Dense retrieval is normalized, deterministic, filter-aware, cached, and
      fingerprint-validated.
- [ ] Three target roles and grouped CV pass no-leak tests.
- [ ] LambdaMART artifacts validate exact feature order and produce aligned
      feature contributions.
- [ ] Frozen gate cannot be manually or accidentally bypassed.
- [ ] Provider extra body, timeout handling, and rule fallback pass tests.
- [ ] Demo state is isolated by session and works without an API key.
- [ ] pytest, Ruff, CLI smokes, and coverage ≥90% pass freshly.
- [ ] Local baseline/modified/demo runs are reproducible.
- [ ] Authorized 4090 Qwen smoke evidence exists before related claims.
- [ ] Every published number is traceable to a config, fingerprint, and JSON.
- [ ] Resume/interview/demo materials agree with actual evidence.

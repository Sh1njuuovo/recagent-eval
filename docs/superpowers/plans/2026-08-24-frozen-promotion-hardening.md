# Frozen Promotion Hardening Implementation Plan

> **Execution boundary:** implement and test the protected path with synthetic
> cases only. Do not open `cases/fixed_cases.json`, create a real marker/output,
> or run `run-frozen-promotion` without a later authorization naming the exact
> canonical manifest identity.

**Goal:** publish an immutable, manifest-driven Confirmation-B `current_v2b`
promotion package and prove label-free preflight plus one-shot execution
semantics without consuming the frozen set.

**Design:** keep the registered training config and fingerprint unchanged.
Move protected execution identity into a strict promotion schema, copy the
original model/evidence/bundle/latent/semantic bytes into one atomically
published package, replay Confirmation-B in evidence order, and share one
three-route v2b candidate builder between validation and frozen execution.

**Environment:** use only `.venv`; no dependency changes are expected.

## Task 1: Exact ordered Confirmation-B replay

**Files:**

- Modify: `src/recagent_eval/lambdamart_pipeline.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_lambdamart_pipeline.py`
- Modify: `tests/test_cli.py`

1. RED: prove explicit user IDs are currently re-sorted and a permutation can
   replay the wrong order.
2. Add `ordered_user_ids` to validation/query construction. Validate uniqueness,
   exact target membership, eligibility, and completeness; iterate exactly as
   supplied without `sorted()`.
3. Construct `LearnedRanker` from artifact/config with `feature_version="v2b"`
   and the bound score calibration.
4. GREEN: replay all Confirmation-B users from compact evidence and compare
   canonical rows/fingerprint/model checksum.
5. Commit this behavior separately.

## Task 2: Strict promotion schemas and derived identity

**Files:**

- Create: `src/recagent_eval/promotion.py`
- Modify: `src/recagent_eval/config.py`
- Modify: `src/recagent_eval/cli.py`
- Create/Modify: `tests/test_promotion.py`, `tests/test_config.py`

1. RED strict Pydantic schemas for source inventory, manifest, YAML, member
   identity, semantic identity, marker, and validation receipt.
2. Accept normalized repository-relative paths only. Reject absolute paths,
   `.`, `..`, symlink components, hardlinks, and non-regular files; enforce real
   and synthetic root separation.
3. Make `execution.mode` valid only in promotion YAML. Reject it in normal
   experiment config.
4. Derive marker/output/log paths from canonical manifest identity, case fingerprint, dataset
   fingerprint, and model checksum. YAML may only repeat these values.
5. Prove execution-only changes do not affect the training fingerprint and all
   training/candidate/feature/model changes alter identity or fail.
6. Commit strict schema/identity behavior.

## Task 3: Lock source identities before publication

**Files:**

- Create: `reports/promotion/current-v2b-source-inventory.json`
- Modify: `src/recagent_eval/promotion.py`
- Modify: `tests/test_promotion.py`

1. Write a failing test for a strict source-inventory record containing all
   seven package members plus SHA-256 and byte size.
2. Record existing Confirmation-B model, validation, bundle, latent, and
   semantic source bytes before any package copy. Bind the dense manifest's
   immutable revision and internal metadata.
3. Reject absent members, a changed byte/size, empty provenance, or a generated
   substitute. Do not retrain or regenerate any member.
4. Commit the reviewed inventory independently before promotion publication.

## Task 4: Atomic whole-package publication

**Files:**

- Modify: `src/recagent_eval/promotion.py`
- Modify: `tests/test_promotion.py`

1. RED tests for partial publication, overwrite, symlink/hardlink sources,
   missing/tampered members, semantic cache drift, and rename failure.
2. Copy the seven named files (`model.json`, `validation.json`, `bundle.json`,
   `latent.npz`, `latent.npz.json`, `semantic.npz`, `semantic.npz.json`) into a
   sibling build directory with exclusive/no-follow operations.
3. Verify inventory SHA/size before and after copy, fsync members and directory,
   then atomically rename the complete directory. Refuse overwrite.
4. Parse semantic manifest and bind model name, immutable resolved revision,
   movie-catalog fingerprint, dimension, float32 dtype, L2 normalization, and
   canonical cache-manifest fingerprint.
5. Commit atomic publication.

## Task 5: Three-route v2b candidate/feature parity

**Files:**

- Modify: `src/recagent_eval/ranker_selection.py`
- Modify: `src/recagent_eval/learned_ranking.py`
- Modify: `src/recagent_eval/lambdamart_pipeline.py`
- Modify: `tests/test_ranker_selection.py`
- Modify: `tests/test_lambdamart_pipeline.py`

1. RED synthetic contract test for ItemCF top-500 + dense top-1500 + persisted
   latent top-500, latent score flow, recent-ItemCF v2b feature order, and model
   checksum.
2. Extract/reuse one candidate-feature builder for validation and frozen cases.
3. Restrict all frozen histories to `state.liked_movie_ids`. Recover timestamps
   only for those IDs before the target boundary; use the registered
   minimum-timestamp fallback when missing.
4. Add an unexposed extra interaction and hidden target/relevant IDs to the
   synthetic dataset and prove none affects retrieval, fold-in, recent history,
   candidates, or features.
5. Load latent and semantic retrievers only from package-bound members.
6. Commit candidate/feature parity.

## Task 6: Label-free complete preflight

**Files:**

- Modify: `src/recagent_eval/promotion.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_promotion.py`, `tests/test_cli.py`

1. RED spies on `load_cases`, `Path.open/read_text/read_bytes`, built-in `open`,
   and the real fixed-case path.
2. Recompute the complete ranking dataset fingerprint.
3. Verify Git ancestry/protected-path diff, config/training fingerprint, every
   member SHA/size, bundle/model/evidence, semantic/latent manifests, feature
   schema/top-k/calibration, exact ordered Confirmation-B validation replay,
   and derived marker/output identities.
4. Return an immutable validation evidence/receipt without reading case bytes or
   creating marker/output/log files.
5. GREEN zero frozen-case content reads, including all tamper/error paths.
6. Commit preflight.

## Task 7: One-shot marker and output lifecycle

**Files:**

- Modify: `src/recagent_eval/promotion.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_promotion.py`, `tests/test_cli.py`

1. RED tests for `started`, capturable `failed`, `completed`, malformed marker,
   retry rejection, and output-published/started crash window.
2. Atomically create `started` immediately before the injected case loader.
3. On capturable exceptions atomically transition to `failed`; leave `started`
   for simulated uncatchable interruption.
4. Publish output atomically and fsync it before transitioning to `completed`.
5. Add a read-only audit that hashes output in the crash window and never
   changes marker/output.
6. Commit lifecycle behavior.

## Task 8: CLI, synthetic end-to-end rehearsal, and package metadata

**Files:**

- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_cli.py`, `tests/test_promotion.py`
- Create: `reports/promotion/current-v2b-manifest.json`
- Create: `reports/promotion/current-v2b.yaml`
- Create: `reports/promotion/current-v2b-validation.json`
- Create (gitignored): `artifacts/promotion/current-v2b/`

1. Add prepare, preflight, run, and audit commands with manifest-driven dispatch.
2. Exercise success, retry, failed, output+started crash window, latent/semantic
   tamper, order drift, feature/top-k drift, and zero-read preflight entirely
   under a synthetic root.
3. Commit production/test implementation. Record this commit as the hardening
   implementation commit.
4. Prepare the real artifact package by copy-only publication from the locked
   source inventory. If any original member is absent or changed, stop and
   report the blocker without regenerating it.
5. Create manifest, YAML, and validation evidence with exclusive atomic writes.
   Bind the implementation commit and verify it is an ancestor of HEAD with no
   later protected-path changes.
6. Commit only promotion metadata/docs after implementation identity.

## Task 9: Independent verification and authorization handoff

1. Run:

   ```bash
   .venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered --cov-fail-under=90
   .venv/bin/ruff check .
   UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
   git diff --check
   find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
   ```

2. Run exact Confirmation-B replay and the synthetic rehearsal suite.
3. Prove real marker/output absence and zero reads of
   `cases/fixed_cases.json`.
4. Report HEAD, quality gates, manifest/YAML/evidence paths and SHA-256,
   package member hashes/sizes, replay result, candidate/feature contract,
   synthetic lifecycle matrix, derived real marker/output paths, and the one
   future command.
5. Stop and request authorization naming the exact canonical manifest identity. Do not consume
   frozen cases in this plan.

## Task 10: Read-only promotion package and no-replace output amendment

**Files:**

- Modify: `src/recagent_eval/retrieval.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `src/recagent_eval/promotion.py`
- Modify: `tests/test_retrieval.py`, `tests/test_promotion.py`, `tests/test_cli.py`
- Regenerate: `reports/promotion/current-v2b-manifest.json`
- Regenerate: `reports/promotion/current-v2b.yaml`
- Regenerate: `reports/promotion/current-v2b-validation.json`
- Republish: `artifacts/promotion/current-v2b/`

1. Write a RED test showing ordinary `DenseSemanticRetriever.load()` creates
   the adjacent cache lock while the new promotion-only read-only load must not
   create it or mutate cache/manifest bytes.
2. Write a RED preflight invariant test that snapshots the exact seven member
   names, SHA-256 values, and sizes before replay and rejects an extra lock file
   or any post-load mutation.
3. Add the minimal lock-free read-only dense-cache loading path and use it only
   for promotion preflight/execution package members; retain locking for
   mutable cache workflows.
4. Write a RED race test that creates the final output immediately before the
   publication primitive and requires publication to fail without replacing
   those bytes.
5. Implement atomic strict no-replace output publication using a temporary
   regular file plus a filesystem operation whose success is conditional on
   destination absence; fsync the published file and parent before `completed`.
6. Rename identity fields, CLI arguments, validation messages, and docs so
   `canonical_manifest_identity` is distinct from `manifest_file_sha256`.
7. Commit production/test changes as the new hardening implementation commit.
8. Remove the contaminated package by a recoverable move, then republish the
   seven original source bytes through the existing whole-directory atomic
   publisher. Do not train or rematerialize any member.
9. Regenerate manifest, YAML, and label-free receipt against the new
   implementation commit and canonical identity; preflight must prove identical
   package snapshots before and after the complete ordered Confirmation-B
   replay.
10. Run the full quality gates and prove the newly derived real marker/output
    paths do not exist. Stop before frozen consumption.

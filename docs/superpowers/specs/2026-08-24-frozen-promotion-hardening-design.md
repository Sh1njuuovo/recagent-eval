# Frozen Promotion Hardening Design

**Date:** 2026-08-24
**Status:** approved for hardening implementation; frozen execution unauthorized
**Authorization boundary:** this design and its implementation are authorized.
Reading the real frozen cases, creating the real marker, and writing the real
frozen output remain prohibited until the user separately replies
`批准消费 frozen test`.

## 1. Goal

Build a manifest-driven promotion package that can prove the future frozen run
uses the exact Confirmation-B `current_v2b` contract:

- the recorded Confirmation-B users in their original evidence order;
- ItemCF top-500, dense top-1500, and persisted latent top-500 candidates;
- raw score calibration and the v2b feature schema;
- the bundle-bound LambdaMART model and latent artifact;
- an atomic, permanently one-shot marker lifecycle; and
- a label-free preflight that never reads real frozen-case contents.

The hardening work does not tune the recommender, change thresholds, consume a
reserve/frozen target, or authorize a frozen run.

## 2. Verified root causes

The current protected path cannot yet reproduce Confirmation-B:

1. `_evaluate_learned_ranker` constructs `LearnedRanker` without
   `score_calibration` or `feature_version`, so the defaults are raw/v1 even
   when the artifact is v2b.
2. Validation replay passes only `max_users`; `build_candidate_queries` sorts
   targets and slices the first N users instead of consuming the evidence's
   ordered Confirmation-B IDs.
3. `evaluate_frozen_cases` retrieves only ItemCF and semantic candidates. It
   does not load the bundle-bound latent artifact, supply latent scores, build
   the three-route union, or provide the v2b recent-ItemCF feature.
4. Learned execution is dispatched by `ranker.kind=lambdamart`, while the
   registered training configuration has `ranker.kind=minmax_linear` and
   fingerprint
   `530ec64871b9e042da5e566d1a1ad9358dbed811dace4caaaa9de15419b3800c`.
   Editing that training YAML would obscure the registered identity.
5. The current marker has only a `consumed` state, and no immutable promotion
   manifest, promotion YAML, or repository-owned copy of the exact bundle has
   been published.

These are contract and orchestration defects. The ranking algorithm and its
selected parameters remain unchanged.

## 3. Chosen architecture

Promotion is a separate domain module, not another mode embedded in the
training configuration. The implementation creates `recagent_eval/promotion.py`
with four bounded responsibilities:

1. strict manifest/YAML parsing and identity validation;
2. atomic promotion-package publication;
3. label-free preflight; and
4. one-shot execution-state transitions.

The CLI exposes separate commands for package preparation, preflight, execution,
and read-only audit. The final execution command consumes a promotion YAML; it
does not dispatch by mutating `ExperimentConfig.ranker_kind`.

### 3.1 Repository layout

Small, reviewable identity files are committed:

```text
reports/promotion/current-v2b-manifest.json
reports/promotion/current-v2b.yaml
```

Large immutable files remain under the gitignored artifact tree:

```text
artifacts/promotion/current-v2b/
  model.json
  validation.json
  bundle.json
  latent.npz
  latent.npz.json
  semantic.npz
  semantic.npz.json
```

The manifest contains only repository-relative POSIX paths. No manifest or YAML
field may contain an absolute path, `..`, a home-relative path, a URI, or a path
under `/private/tmp`.

Before the worktree is deleted, the entire
`artifacts/promotion/current-v2b/` directory must be backed up and its manifest
SHA reverified. A later publication may use GitHub Releases or Git LFS, but that
is outside this implementation and cannot change the recorded bytes.

## 4. Atomic package publication

### 4.1 Whole-directory transaction

The large artifact package is built in a randomly named sibling directory:

```text
artifacts/promotion/.current-v2b.build-<nonce>/
```

Every member is completely materialized there. The builder then:

1. validates the bundle's internal checksums and metadata;
2. validates every member's expected SHA-256 and byte size;
3. fsyncs each file and the temporary directory;
4. verifies the final path does not exist; and
5. atomically renames the complete directory to
   `artifacts/promotion/current-v2b/`, followed by parent-directory fsync.

It never publishes individual members into the final directory. A failure
before rename leaves only an inert sibling build directory. A failure after
rename but before report metadata publication leaves a complete, non-executable
artifact package. A resume operation may verify and adopt that exact package;
it may not overwrite or mutate it.

### 4.2 File safety

For every source, temporary member, and final member the builder:

- resolves and validates containment under an allowed repository directory;
- uses `lstat`/no-follow opens to reject symlinks at every existing component;
- accepts only regular files;
- rejects files with link count other than one, thereby rejecting hardlinks;
- rejects devices, sockets, FIFOs, and other special files;
- creates destinations exclusively;
- copies through bounded binary streams;
- records SHA-256 and byte size while copying; and
- compares the copied digest/size with a second read before publication.

Generated report metadata follows the same exclusive-create, fsync, checksum,
and regular-file rules. All mutation stays inside repository-controlled
promotion directories. The implementation must not use a `/private/tmp`
artifact as an execution input.

The Confirmation-B artifact package itself is copy-only. Preparation may copy
only the existing original model, validation evidence, compact bundle, latent
NPZ/manifest, and semantic NPZ/manifest bytes. Every source and copied member
must match a previously recorded SHA-256 and byte size before publication. A
missing source or identity record stops preparation immediately.

Preparation must not retrain a model, regenerate validation evidence, rebuild
retriever arrays, or substitute newly produced bytes merely because a config,
model, or cache fingerprint is equal. Recovery is accepted only when the
recovered file is byte-for-byte identical to the pre-recorded SHA-256 and size.
Any differing bytes require a new model/package identity and a new confirmation
process before frozen promotion.

## 5. Manifest and promotion YAML identity

### 5.1 Non-circular hashing

The manifest is the execution identity root. Its canonical JSON SHA-256 is the
promotion identity. The promotion YAML records:

- the repository-relative manifest path;
- the manifest SHA-256;
- `execution.mode: learned_frozen`; and
- a minimal duplicate of execution-critical identity fields needed for
  cross-checking.

The manifest does not store the YAML SHA. This removes circular hashing. At
preflight, YAML and manifest must agree on execution mode, training config
fingerprint, dataset fingerprint, case fingerprint, candidate-policy
fingerprint, model checksum, feature schema/fingerprint, score calibration,
route top-k values, marker path, and output path.

Both report files reject overwrite and are atomically published. Publication
order is artifact directory, manifest, then YAML. Without all three valid
layers, the package is not executable.

### 5.2 Manifest contents

Schema `frozen-promotion/v1` binds at least:

- the hardening implementation commit;
- training config path and fingerprint;
- dataset, case, cohort-ledger, Confirmation-B compact-bundle, summary, and
  ordered-user digests;
- ordered Confirmation-B user IDs and user count;
- model checksum and feature schema version/names/fingerprint;
- score calibration;
- ItemCF, dense, and latent top-k values;
- candidate-policy and validation-gate fingerprints;
- dependency versions and platform identity;
- semantic-cache model name and immutable model revision, dataset fingerprint,
  embedding dimension, dtype, normalization mode, and cache-manifest
  fingerprint;
- every package member's repository-relative path, SHA-256, and byte size;
- repository-relative marker, output, command-log, and failure-log paths; and
- the one permitted execution command in canonical argument order.

The manifest never fabricates absent provenance. Derived and recovered fields
retain their existing evidence-source semantics.

### 5.3 Git identity without self-reference

The implementation is completed and committed before the immutable manifest is
generated. The manifest binds that hardening implementation commit. Later
commits may add only the manifest and documentation.

Preflight verifies:

1. the implementation commit is an ancestor of HEAD;
2. `src/`, `pyproject.toml`, `uv.lock`, the registered training config, and
   dependency-affecting configuration have no diff between that commit and
   HEAD; and
3. later changes are limited to approved promotion identity/docs paths.

Any production-code, training/candidate/feature configuration, or dependency
change after the bound commit invalidates promotion and requires a newly
reviewed implementation identity. The manifest itself never attempts to bind
the commit that contains the manifest, avoiding a Git self-reference.

## 6. Training identity and execution dispatch

`execution.mode` exists only in the independent promotion YAML schema. It is
not added to `ExperimentConfig`, the registered training YAML, candidate-policy
payloads, or LambdaMART training-fingerprint payloads.

The promotion executor loads the original training configuration unchanged,
checks that `lambdamart_config_fingerprint(config)` remains
`530ec64871b9e042da5e566d1a1ad9358dbed811dace4caaaa9de15419b3800c`,
then dispatches the protected learned path from the promotion schema.

Tests must demonstrate:

- changing only an execution-only promotion field leaves the training
  fingerprint unchanged;
- changing retrieval top-k, semantic top-k, latent top-k, score calibration,
  feature version, negative policy, model checksum, candidate-policy identity,
  or another training/candidate/feature/model field changes the relevant
  fingerprint or is rejected; and
- setting `execution.mode` in a normal training YAML is rejected as an unknown
  promotion concern rather than silently accepted.

## 7. Exact Confirmation-B validation replay

The ordered validation user IDs come from the committed evidence, not from a
size or a newly sorted cohort. Preparation verifies that they exactly equal the
Confirmation-B compact bundle and ledger order.

`build_validation_rows` and the lower candidate-query builder gain an explicit
`ordered_user_ids` input. When present they:

- reject duplicate, missing, extra, or ineligible users;
- iterate exactly in the supplied order;
- never call `sorted()` on those IDs;
- never replace them with the first N eligible users; and
- emit rows in the same order before computing the validation-row digest.

Replay compares canonical rows byte-for-byte with validation evidence, checks
the recorded validation-row fingerprint, and checks the model checksum. A
user-order-only permutation must fail.

## 8. Three-route v2b execution contract

### 8.1 Artifact-driven ranker construction

The model artifact determines the feature schema. A v2b promotion requires:

- artifact schema v2;
- feature schema version v2b;
- exact v2b feature names in order;
- the v2b feature fingerprint;
- the recorded model checksum; and
- a latent checksum/provenance record matching the bundle and manifest.

The training config supplies `score_calibration=raw`; the candidate-policy
fingerprint and manifest bind that value. `LearnedRanker` is constructed with
both the validated calibration and artifact-derived `feature_version="v2b"`.
The executor does not rely on constructor defaults.

### 8.2 Persisted latent retriever

Preflight first verifies the bundle-bound latent NPZ and its JSON manifest.
Execution loads it with `LatentFactorRetriever.load`, including the expected
training fingerprint from artifact provenance. Missing, modified, symlinked,
hardlinked, wrong-shape, non-finite, or checksum-drifted latent data fails
before any real case content is read.

### 8.3 Persisted semantic cache

Preflight verifies the package-local `semantic.npz` and `semantic.npz.json`
against their recorded SHA-256, byte size, cache-manifest fingerprint, model
name, immutable model revision, complete dataset fingerprint, embedding
dimension, dtype, and normalization mode. Execution may load dense embeddings
only from this package member. It must not discover, fall back to, or read a
mutable semantic cache elsewhere in the repository or filesystem.

### 8.4 Candidate and feature construction

For each case the protected executor uses:

- ItemCF top-500;
- dense semantic top-1500;
- persisted latent fold-in top-500; and
- the union of all three routes after the same hard filter/history exclusion.

It passes `latent_scores` into `LearnedRanker.rank`. Candidate retrieval and all
features treat `case.state.liked_movie_ids` as the complete visible positive
history. They may recover timestamps only for those explicitly listed movie
IDs, and only from interactions before the frozen target boundary. Interactions
for the same user that are absent from state must never enter retrieval,
fold-in, recent-history selection, or feature computation. The hidden test
target and all relevant IDs must never enter history or features.

For v2b recent-ItemCF features, the executor orders only the state-exposed liked
IDs by `(recovered_timestamp, movie_id)`, takes the last ten, and scores that
recent set over the complete three-route union. A missing timestamp uses the
pre-registered deterministic fallback `(minimum_allowed_timestamp, movie_id)`;
the fallback value and ordering rule are bound by the candidate-policy
fingerprint and manifest. Those `recent_itemcf_scores` are passed explicitly
into candidate feature creation.

Contract tests compare frozen-preparation candidates/features with the
Confirmation-B validation builder on synthetic data. They lock route sizes,
union membership, feature value order, feature names/fingerprint, calibration,
and model checksum. A dedicated fixture includes extra dataset interactions for
the case user that state does not expose and proves they cannot affect latent
fold-in, candidates, recent history, or features.

## 9. Label-free preflight

Preflight may read Git metadata, training config, promotion YAML/manifest, the
complete non-frozen training dataset, model, validation evidence, bundle,
semantic NPZ/manifest, latent NPZ/manifest, and existing marker/output metadata.
It must:

- recompute the complete dataset fingerprint from the registered dataset;
- run the complete Confirmation-B validation replay in the evidence's exact
  ordered-user sequence;
- compare the replayed canonical rows, aggregate, model checksum, and validation
  fingerprint with the package evidence; and
- verify semantic, latent, model, validation evidence, and bundle identities
  before any frozen-case read is possible.

It must not:

- call `load_cases`;
- call `case_fingerprint` on case contents;
- open, read, mmap, hash, or parse `cases/fixed_cases.json`;
- create or modify a marker;
- create output or logs; or
- infer frozen labels from another artifact.

The registered case fingerprint is checked only as an already-bound identity.
Path resolution and `lstat` metadata checks are allowed to enforce containment
and reject symlinks; file-content access is forbidden. Spy tests patch
`load_cases`, `Path.open`, `Path.read_text`, `Path.read_bytes`, and the built-in
open path for the real case path and prove none is invoked.

Preflight fails closed on any manifest/YAML disagreement, SHA/size drift,
training or Git identity drift, ordered-user mismatch, model/feature/candidate
policy drift, semantic or latent drift, incomplete validation replay, existing
marker, or existing output.

## 10. Marker and output lifecycle

### 10.1 Marker states

Schema `frozen-run-marker/v1` has states `started`, `completed`, and `failed`.
Every state binds the manifest SHA, case fingerprint, model checksum, evidence
SHA, marker identity, and start timestamp.

The real marker path is deterministically derived from the promotion manifest
SHA-256, registered case fingerprint, complete dataset fingerprint, and model
checksum. YAML cannot choose it. YAML only repeats the expected derived path;
preflight recomputes and compares it with both YAML and manifest. The output and
log paths are derived under the same identity-scoped directory.

Execution creates `started` with exclusive atomic creation and directory fsync
immediately before the first case-content read. If any valid marker already
exists, all execution attempts fail. An invalid or unreadable marker also
fails closed.

Capturable Python exceptions after `started` atomically transition the marker
to `failed` with an error-type/message digest and optional failure-log digest.
SIGKILL, power loss, interpreter abort, segmentation fault, and other native
crashes may leave `started`. The design makes no promise that those failures
become `failed`; `started` permanently consumes the opportunity and blocks a
rerun.

### 10.2 Output ordering and crash window

The executor refuses an existing output. A successful result is written to an
exclusive sibling temporary file, fsynced, checksum-verified, atomically
renamed to the final output, and parent-directory fsynced. Only then does the
marker atomically transition from `started` to `completed` with output SHA and
size.

A crash after output publication and before the marker transition leaves
`output + started`. This state still permanently rejects execution. The
read-only audit command may hash the output and report whether it matches a
captured temporary execution receipt; it cannot promote the marker to
`completed`, delete the marker, or authorize another run.

Marker transitions use same-directory temporary files, exclusive creation,
fsync, validation of the current `started` binding, and atomic replacement.

## 11. Path containment

Promotion schemas accept only normalized repository-relative paths. Each path
is joined to the discovered repository root and resolved before use. Allowed
roots are explicit:

- `reports/promotion/` for small identity files;
- `artifacts/promotion/current-v2b/` for real promotion members;
- dedicated `artifacts/frozen/` subdirectories for the future real marker,
  output, and logs; and
- a separately named synthetic/rehearsal root.

Absolute paths, empty components, `.`, `..`, path escape, symlink components,
hardlinked files, and non-regular files are rejected. Real and synthetic roots
may not be parents, children, aliases, or resolved equivalents of each other.

A future user authorization must name the exact canonical promotion manifest
SHA-256. Replacing or changing the manifest invalidates that authorization and
requires a new explicit approval before any case-content read.

## 12. Synthetic one-shot rehearsal

Synthetic rehearsal uses generated movies, ratings, model fixture, validation
evidence, latent and semantic artifacts, cases, manifest, YAML, marker, output,
and logs under a dedicated synthetic root. No path is shared with the real
promotion marker, output, cases, or artifact package.

The rehearsal/test matrix covers:

1. successful preflight and execution ending in `completed`;
2. a second execution rejected for an existing completed marker;
3. a capturable evaluation exception ending in `failed`, with retry rejected;
4. simulated interruption after atomic output publication but before marker
   completion, leaving `output + started`, with retry rejected and read-only SHA
   audit available;
5. missing and byte-modified latent artifacts rejected before case reads;
6. missing and byte-modified semantic artifacts, model revision drift, or
   cache-manifest drift rejected before case reads;
7. Confirmation-B ordered-user permutation rejected;
8. an extra dataset interaction absent from state has no candidate or feature
   effect, while target/relevant IDs remain hidden;
9. v2b feature schema/name/order/fingerprint drift rejected;
10. ItemCF/dense/latent top-k drift rejected;
11. model checksum or bundle-member drift rejected;
12. arbitrary marker-path drift rejected in favor of the derived path; and
13. label-free preflight proving complete dataset/replay verification and zero
    real-case content access.

Unit tests separately prove all three marker states permanently block a new
run and malformed markers fail closed.

## 13. CLI boundaries

The planned commands are:

```text
prepare-frozen-promotion   # no frozen-case read; refuses overwrite
preflight-frozen-promotion # label-free and read-only
run-frozen-promotion       # future authorization required
audit-frozen-promotion     # read-only; never changes marker/output
```

`run-frozen-promotion` is guarded by explicit promotion identity and is never
invoked during hardening. Tests call the same execution service only with a
synthetic package and synthetic cases.

## 14. TDD and commits

Implementation proceeds in small RED-GREEN commits:

1. ordered-user replay and artifact-driven v2b ranker construction;
2. three-route frozen candidate/feature parity;
3. strict promotion schemas and path containment;
4. atomic whole-directory artifact publication;
5. manifest/YAML/Git identity validation;
6. label-free preflight;
7. marker/output lifecycle and read-only audit; and
8. synthetic end-to-end rehearsal and immutable promotion generation.

Every production change is preceded by a focused failing test that demonstrates
the reviewed defect or missing contract. No test may load the real fixed cases.

## 15. Completion gate

Hardening is ready for a new authorization review only when:

- the full pytest coverage command passes with actual coverage at least 90.00%;
- Ruff, lock, diff, and shell syntax gates pass;
- exact ordered Confirmation-B replay matches its recorded evidence;
- candidate/feature/model parity tests pass;
- the synthetic success, retry, failed, and crash-window rehearsals pass;
- label-free preflight proves zero real-case content reads;
- promotion artifact, manifest, and YAML paths and hashes are reported;
- the real marker and real output are absent; and
- the worktree is clean at the reported HEAD.

After that report, execution remains paused until the user explicitly replies
`批准消费 frozen test`.

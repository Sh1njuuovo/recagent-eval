# RecAgent-Eval v2 Evidence Hygiene Design

**Date:** 2026-08-24
**Status:** approved for P0 implementation
**Scope:** evidence integrity, replay, resource provenance, robustness, coverage,
and factual documentation. Recommendation algorithms and success thresholds are
out of scope.

## 1. Safety and experimental identity

All work is performed on `feat/dense-recall-v2` in the existing
`agent-search-v2` worktree with its `.venv`. The implementation must not read a
reserve/frozen target, create a frozen marker, call an LLM, start Qwen/vLLM,
change recommendation behavior, reselect hyperparameters, change success
thresholds, overwrite historical artifacts, merge, push, or create a pull
request.

Confirmation-A is retained byte-for-byte as historical
development/debugging/replication evidence. Its first reading was followed by
BPR/LightGCN and ALS selection-metric fixes, so it is not an untouched final
confirmation cohort. Confirmation-B is the sole final certification cohort.
Success A is decided only from the seed-42 Confirmation-B result.

P0 ends before frozen consumption. This approval covers evidence-hygiene work
only. A later preflight must not load frozen labels, and neither preflight nor
frozen execution may occur until the user gives a new, explicit authorization.

## 2. Immutable v1 and strict v2 evidence

Existing `baseline-evaluation/v1` artifacts remain byte-identical. A versioned
reader supports v1 for historical recovery and v2 for newly recorded evidence.
Every summary invocation accepts exactly one known artifact schema. Unknown
versions and v1/v2 mixtures fail closed.

`baseline-evaluation/v2` contains these required identity fields:

- method, cohort, user count, ordered user IDs;
- dataset, config, model, and cohort-ledger fingerprints;
- selected parameters, parameter grid, primary seed, and robustness seeds;
- Python, NumPy, torch, LightGBM, and scikit-learn versions when applicable;
- platform, operating-system release, machine/processor, and CPU information;
- resource metrics and per-user ranking metrics;
- a canonical artifact fingerprint covering all evidence-bearing fields.

An unavailable dependency is recorded explicitly as `not_applicable`; an
unknown value is never invented.

Each provenance-bearing field uses one of three source classes:

- `observed`: recorded directly by the original run;
- `derived`: deterministically recomputed from observed evidence;
- `recovered`: reconstructed after the run through a documented deterministic
  process.

Recovered parameters are labeled `recovered_after_run` and include the exact
recovery command, source artifact SHA-256, recovery input fingerprint, output
fingerprint, commit SHA, and recovery timestamp. Recovery cannot be represented
as original observation.

## 3. Strict summary validation

Before aggregation, the summary reader validates all inputs as one atomic set:

1. Each input slot name equals the artifact `method`; each method appears once.
2. Every artifact cohort equals the requested cohort; A and B cannot mix.
3. All artifacts use the same known schema version.
4. Dataset/config/model fingerprints are non-empty strings; ledger identity is
   present for v2 and externally supplied for v1 recovery.
5. Artifact user count, ordered IDs, and rows match the selected ledger cohort
   exactly.
6. Duplicate users, missing or extra users, row-order drift, NaN/Inf, invalid
   booleans, malformed metrics, and fingerprint drift are rejected.

Validation precedes aggregate or bootstrap calculation. No partial summary is
published after any validation error. Output publication refuses overwrite and
uses atomic creation where the existing safe-I/O patterns permit it.

## 4. Compact evidence bundle and replay chain

A committed bundle is generated separately for Confirmation-A and
Confirmation-B. It excludes recommendation IDs and model weights. Each method
entry contains:

- schema and generator name/version;
- method and cohort;
- ordered user IDs;
- dataset/config/model fingerprints;
- selected parameters and seed with provenance source;
- per-user Recall@10, NDCG@10, MRR@10, candidate recall, and constraint result;
- source artifact SHA-256;
- cohort-ledger SHA-256 and logical ledger fingerprint;
- generating commit SHA;
- canonical per-user-row digest.

The bundle also binds the committed summary digest/fingerprint, bootstrap seed
42 and 2,000 resamples, and its own canonical fingerprint.

The independent replay CLI verifies the complete chain:

```text
source rows
  -> canonical per-user digest
  -> aggregate metrics
  -> all pairwise bootstrap comparisons
  -> summary digest and summary fingerprint
```

Every arrow is checked. Duplicate/misaligned users, non-finite values, method or
cohort drift, source/ledger SHA drift, canonical digest drift, summary drift,
unknown schema, and attempted overwrite fail closed. Replay reads evidence and
writes nothing unless an explicit fresh output path is requested.

Historical v1 source files stay ignored and unchanged. The compact committed
bundle is the durable replay substrate.

## 5. Peak RSS correction

Resource measurement is implemented in a focused helper. Darwin interprets
`ru_maxrss` as bytes and Linux interprets it as KiB; both normalize to MiB. The
v2 metric is named `process_peak_rss_mib` and records:

- metric name and source (`resource.getrusage(RUSAGE_SELF).ru_maxrss`);
- raw numeric value and raw unit;
- normalized MiB;
- platform/system identity;
- measurement process identity.

Formal method measurement must run one method per independent subprocess,
because `ru_maxrss` is a process-lifetime maximum. Unit tests exercise Darwin,
Linux, and unsupported-platform/error behavior without relying on the host OS.

Existing `peak_memory_mb` values are not corrected in place. The correction
addendum labels them `invalid_due_to_platform_unit_bug`. Peak memory is removed
from public Pareto/resource comparisons until new independently measured runs
exist. Full A/B models are not rerun solely to repair this field. Post-hoc
robustness and any later authorized frozen execution use the corrected
subprocess measurement path.

## 6. Post-hoc robustness protocol

Seed 42 remains the formal Confirmation-B main result. Seeds 7 and 2026 are
fixed on 2026-08-24 as **post-hoc robustness** for the project-local BPR-MF and
LightGCN implementations. This work is not described as completion of an
original preregistered three-seed protocol.

The robustness runner uses the same Confirmation-B ordered users and the seed-42
selected parameters. It does not rerun parameter grids or select a favorable
seed. If the selected parameters can only be deterministically reconstructed,
they are marked `recovered_after_run` with recovery command and fingerprints as
specified above.

The report lists every seed, arithmetic mean, sample standard deviation, and
worst seed for Recall@10, NDCG@10, MRR@10, candidate recall, and constraints.
All results are retained. No outcome changes current_v2b, Success A thresholds,
the seed-42 certification, or any hyperparameter.

Language is limited to the project-local fixed-protocol implementations.
LightGCN is not called canonical or official. If it remains below Popularity,
the report discusses the fixed CPU budget, training epochs, and differences
from standard-library implementations.

## 7. Coverage and documentation

Coverage is raised above the real 90.00% gate through behavior-focused tests,
prioritizing baseline artifact/schema validation, summary/replay, CLI refusal
paths, resource provenance, and malformed evidence. Coverage configuration,
precision, omissions, thresholds, and production code are not manipulated to
manufacture a pass.

README, HANDOFF, methodology, demo material, baseline reports, and interview
pack are reconciled to one timeline: early LambdaMART failures; candidate-depth
diagnostics; ALS latent recall gains; historical all-negative v2b point estimate
with CI crossing zero; seed-42 Confirmation-B certification; A's downgraded
identity; frozen still unconsumed; Qwen/4090 still pending. Numbers are copied
from validated JSON or replayed bundles.

## 8. Verification and stopping condition

P0 completion requires the full pytest coverage gate, Ruff, `uv lock --check`,
`git diff --check`, shell syntax checks, A/B replay, cohort disjointness, marker
absence, Markdown/JSON consistency, stale-claim search, unchanged historical
artifact hashes, and a clean worktree.

The P0 handoff includes a draft promotion manifest, exact preflight/frozen
commands, output and marker paths, current commit and locked identities, plus an
explicit promise that a failed frozen run will not trigger tuning or a rerun.
Execution stops there and asks for a second authorization.

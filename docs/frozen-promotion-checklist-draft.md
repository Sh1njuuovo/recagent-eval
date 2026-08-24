# Frozen promotion checklist — draft only

This document fixes the promotion process without authorizing any frozen-label
read. The 2026-08-24 approval covers P0 evidence hygiene only. After P0, work
must stop until the user explicitly replies that one-time frozen consumption is
approved.

## P0 handoff requirements

- Full quality gates are fresh and green with actual coverage at least 90.00%.
- Confirmation-A and Confirmation-B compact bundles replay exactly.
- Confirmation-A is labeled development/debugging evidence.
- Confirmation-B seed 42 is the only certification result.
- Post-hoc seeds 7 and 2026 are complete for BPR-MF and LightGCN.
- Historical v1 artifact SHA-256 values are unchanged.
- The frozen marker path derived from the locked identity does not exist.
- Branch, commit SHA, dataset/config/model/cohort/bundle fingerprints are
  printed for user review.

## Promotion manifest draft contract

The immutable manifest will use schema `frozen-promotion/v1` and bind:

- P0 commit SHA and dirty-state assertion;
- dataset and frozen case fingerprints;
- Confirmation-B compact-bundle fingerprint and summary fingerprint;
- current_v2b config and model fingerprints;
- candidate-policy, feature-schema, and validation-gate fingerprints;
- model, validation evidence, bundle manifest, semantic cache, latent artifact,
  and dependency SHA-256 values;
- dependency versions and platform identity;
- exact output path and identity-derived marker path;
- the single authorized command.

The draft output location is
`reports/promotion/v2-frozen-promotion-manifest.json`. Creating a final manifest
must refuse overwrite.

## Preflight boundary

Preflight may inspect Git state, configs, model/evidence/bundle hashes, dataset
identity, expected case fingerprint already registered in configuration, output
absence, and marker absence. It must not call `load_cases`, open the frozen case
file, derive labels, create a marker, or create result output.

At P0 handoff, the exact preflight and execution commands will be printed with
the final immutable paths and fingerprints. The execution command will use the
repository's protected `evaluate-ranker` path only after a separate user
authorization.

## One-shot execution rules after future authorization

1. Verify the promotion manifest and run label-free preflight.
2. Atomically create a `started` marker before loading frozen labels.
3. Execute exactly once and write aggregate JSON plus command/log sidecars.
4. Atomically finish the marker as `completed` or `failed`, including a result
   or failure-log digest.
5. A `started`, `completed`, or `failed` marker permanently blocks another run.
6. A poor result, exception, or crash never authorizes tuning or a rerun.
7. Confirmation-B remains the main resume result; the 50-case frozen result is
   labeled as a small final stress test.

The current state remains **unapproved and unconsumed**.

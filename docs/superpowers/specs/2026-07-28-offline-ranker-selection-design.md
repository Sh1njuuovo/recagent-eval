# Offline Ranker Selection Design

## Goal

Improve second-stage ranking without spending LLM tokens or selecting against
the frozen test set. The experiment compares deterministic fusion methods on
the existing MovieLens-1M validation split. Every validation result is retained,
but a new ranker may be evaluated on the frozen test cases only if its
validation NDCG@10 strictly exceeds the same-depth ItemCF ranker.

This iteration does not change preference extraction, tool planning, retrieval,
hard constraints, formal cases, candidate depth, or the DeepSeek provider.

## Motivation

The constraint-aware DeepSeek matrix showed that the full hybrid route raised
union candidate recall from 0.78 to 0.88 relative to the same-depth structured
ItemCF variant, while Recall@10 fell from 0.06 to 0.04 and NDCG@10 fell from
0.0360 to 0.0149. The candidate trace therefore identifies a ranking failure:
TF-IDF contributes useful candidates, but independently min-max-normalized
scores are not directly comparable.

The next experiment must test scale-robust fusion before adding a learned
ranker. A learned ranker is excluded from this iteration because the validation
labels are sparse, it adds dependency and overfitting risk, and simpler
rank-based calibration has not yet been falsified.

## Compared Rankers

All rankers receive the same hard-filtered Top-500 ItemCF and TF-IDF candidate
lists and produce a deterministic Top-10 result. Movie ID ascending is the
final tie-breaker.

### ItemCF

The control ranks by raw ItemCF score descending. Candidates absent from the
ItemCF route are not promoted by the semantic route. This is the same-depth
baseline that a new fusion method must beat.

### Min-max linear

The existing control independently min-max normalizes ItemCF and TF-IDF scores,
then combines them with the validation-selected weights. It remains in the
ablation table but is not treated as a new method.

### Reciprocal Rank Fusion

RRF converts each present route rank to:

```text
1 / (k + rank)
```

and sums the route contributions. Missing-route contribution is zero. The
validation search considers `k` in `{10, 30, 60, 100}`. RRF is the primary
candidate because it ignores incompatible raw score scales while rewarding
items supported by both routes.

### Percentile fusion

Each route converts its ordered candidates into a descending percentile in
`[0, 1]`; the top item receives 1 and the last receives 0. A single-item route
assigns that item 1. Missing-route contribution is zero. Validation searches
ItemCF/semantic weights on the existing 0.1 simplex grid, with no explicit
affinity term in this iteration.

## Interfaces and Configuration

Introduce a ranker specification with these public fields:

```yaml
ranker:
  kind: rrf
  rrf_k: 60
  weights: [0.7, 0.3]
```

Supported kinds are `itemcf`, `minmax_linear`, `rrf`, and
`percentile_linear`. Only parameters used by the selected kind affect ranking.
Legacy top-level three-way `weights` remain readable and map to
`minmax_linear`, so existing formal configurations continue to reproduce their
recorded results.

The ranking module exposes one dispatcher that accepts raw per-route score
dictionaries and returns the existing `RecommendedMovie` objects. Score
breakdowns retain normalized route contributions and final score so traces and
the demo remain interpretable.

## Validation Data Flow

The selector reuses the chronological MovieLens split and the fixed retrieval
policy:

1. Fit ItemCF only on training interactions.
2. Build semantic profiles from allowed training history, capped at 50 items.
3. Retrieve both routes at the frozen Top-500 depth.
4. Apply each ranker to the identical candidate inputs.
5. Measure validation Recall@10, NDCG@10, HitRate@10, route candidate recall,
   and latency.
6. Select only among `itemcf`, `rrf`, and `percentile_linear`, by validation
   NDCG@10, then Recall@10, then the fixed conservative method order
   `itemcf`, `rrf`, `percentile_linear`; parameter values provide the final
   deterministic tie-break. The existing `minmax_linear` row remains an
   informative control but is not eligible for selection.
7. Persist every row, the selected configuration, and the gate decision.

The fixed method order prevents a tie from being presented as an improvement.
`test_unlocked` is true only when the selected RRF or percentile ranker's
NDCG@10 is strictly greater than ItemCF by more than `1e-12`.

## Frozen-Test Gate

The offline selection command writes:

- `artifacts/ranker_ablation.json`, containing all validation rows;
- `artifacts/selected_ranker.yaml`, containing the selected configuration;
- `test_unlocked`, the ItemCF score, selected score, and numeric margin.

The test-evaluation command must read this evidence file and refuse to run when
`test_unlocked` is false. It must also verify that candidate depth, semantic
history cap, data fingerprint, and selected ranker parameters match the evidence
record. The first successful unlocked run is the only test result used for this
iteration; subsequent reruns may verify determinism but may not select a new
method.

If no fusion method beats ItemCF, the iteration ends with a negative ablation.
No DeepSeek matrix is rerun, and no test metric is generated for a rejected
ranker.

## Failure Handling

- Empty route: contribute zero for that route; ItemCF control returns an empty
  result if ItemCF itself is empty.
- Single-item route: assign percentile 1 to avoid division by zero.
- Non-finite score: reject the ranker input with a clear error.
- Unknown ranker kind or invalid parameter: reject configuration before data
  loading.
- Evidence/config mismatch: refuse frozen-test evaluation and report all
  mismatched fields.
- Validation tie: prefer the fixed conservative method order; a tie with ItemCF
  never unlocks testing.

## Tests

Unit tests cover:

- exact RRF scores and deterministic tie-breaking;
- percentile calibration for empty, singleton, tied, and ordinary routes;
- missing-route behavior;
- rejection of non-finite scores and invalid ranker configuration;
- selection ordering and the strict-improvement threshold;
- backward compatibility for existing top-level weights.

Integration tests cover:

- one small MovieLens validation selection run that writes all evidence files;
- a rejected ranker being unable to start frozen-test evaluation;
- an unlocked synthetic ranker passing the evidence/config consistency check;
- deterministic repeated selection with the same seed and input fingerprint.

Existing tests must continue to pass, and the checked-in DeepSeek aggregate and
formal configuration must not change.

## Acceptance Criteria

- ItemCF, current min-max linear, RRF, and percentile fusion appear in the
  versioned validation ablation.
- Selection uses validation labels only and is deterministic.
- A tie or regression does not unlock the frozen test.
- An improvement unlocks exactly the selected configuration, with a recorded
  numeric margin and reproducibility fields.
- Existing formal DeepSeek configurations reproduce their current ranker
  behavior.
- The experiment truthfully reports either a validation improvement or a
  negative result; it does not promise a test or DeepSeek gain.

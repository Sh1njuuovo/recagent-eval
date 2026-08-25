# v2 Ranking Diagnosis and Calibration Design

## Status

Approved scope: keep the `semantic.top_k=1500` candidate policy fixed and
change only ranking inputs, ranking targets, or score calibration. The frozen
test remains locked until the existing LambdaMART promotion gate passes.

## Objective

Identify why LambdaMART fails to improve Top-10 NDCG after candidate union
recall reaches 87.8%, then evaluate a small set of leakage-safe ranking-input
variants on the same 500-user validation protocol. Every variant must be
reproducible from a YAML config and a machine-readable evidence file.

## Design

### Stage 1: Diagnostic evidence

Add a deterministic `diagnose-ranker` CLI path that rebuilds validation
candidate queries under `configs/v2_dense_recall1500.yaml` and writes a new
JSON artifact. It must report:

- candidate recall by route and target presence;
- ItemCF and LambdaMART Top-10 ranks, restricted and unrestricted to users
  whose target is in the candidate union;
- per-feature target-vs-negative separation on candidate-present users;
- route score/rank scale summaries and pairwise route agreement; and
- data, candidate-policy, feature-schema, model, and case fingerprints.

The diagnostic path may inspect validation targets only. It must not read frozen
test targets, authorize a frozen marker, mutate existing artifacts, or write
credentials/taste fields.

### Stage 2: Calibrated ranking variants

Add an explicit ranker feature calibration setting with a backward-compatible
`raw` default and a validation-only `percentile` option. Percentile calibration
maps each route score to its within-user rank percentile before feature-row
construction; route membership and reciprocal-rank features remain unchanged.
The calibration name is included in the feature/config/candidate fingerprints,
so artifacts from different calibration modes cannot be mixed.

The first comparison contains exactly:

1. the existing LambdaMART feature path (`raw`),
2. percentile-calibrated route scores (`percentile`), and
3. an ItemCF baseline using the same `top1500` union candidate policy.

No broad hyperparameter search is added. Existing grouped three-fold CV,
2,000 paired bootstrap samples, constraint recomputation, and the strict
promotion gate are reused unchanged.

## Data flow

```text
top1500 config + legal histories
  -> fixed union candidate queries
  -> diagnostic artifact (stage 1)
  -> raw/percentile feature rows
  -> grouped LambdaMART CV
  -> per-user validation evidence
  -> existing promotion gate
```

## Failure handling

- A target absent from the union is reported as a retrieval miss and excluded
  from feature-separation denominators, while remaining in aggregate ranking
  denominators.
- Empty routes produce zero score, zero reciprocal rank, and zero percentile.
- Non-finite scores or features fail with user/movie/feature context.
- Existing v1/v2 raw artifacts remain readable; calibration mismatches fail
  closed with a fingerprint error.
- If percentile calibration fails the gate, the negative result is retained and
  no frozen or LLM evaluation is launched.

## Verification

Focused tests cover diagnostic denominators, target-present conditioning,
percentile tie handling, empty routes, fingerprint separation, and CLI refusal
to overwrite evidence. Then run the 30-user stability training, the 500-user
validation training for both variants, the full pytest/coverage/Ruff/lock/diff
gate, and inspect all generated JSON before making any promotion claim.

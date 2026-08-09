# Offline ranker selection

## Decision

The frozen test remained locked. Neither Reciprocal Rank Fusion nor a genuine
two-route percentile fusion strictly exceeded the same-depth ItemCF control on
validation NDCG@10.

- Dataset: MovieLens-1M chronological validation split
- Validation users: 500
- Candidate depth: Top-500 per route
- Semantic profile history cap: 50
- Data/config fingerprint:
  `4c7b82853d31f6518ac2c6abe094ff5cd29d5e956405a7e32afec5197a22bc31`
- Selection metric: validation NDCG@10, then Recall@10
- Gate: strict improvement over ItemCF by more than `1e-12`
- Frozen-case evaluation: not run
- DeepSeek calls: 0

## Validation ablation

Every row uses the same raw ItemCF and TF-IDF candidate inputs. Candidate
recall is therefore constant across fusion methods: ItemCF 0.696, semantic
0.408, and union 0.818.

| Method | Parameters | Recall@10 | NDCG@10 | HitRate@10 | ms/user |
| --- | --- | ---: | ---: | ---: | ---: |
| ItemCF | — | 0.064 | 0.033388 | 0.064 | 0.97 |
| Existing min-max linear | 0.7 / 0.3 | **0.072** | **0.036600** | **0.072** | 1.94 |
| RRF | k=10 | 0.046 | 0.027275 | 0.046 | 1.96 |
| RRF | k=30 | 0.056 | 0.031145 | 0.056 | 1.99 |
| RRF | k=60 | 0.060 | 0.031148 | 0.060 | 2.02 |
| RRF | k=100 | 0.054 | 0.029138 | 0.054 | 2.01 |
| Percentile | 1.0 / 0.0 | 0.064 | 0.033388 | 0.064 | 2.05 |
| Percentile | 0.9 / 0.1 | 0.058 | 0.029542 | 0.058 | 2.18 |
| Percentile | 0.8 / 0.2 | 0.066 | 0.032392 | 0.066 | 2.05 |
| Percentile | 0.7 / 0.3 | 0.060 | 0.028361 | 0.060 | 2.07 |
| Percentile | 0.6 / 0.4 | 0.060 | 0.029485 | 0.060 | 2.10 |
| Percentile | 0.5 / 0.5 | 0.060 | 0.029717 | 0.060 | 2.08 |
| Percentile | 0.4 / 0.6 | 0.054 | 0.028319 | 0.054 | 2.10 |
| Percentile | 0.3 / 0.7 | 0.052 | 0.027458 | 0.052 | 2.06 |
| Percentile | 0.2 / 0.8 | 0.050 | 0.027579 | 0.050 | 2.08 |
| Percentile | 0.1 / 0.9 | 0.052 | 0.026686 | 0.052 | 2.08 |
| Percentile | 0.0 / 1.0 | 0.028 | 0.012205 | 0.028 | 2.06 |

Latency is local wall-clock ranking time and is reported for completeness; it
is not a selection criterion.

## Why the old min-max row did not unlock a test rerun

The existing 0.7/0.3 min-max control reaches validation NDCG@10 0.036600, but
it is not a new candidate. That exact family selected the current formal hybrid
and has already produced the retained negative DeepSeek test result
(`NDCG@10=0.0149`). Allowing it to unlock another test run would reuse a known
validation winner after observing its test failure.

The preregistered selectable set therefore contains ItemCF, RRF, and percentile
fusion only. Percentile 1.0/0.0 ties ItemCF exactly, while the best genuine
percentile fusion (0.8/0.2) reaches 0.032392 and the best RRF configuration
(k=60) reaches 0.031148. Neither passes the strict gate.

## Interpretation

Rank-only calibration did not solve the ranking bottleneck. Increasing the
semantic contribution consistently harms NDCG, despite semantic retrieval
raising union candidate coverage. This supports a more specific diagnosis:
route membership and rank alone do not identify which semantic-only candidates
deserve promotion.

The next justified experiment is a small validation-trained reranker using raw
route scores, route ranks, popularity, and genre-overlap features with explicit
regularization. It should use nested validation or cross-validation before any
request to unlock the frozen cases. No DeepSeek rerun is justified by this
ablation.

## Reproduction

```bash
uv run recagent-eval select-ranker \
  --config configs/full_constraint_aware.yaml \
  --data-dir data/raw/ml-1m \
  --evidence-output artifacts/ranker_ablation.json \
  --config-output configs/full_ranker_selected.yaml \
  --max-users 500
```

The command writes all validation rows to
`artifacts/ranker_ablation.json`. Because the gate is locked, it does not write
`configs/full_ranker_selected.yaml` and `evaluate-ranker` must not be run.

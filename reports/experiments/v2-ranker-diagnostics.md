# v2 ranker diagnostics — raw recall-1500 model

## Status

Read-only diagnostic over the 500 validation users under the
`semantic.top_k=1500` candidate policy, using the raw-feature LambdaMART model
from `artifacts/experiments/v2-recall-1500/`. The diagnostic only reads
validation targets, never creates a frozen authorization marker, and the frozen
test remains locked.

- Date: 2026-08-22
- Config: `configs/v2_dense_recall1500.yaml`
- Model checksum: `3e12b62708d96b1c3edfe7ec05226ebe9376d2d0ca4f9101ee674072db5b1a24`
- Evidence: `artifacts/experiments/v2-ranker-diagnostics/diagnostics.json`

## Summary (500 users, 439 with the target in the candidate union)

| Metric | Value |
| --- | ---: |
| Union candidate recall | 0.878 |
| ItemCF candidate recall | 0.696 |
| Dense candidate recall | 0.612 |
| ItemCF top-10 hit (all / present) | 0.064 / 0.073 |
| LambdaMART top-10 hit (all / present) | 0.066 / 0.075 |
| ItemCF NDCG@10 (all / present) | 0.0334 / 0.0380 |
| LambdaMART NDCG@10 (all / present) | 0.0299 / 0.0341 |

Target rank quantiles (present users):

| Route | p25 | p50 | p75 |
| --- | ---: | ---: | ---: |
| ItemCF | 53 | 172 | 430 |
| LambdaMART | 57 | 173 | 374 |

## Feature separation (target mean minus negative mean, present users)

| Feature | Separation |
| --- | ---: |
| itemcf_score | 8.800 |
| log1p_popularity | 2.085 |
| in_itemcf | 0.511 |
| preference_affinity | 0.053 |
| history_year_match | 0.040 |
| itemcf_reciprocal_rank | 0.035 |
| history_genre_jaccard | 0.026 |
| dense_reciprocal_rank | 0.004 |
| dense_score | −0.050 |
| in_dense | −0.149 |

## Interpretation

The target typically sits deep in both rankings (median rank ~172 out of roughly
2,000 union candidates), so only ~7% of present users can reach the top 10.
ItemCF score and popularity separate targets from negatives far better than any
dense feature; the dense score and dense-membership features are actually
anti-predictive on this validation set. This explains why the raw LambdaMART
barely changes hit rate (7.5% vs 7.3% on present users) and lowers NDCG: it adds
no ranking signal beyond ItemCF at the top of the list.

# Strong-baseline comparison — cohort `confirmation_b`

- Users: 1000
- Pairwise: 2,000 paired bootstrap, seed 42, NDCG@10 deltas
- Evidence identity: **sole final certification cohort**; seed 42 is the formal
  main result.

| Method | Recall@10 | NDCG@10 | MRR@10 | Candidate recall | Constraints |
| --- | ---: | ---: | ---: | ---: | ---: |
| als_direct | 0.0690 | 0.0323 | 0.0214 | 1.0000 | 1.0000 |
| bpr_mf | 0.0470 | 0.0205 | 0.0126 | 1.0000 | 1.0000 |
| current_v2b | 0.1180 | 0.0555 | 0.0368 | 0.9210 | 1.0000 |
| itemcf_direct | 0.0640 | 0.0323 | 0.0228 | 1.0000 | 1.0000 |
| lightgcn | 0.0350 | 0.0163 | 0.0106 | 1.0000 | 1.0000 |
| popularity | 0.0440 | 0.0187 | 0.0112 | 1.0000 | 1.0000 |

## Pairwise NDCG@10 (delta, 95% CI)

| Pair | Mean | 95% CI |
| --- | ---: | ---: |
| bpr_mf_vs_als_direct | -0.0118 | [-0.0214, -0.0031] |
| current_v2b_vs_als_direct | 0.0232 | [0.0111, 0.0346] |
| current_v2b_vs_bpr_mf | 0.0350 | [0.0234, 0.0469] |
| itemcf_direct_vs_als_direct | 0.0000 | [-0.0108, 0.0099] |
| itemcf_direct_vs_bpr_mf | 0.0119 | [0.0036, 0.0202] |
| itemcf_direct_vs_current_v2b | -0.0231 | [-0.0346, -0.0118] |
| lightgcn_vs_als_direct | -0.0160 | [-0.0254, -0.0075] |
| lightgcn_vs_bpr_mf | -0.0042 | [-0.0069, -0.0016] |
| lightgcn_vs_current_v2b | -0.0392 | [-0.0511, -0.0276] |
| lightgcn_vs_itemcf_direct | -0.0160 | [-0.0247, -0.0080] |
| popularity_vs_als_direct | -0.0136 | [-0.0233, -0.0050] |
| popularity_vs_bpr_mf | -0.0018 | [-0.0044, 0.0006] |
| popularity_vs_current_v2b | -0.0368 | [-0.0485, -0.0252] |
| popularity_vs_itemcf_direct | -0.0137 | [-0.0216, -0.0061] |
| popularity_vs_lightgcn | 0.0023 | [0.0005, 0.0044] |

## Success criteria evaluation (Confirmation-B, certification)

Directional deltas are **current_v2b minus the other method**:

| Comparison | Delta | 95% CI | Significant |
| --- | ---: | ---: | --- |
| vs ItemCF direct | +0.0231 | [0.0118, 0.0346] | yes |
| vs ALS direct | +0.0232 | [0.0111, 0.0346] | yes |
| vs BPR-MF | +0.0350 | [0.0234, 0.0469] | yes |
| vs LightGCN | +0.0392 | [0.0276, 0.0511] | yes |
| vs Popularity | +0.0368 | [0.0252, 0.0485] | yes |

**Success A — certified on Confirmation-B:**

1. NDCG@10 > ItemCF with 95% CI lower bound > 0: 0.0555 vs 0.0323,
   CI [0.0118, 0.0346] → **pass**.
2. Beats at least one strong baseline (ALS direct / BPR-MF) with the same CI
   rule: beats both ALS direct and BPR-MF → **pass**.
3. Recall@10 does not regress: 0.118 vs ItemCF 0.064 (+84% relative) →
   **pass**.
4. Constraint satisfaction: 100% → **pass**.

## Recorded effect and costs (peak RSS excluded)

| Method | NDCG@10 | Recall@10 | Train (s) | Latency p50/p95 (ms) | Model size (MB) | CPU-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current_v2b | 0.0555 | 0.118 | 1675.5 | 110.1 / 114.1 | 1.2 | yes |
| ALS direct | 0.0323 | 0.069 | 2.1 | 0.8 / 0.9 | 0.3 | yes |
| ItemCF direct | 0.0323 | 0.064 | 14.5 | 14.7 / 75.0 | 208.4 | yes |
| BPR-MF | 0.0205 | 0.047 | 24.7 | 1.1 / 1.2 | 0.6 | yes |
| Popularity | 0.0187 | 0.044 | 0.0 | 0.7 / 0.7 | 0.03 | yes |
| LightGCN | 0.0163 | 0.035 | 2220.1 | 1.2 / 1.3 | 2.4 | yes |

The table documents recorded training time, latency, and model size for the
project-local implementations. Peak RSS is excluded because the historical
field had a platform-unit bug and the runs were not guaranteed to share an
independent-process measurement protocol. See the correction addendum.

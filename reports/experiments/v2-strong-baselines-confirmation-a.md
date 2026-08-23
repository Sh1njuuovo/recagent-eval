# Strong-baseline comparison — cohort `confirmation_a`

- Users: 1000
- Pairwise: 2,000 paired bootstrap, seed 42, NDCG@10 deltas

| Method | Recall@10 | NDCG@10 | MRR@10 | Candidate recall | Constraints |
| --- | ---: | ---: | ---: | ---: | ---: |
| als_direct | 0.0980 | 0.0440 | 0.0283 | 1.0000 | 1.0000 |
| bpr_mf | 0.0530 | 0.0253 | 0.0170 | 1.0000 | 1.0000 |
| current_v2b | 0.0990 | 0.0524 | 0.0384 | 0.9210 | 1.0000 |
| itemcf_direct | 0.0750 | 0.0378 | 0.0265 | 1.0000 | 1.0000 |
| lightgcn | 0.0460 | 0.0235 | 0.0168 | 1.0000 | 1.0000 |
| popularity | 0.0510 | 0.0257 | 0.0182 | 1.0000 | 1.0000 |

## Pairwise NDCG@10 (delta, 95% CI)

| Pair | Mean | 95% CI |
| --- | ---: | ---: |
| bpr_mf_vs_als_direct | -0.0187 | [-0.0295, -0.0085] |
| current_v2b_vs_als_direct | 0.0084 | [-0.0039, 0.0207] |
| current_v2b_vs_bpr_mf | 0.0271 | [0.0146, 0.0399] |
| itemcf_direct_vs_als_direct | -0.0063 | [-0.0160, 0.0030] |
| itemcf_direct_vs_bpr_mf | 0.0124 | [0.0042, 0.0209] |
| itemcf_direct_vs_current_v2b | -0.0146 | [-0.0267, -0.0024] |
| lightgcn_vs_als_direct | -0.0205 | [-0.0317, -0.0099] |
| lightgcn_vs_bpr_mf | -0.0018 | [-0.0052, 0.0017] |
| lightgcn_vs_current_v2b | -0.0289 | [-0.0415, -0.0156] |
| lightgcn_vs_itemcf_direct | -0.0142 | [-0.0228, -0.0057] |
| popularity_vs_als_direct | -0.0183 | [-0.0294, -0.0075] |
| popularity_vs_bpr_mf | 0.0004 | [-0.0034, 0.0042] |
| popularity_vs_current_v2b | -0.0267 | [-0.0398, -0.0135] |
| popularity_vs_itemcf_direct | -0.0121 | [-0.0204, -0.0037] |
| popularity_vs_lightgcn | 0.0022 | [0.0003, 0.0043] |

## Success criteria evaluation (Confirmation-A)

Pairwise deltas are reported as **row-method minus column-method**. Key
directional deltas for `current_v2b`:

| Comparison | Delta (current − other) | 95% CI | Significant |
| --- | ---: | ---: | --- |
| vs ItemCF direct | +0.0146 | [0.0024, 0.0267] | yes (lower > 0) |
| vs ALS direct | +0.0084 | [−0.0039, 0.0207] | no |
| vs BPR-MF | +0.0271 | [0.0146, 0.0399] | yes |
| vs LightGCN | +0.0289 | [0.0156, 0.0415] | yes |
| vs Popularity | +0.0267 | [0.0135, 0.0398] | yes |

**Success A (effect win) — met on Confirmation-A:**

1. NDCG@10 > ItemCF with paired-bootstrap 95% CI lower bound > 0:
   current_v2b 0.0524 vs ItemCF 0.0378, CI [0.0024, 0.0267] → **pass**.
2. Beats at least one strong collaborative baseline (ALS direct or BPR-MF)
   with the same CI rule: beats BPR-MF (CI [0.0146, 0.0399]) → **pass** (ALS
   difference is positive but not significant).
3. Recall@10 does not regress: 0.099 vs ItemCF 0.075 (+32% relative) → **pass**.
4. Constraint satisfaction: 100% for every method → **pass**.

The current method was not changed as a result of Confirmation-A (the fixes
made during baseline development corrected baseline implementation bugs and
the dev-selection denominator; the current_v2b config and training cohort were
pre-registered). Confirmation-B runs next to certify the conclusion on a
second untouched cohort, per the pre-registered protocol.

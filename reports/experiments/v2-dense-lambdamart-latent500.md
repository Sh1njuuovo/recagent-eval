# v2 dense LambdaMART validation — latent route with route-balanced hard negatives (500 users)

## Status

**Negative result, preserved.** The 500-user validation with schema-v2 latent
features and the pre-registered route-balanced hard-negative policy
(`max_negatives=200`, `negative_policy=route_balanced`) fails the formal gate:
the learned ranker produces zero Top-10 hits. Constraint satisfaction stays
100% and the frozen test stays **locked**.

- Date: 2026-08-23
- Config: `configs/v2_dense_latent.yaml`
- Wall time: 395.12 s
- Dataset fingerprint: `0d2c756a...`
- Candidate-policy fingerprint: `30b325ed...`
- Feature schema fingerprint (v2): `4f4c4a4b...`
- Evidence fingerprint: `289bfde7...`
- Model checksum: `c91edbee...`

## Gate conditions

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0 vs 0.0334 | no |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0461 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

## Measured metrics (500 validation users, 472 training groups)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 / recall@10 / hit@10 | 0.0 / 0.0 / 0.0 |
| ItemCF mean NDCG@10 / recall@10 / hit@10 | 0.0334 / 0.064 / 0.064 |
| Mean NDCG@10 delta | −0.0334 |
| Bootstrap 95% CI (delta) | [−0.0461, −0.0213] |
| Union candidate recall | 0.928 |
| ItemCF / Dense candidate recall | 0.696 / 0.612 |
| Constraint satisfaction rate | 1.000 |

Selected CV parameters: `learning_rate=0.05, min_child_samples=50,
n_estimators=100, num_leaves=15`.

## Root-cause diagnosis (attribution, same validation users)

A 200-user controlled comparison isolated the failure to the negative-sampling
policy, not the latent features:

| Training variant | Validation hit@10 | Median target rank |
| --- | ---: | ---: |
| v1 features, all negatives | 0.061 | 170 |
| v2 latent features, all negatives | 0.084 | 124 |
| v2 latent features, route-balanced hard negatives (200) | 0.000 | 1607 |

The route-balanced hard-negative training sets contain only the target plus the
200 highest-scoring candidates. The model learns a boundary inside that biased
subset that does not transfer to full-union ranking, so the target is ranked
deeply at validation. The latent features themselves improve ranking depth
(median 124 vs 170 and hit@10 0.084 vs 0.061 on the 200-user slice), which
matches the candidate-stage diagnosis. This negative result is preserved;
the frozen test stays locked.

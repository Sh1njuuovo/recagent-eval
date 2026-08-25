# v2 dense LambdaMART validation — latent route, all negatives (500 users)

## Status

**Negative result, preserved.** Attribution control (E5): the schema-v2 latent
features with the default all-negatives sampling (`negative_policy=all`) no
longer collapse (0 hits), but the formal gate still fails: LambdaMART
NDCG@10 (0.0258) is below ItemCF (0.0334) and the paired-bootstrap 95% CI
crosses zero. Constraint satisfaction stays 100%; the frozen test stays
**locked**.

- Date: 2026-08-23
- Config: `configs/v2_dense_latent_allneg.yaml`
- Wall time: 803.64 s
- Dataset fingerprint: `0d2c756a...`
- Candidate-policy fingerprint: `dce49623...`
- Feature schema fingerprint (v2): `4f4c4a4b...`
- Evidence fingerprint: `f2f13994...`
- Model checksum: `e4d38558...`

## Gate conditions

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0258 vs 0.0334 | no |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0214 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

## Measured metrics (500 validation users, 472 training groups)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 / recall@10 / hit@10 | 0.0258 / 0.060 / 0.060 |
| ItemCF mean NDCG@10 / recall@10 / hit@10 | 0.0334 / 0.064 / 0.064 |
| Mean NDCG@10 delta | −0.0076 |
| Bootstrap 95% CI (delta) | [−0.0214, 0.0073] |
| Union candidate recall | 0.928 |
| Constraint satisfaction rate | 1.000 |

Selected CV parameters: `learning_rate=0.03, min_child_samples=50,
n_estimators=200, num_leaves=15`.

## Interpretation

Compared with E4 (route-balanced hard negatives, 0 hits), the all-negatives
training restores normal ranker behavior and the latent route's candidate
signal (union recall 0.928), but Top-10 ranking still does not beat the fixed
ItemCF baseline with confidence. The negative result is preserved; the frozen
test stays locked.

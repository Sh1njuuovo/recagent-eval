# v2 dense LambdaMART validation — latent v2b features (500 users)

## Status

**Negative result, preserved (final contingency E6).** The schema-v2b latent
features (`itemcf_latent_cross`, `recent_itemcf_score`, `year_recency`) with
all-negatives sampling produce the best learned ranking so far: mean NDCG@10
0.0446 beats ItemCF 0.0334 and recall@10 rises to 0.102 (vs 0.064), but the
pre-registered formal gate still fails because the paired-bootstrap 95% CI
lower bound (−0.0024) crosses zero. Constraint satisfaction stays 100%; the
frozen test stays **locked**. Per the plan, E6 is the last ranker experiment;
no further variants are run.

- Date: 2026-08-23
- Config: `configs/v2_dense_latent_bfeat.yaml`
- Wall time: 5245.62 s (elapsed; 801 s user)
- Dataset fingerprint: `0d2c756a...`
- Candidate-policy fingerprint: `88f3288b...`
- Feature schema fingerprint (v2b): `70de3e17...`
- Evidence fingerprint: `2d2d0748...`
- Model checksum: `0600629f...`

**Sampling note:** the plan's E6 description inherited `route_balanced` hard
negatives from the base config, but E4 and a controlled 200-user attribution
diagnosis proved that policy catastrophically breaks generalization (0 hits,
median target rank 1607). E6 therefore runs with the default all-negatives
sampling (`negative_policy=all`) so the v2b features are evaluated on the
working axis. The thresholds, seed, user set, and metric definitions are
unchanged.

## Gate conditions

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0446 vs 0.0334 | yes |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0024 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

## Measured metrics (500 validation users, 472 training groups)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 / recall@10 / hit@10 | 0.0446 / 0.102 / 0.102 |
| ItemCF mean NDCG@10 / recall@10 / hit@10 | 0.0334 / 0.064 / 0.064 |
| Mean NDCG@10 delta | +0.0112 |
| Bootstrap 95% CI (delta) | [−0.0024, 0.0280] |
| Union candidate recall | 0.928 |
| Constraint satisfaction rate | 1.000 |

Selected CV parameters: `learning_rate=0.05, min_child_samples=20,
n_estimators=100, num_leaves=15`.

## Interpretation

The latent route plus the three v2b features moves the held-out target into
Top-10 for 51/500 users (ItemCF 32/500) and lifts NDCG@10 by 0.0112, but the
2,000-sample paired bootstrap interval still includes zero at its lower tail.
The result is retained as evidence and the frozen test remains locked; the
remaining gap to the gate is statistical confidence at Top-10, not candidate
coverage (0.928) or constraint satisfaction (1.0).

# v2 dense LambdaMART validation — percentile-calibrated recall-1500

## Status

Second calibrated variant of the recall-1500 candidate policy. The two raw
route scores were replaced by within-user competition-rank percentiles before
feature-row construction (`ranker.score_calibration=percentile`). The 500-user
validation completed with constraint satisfaction 100%, but the ranking gate
failed **worse** than the raw variant: percentile calibration removes the
scale information in the strongly separating ItemCF score, so LambdaMART
recall@10 drops from 0.066 to 0.038. The frozen test stays **locked** and the
negative result is preserved.

- Date: 2026-08-22
- Config: `configs/v2_dense_recall1500_percentile.yaml`
- Seed: 42; 2,000 paired-bootstrap resamples
- Dataset fingerprint:
  `0d2c756a78f02a8ea8680bd216364a6ab7afffc2ae1a26df0611591147897eac`
- Evidence fingerprint:
  `be72a00fcacee23a6fbbf8a49482d67789afbac0fea9040b3ff1d3962e9e588b`
- Model checksum: `87dd45de49c5bc8de46d4bda9b5f558a16b57d861197f5b5dfb52d03d3092b87`

## Gate conditions

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0207 vs 0.0334 | no |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0266 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

## Comparison with the raw recall-1500 variant

| Metric | Raw | Percentile |
| --- | ---: | ---: |
| LambdaMART NDCG@10 | 0.0299 | 0.0207 |
| LambdaMART recall@10 | 0.066 | 0.038 |
| Mean NDCG delta | −0.0035 | −0.0126 |
| Bootstrap 95% CI | [−0.0190, 0.0111] | [−0.0266, 0.0004] |

## Interpretation

Percentile calibration compresses the raw ItemCF score — the feature with by
far the strongest target-vs-negative separation (8.8 on the raw scale) — into a
uniform [0, 1] rank percentile. The learned trees lose the magnitude signal
that separates strongly preferred candidates, so both hit rate and NDCG
degrade. The calibrated variant is retained as a falsified hypothesis rather
than a tuning success; the next experiments should keep raw score magnitude and
instead address candidate-side ranking depth or richer separating features.

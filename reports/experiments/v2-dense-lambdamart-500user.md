# v2 dense LambdaMART validation — 500 users

## Status

Formal local validation run of the leakage-safe LambdaMART pipeline with dense
semantic retrieval. The run completes end to end; the frozen-test gate stays
**locked** because the learned ranker does not beat ItemCF with statistical
confidence on this 500-user evaluation set. The negative result is preserved as
valid internship evidence and identifies the candidate-recall-to-Top-10-ranking
bottleneck.

- Date: 2026-08-22
- Platform: macOS 15.7.4 arm64
- Python: 3.13.11 (project supports 3.11+)
- Data: MovieLens 1M, 3,883 movies, 1,000,209 ratings
- Config: `configs/v2_dense_validation.yaml` (dense `all-MiniLM-L6-v2`, no taste)
- Seed: 42; 2,000 paired-bootstrap resamples
- LightGBM 4.7.0 / NumPy 2.5.1 / scikit-learn 1.9.0
- Dataset fingerprint:
  `0d2c756a78f02a8ea8680bd216364a6ab7afffc2ae1a26df0611591147897eac`
- Evidence fingerprint:
  `be36fa7447b7137aeab0b3596e7fd88cd45be860185f6ef08cb1f2f05dda6c3b`

## Gate conditions

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0327 vs 0.0334 | no |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0146 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

## Measured metrics (validation users = 500, training groups = 408)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 | 0.0327 |
| ItemCF mean NDCG@10 | 0.0334 |
| Mean NDCG@10 delta | −0.0007 |
| Bootstrap 95% CI (delta) | [−0.0146, 0.0129] |
| LambdaMART recall@10 | 0.064 |
| ItemCF recall@10 | 0.064 |
| LambdaMART hit@10 | 0.064 |
| ItemCF hit@10 | 0.064 |
| Union candidate recall | 0.776 |
| ItemCF candidate recall | 0.696 |
| Dense candidate recall | 0.288 |
| Constraint satisfaction rate | 1.000 |

Selected CV parameters: `learning_rate=0.03`, `min_child_samples=50`,
`n_estimators=100`, `num_leaves=15`. Model checksum
`b868417bf6c4d6ad397b07a787fad71b3ceec7040b523fd2d923ec0d3771e4e5`.

## Interpretation

The bottleneck is at candidate recall, not at the final ten-item ranking:

1. The held-out target is absent from the union candidate set for 22.4% of
   users, so no ranker can retrieve it. Dense semantic candidates alone cover
   only 28.8% of targets, and the dense path adds no recall beyond ItemCF's
   69.6%.
2. When the target is in the candidate union, both rankers place it inside the
   top 10 for the same 6.4% of users. LambdaMART neither recovers additional
   hits nor changes their order, so its NDCG@10 is statistically
   indistinguishable from ItemCF.
3. The CV-selected parameters favor the smallest model family
   (`num_leaves=15`, 100 trees), consistent with a feature set that does not yet
   separate relevant from irrelevant candidates at ranking depth 10.

The failure is valid evidence: the leakage-safe evaluation contract works and
the next bottleneck is candidate construction (union recall 77.6%) and dense
retrieval quality (28.8%), before any learned Top-10 reranker can show a
measurable gain.

## Artifacts

- Model bundle: `artifacts/experiments/v2-500/bundle.json`
- Validation evidence: `artifacts/experiments/v2-500/validation.json`
- Machine-readable summary: `v2-dense-lambdamart-500user.json`

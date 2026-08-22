# v2 dense LambdaMART validation — 30-user stability milestone

## Status

Intermediate stability run used to confirm that the LightGBM segmentation-fault
fix works end to end before the formal 500-user validation. The run completes
and publishes a model/evidence bundle. The 30-user margin is not statistically
significant and this report is **not** the frozen-test gate; see the 500-user
report for the formal negative result.

- Date: 2026-08-22
- Platform: macOS 15.7.4 arm64
- Python: 3.13.11
- Config: `configs/v2_dense_validation.yaml` (dense `all-MiniLM-L6-v2`, no taste)
- Seed: 42; 2,000 paired-bootstrap resamples
- Dataset fingerprint:
  `0d2c756a78f02a8ea8680bd216364a6ab7afffc2ae1a26df0611591147897eac`
- Evidence fingerprint:
  `81a00681a9613217839fc7b57ca8c9927e8816937dc68965ed95c5a36e56adb9`

## Measured metrics (validation users = 30, training groups = 22)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 | 0.0438 |
| ItemCF mean NDCG@10 | 0.0219 |
| Mean NDCG@10 delta | 0.0219 |
| Bootstrap 95% CI (delta) | [−0.0370, 0.1105] |
| Union candidate recall | 0.867 |
| Constraint satisfaction rate | 1.000 |

Selected CV parameters: `learning_rate=0.05`, `min_child_samples=20`,
`n_estimators=200`, `num_leaves=15`. Model checksum
`2d256bbcf32309bdbe793aa67b0dfda3d1592556d4ce1608fddba91f01d90ed7`.

## Interpretation

With only 30 validation users the paired-bootstrap interval is wide
([−0.0370, 0.1105]) and crosses zero, so this run proves pipeline stability
rather than ranking improvement. It is the fixed-crash reproduction of the
original 30-user blocker and validates that training, prediction with feature
contributions, model serialization, and bundle publication all succeed in the
torch-loaded dense process.

## Artifacts

- Model bundle: `artifacts/experiments/v2-30/bundle.json`
- Validation evidence: `artifacts/experiments/v2-30/validation.json`
- Machine-readable summary: `v2-dense-lambdamart-30user.json`

# v2 dense LambdaMART validation — recall-1500 candidate policy

## Status

Follow-up to the v2-500 validation. A candidate-recall sweep showed that the
dense route top-k is the dominant lever: raising `semantic.top_k` from 500 to
1500 lifts dense candidate recall from 0.288 to 0.612 and union recall from
0.776 to 0.878 on the same 500 validation users. The LambdaMART ranker was
retrained under the new candidate policy. Constraint satisfaction stays 100%,
but the ranking gate still fails: LambdaMART NDCG@10 (0.0299) does not beat
ItemCF (0.0334) and the paired-bootstrap 95% CI crosses zero. The frozen test
stays **locked**; the negative ranking result is preserved as evidence.

- Date: 2026-08-22
- Platform: macOS 15.7.4 arm64
- Python: 3.13.11
- Config: `configs/v2_dense_recall1500.yaml` (`semantic.top_k=1500`, no taste)
- Seed: 42; 2,000 paired-bootstrap resamples
- Dataset fingerprint:
  `0d2c756a78f02a8ea8680bd216364a6ab7afffc2ae1a26df0611591147897eac`
- Evidence fingerprint:
  `ed0e79d97f97278e6ab8848e00f67e3f042625fb06cab09e3d6890364d3cfc68`

## Candidate-recall sweep (500 validation users, ItemCF top-k fixed at 500)

| Variant | Dense recall | ItemCF recall | Union recall |
| --- | ---: | ---: | ---: |
| baseline-top500 | 0.288 | 0.696 | 0.776 |
| top250 | 0.178 | 0.696 | 0.748 |
| top750 | 0.362 | 0.696 | 0.802 |
| top1000 | 0.456 | 0.696 | 0.838 |
| **top1500 (selected)** | **0.612** | **0.696** | **0.878** |
| history20 | 0.270 | 0.696 | 0.774 |
| history100 | 0.288 | 0.696 | 0.776 |
| titles-only | 0.238 | 0.696 | 0.754 |
| titles-genres-year | 0.272 | 0.696 | 0.774 |

Query-construction variants (history cap, titles-only, year tokens) barely move
recall, so the dense top-k is the only material knob found in this sweep.

## Gate conditions under the recall-1500 policy

| Condition | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| 1 | LambdaMART mean NDCG@10 > ItemCF | 0.0299 vs 0.0334 | no |
| 2 | Paired-bootstrap 95% CI lower bound > 0 | −0.0190 | no |
| 3 | Recomputed constraint satisfaction = 100% | 100% | yes |

Additional measured metrics (500 validation users, 447 training groups):

| Metric | Value |
| --- | ---: |
| Dense candidate recall | 0.612 |
| ItemCF candidate recall | 0.696 |
| Union candidate recall | 0.878 |
| LambdaMART recall@10 | 0.066 |
| ItemCF recall@10 | 0.064 |

Selected CV parameters: `learning_rate=0.05`, `min_child_samples=50`,
`n_estimators=100`, `num_leaves=15`. Model checksum
`3e12b62708d96b1c3edfe7ec05226ebe9376d2d0ca4f9101ee674072db5b1a24`.

## Interpretation

Improving candidate recall did not improve ranking: the learned ranker still
does not beat ItemCF at Top-10 with statistical confidence. LambdaMART places
the target inside the top 10 for 6.6% of users (ItemCF 6.4%), but its NDCG is
lower, meaning the recalled targets that do surface are ranked below where
ItemCF ranks them. The bottleneck has moved from candidate construction
(recall now 87.8%) to feature quality and score calibration of the learned
ranker itself. The next experiment should change ranking inputs or calibration
(for example feature engineering, objective/target definition, or score
calibration) while keeping this candidate policy fixed.

## Artifacts

- Model bundle: `artifacts/experiments/v2-recall-1500/bundle.json`
- Validation evidence: `artifacts/experiments/v2-recall-1500/validation.json`
- Recall sweep evidence: `artifacts/experiments/v2-recall-sweep/recall.json`
- Machine-readable summary: `v2-dense-lambdamart-recall1500.json`

## Fingerprint note 2026-08-23

The recorded evidence config fingerprint `7b9373b4...` was produced by the
pre-`e1efee8` fingerprint payload (before `score_calibration` entered the
payload). Status: `legacy/non-replayable-under-current-code`. The current code
computes `3c0abb8c...` for the same YAML. The negative ranking result remains
valid evidence; the bundle cannot be replayed or consumed under current code.

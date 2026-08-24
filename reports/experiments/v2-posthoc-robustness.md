# Confirmation-B post-hoc robustness

This is a post-hoc randomness check for the project-local BPR-MF and LightGCN implementations. Seed 42 remains the formal Confirmation-B result. Seeds 7 and 2026 were fixed in the 2026-08-24 addendum after that run; they do not complete the original preregistration and do not alter hyperparameters, `current_v2b`, the cohort, metrics, or Success A thresholds.

Selected parameters were deterministically recovered after the original run and then locked. The recovery is marked `recovered_after_run` in `reports/evidence/baseline-parameter-recovery.json`; neither extra run executed a parameter grid.

| Method | Seed | Recall@10 | NDCG@10 | MRR@10 | Constraint |
|---|---:|---:|---:|---:|---:|
| BPR-MF | 42 (formal) | 0.047 | 0.02047 | 0.01257 | 1.000 |
| BPR-MF | 7 (post-hoc) | 0.044 | 0.02058 | 0.01359 | 1.000 |
| BPR-MF | 2026 (post-hoc) | 0.040 | 0.01789 | 0.01121 | 1.000 |
| LightGCN | 42 (formal) | 0.035 | 0.01631 | 0.01063 | 1.000 |
| LightGCN | 7 (post-hoc) | 0.043 | 0.01852 | 0.01128 | 1.000 |
| LightGCN | 2026 (post-hoc) | 0.041 | 0.01854 | 0.01188 | 1.000 |

| Method | Recall mean ± sample std | NDCG mean ± sample std | Worst Recall seed | Worst NDCG seed |
|---|---:|---:|---:|---:|
| BPR-MF | 0.04367 ± 0.00351 | 0.01965 ± 0.00152 | 2026 (0.040) | 2026 (0.01789) |
| LightGCN | 0.03967 ± 0.00416 | 0.01779 ± 0.00128 | 42 (0.035) | 42 (0.01631) |

All six runs have candidate recall and constraint satisfaction of 1.0. The additional seeds do not change the core claim: Success A is certified only by `current_v2b` against ItemCF direct and ALS direct on Confirmation-B.

LightGCN remains below the Confirmation-B Popularity result. Plausible contributors include the fixed CPU budget, fixed training epochs, and differences between this internal implementation and canonical library implementations. Claims therefore refer only to the project-local fixed-protocol implementation.

The uniform robustness input preserves each source artifact's schema, SHA-256, and fingerprint in `reports/evidence/posthoc-robustness-input.json`; its fingerprint is `a5bbae17582d3b231e98c8338cf81f54aa6eca5c341c9a4c76bc3a939c38aa9a`. The summary fingerprint is `6074e78c27467f2d1e4de1436d5aee747120d5e284323f820b08570b6f230faa`.

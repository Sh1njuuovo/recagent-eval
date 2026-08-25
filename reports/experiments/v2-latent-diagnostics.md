# v2 latent-route candidate diagnostics — 500 users

## Status

Candidate-stage diagnosis for the new weighted-ALS latent route under the
`semantic.top_k=1500` policy. All four pre-registered candidate gates
(G1–G4) **pass**, so the plan proceeds to the 30-user smoke (E3) and then the
500-user LambdaMART validation (E4). The frozen test remains locked.

- Date: 2026-08-23
- Config: `configs/v2_dense_latent.yaml`
- Dataset fingerprint (evidence-aligned): `0d2c756a...`
- Diagnostic dataset fingerprint: `57c3e8e9...`
- Case fingerprint: `bc2f622c...`
- Candidate-policy fingerprint: `30b325ed...`
- Feature schema (v2): `4f4c4a4b...`
- Evidence: `artifacts/experiments/v2-latent-diagnostics/diagnostics.json`

## Measured summary (500 validation users)

| Metric | Value |
| --- | ---: |
| Latent recall@500 (all users) | 0.838 |
| Latent recall@100 (all users) | 0.440 |
| Latent recall@50 (all users) | 0.264 |
| **Latent recall@10 (all users)** | **0.084** |
| Latent recall@10 (latent-present users) | 0.100 |
| Latent-present users | 419 / 500 |
| Target rank in latent list (p25 / p50 / p75) | 37 / 93 / 191 |
| Three-route union recall | 0.928 |
| Latent-only coverage (target in latent, absent from ItemCF ∪ Dense) | 0.050 |
| Overlap ItemCF↔latent (mean Jaccard) | 0.470 |
| Overlap Dense↔latent (mean Jaccard) | 0.150 |
| Latent fit wall time | 2.34 s |

## Candidate-stage gate

| Gate | Requirement | Measured | Pass |
| --- | --- | ---: | --- |
| G1 | latent recall@500 ≥ 0.55 (all 500 users) | 0.838 | yes |
| G2 | latent-present median rank ≤ 120 and p75 ≤ 300 (latent list) | 93 / 191 | yes |
| G3 | three-route union recall ≥ 0.90 and latent-only coverage ≥ 10/500 | 0.928 / 0.050 | yes |
| G4 | latent recall@10 ≥ 0.08 (all 500 users) | 0.084 | yes |

## Interpretation

The latent route adds a third collaborative signal that ItemCF and Dense do
not provide: 50 users (10%) have their target recovered only by latent, the
three-route union rises from 0.878 to 0.928, and the target's median rank in
the latent list (93) is substantially better than the ItemCF union-order
baseline (~172). Latent recall@10 on all 500 users (0.084) is above the
ItemCF all-user recall@10 (0.064), a ≈31% relative lift. The pre-registered
candidate gates pass without any threshold adjustment; the plan proceeds to
E3 (30-user smoke) and E4 (500-user LambdaMART) with the fixed ALS
hyperparameters (`rank=20, iterations=12, alpha=40, lambda_reg=0.1, seed=42`).

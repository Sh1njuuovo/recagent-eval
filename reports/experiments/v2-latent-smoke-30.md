# v2 latent 30-user smoke — benchmark and replay gate

## Status

Stability milestone for the latent-route LambdaMART pipeline. The run
publishes a complete `lambdamart-bundle/v2` with a bound latent artifact, keeps
constraint satisfaction at 100%, and reproduces byte-identical validation rows,
model, and latent factors across two independent runs. The 30-user ranking
numbers are **not** a gate; the frozen test stays locked.

- Date: 2026-08-23
- Config: `configs/v2_dense_latent_30.yaml` (run 1), `configs/v2_dense_latent_30b.yaml` (run 2)
- Wall time (run 1): 60.90 s (`real 60.90, user 59.89, sys 2.63`)
- Training users: 26; validation users: 30

## Measured metrics (run 1, stability only)

| Metric | Value |
| --- | ---: |
| LambdaMART mean NDCG@10 | 0.0 |
| ItemCF mean NDCG@10 | 0.0219 |
| Constraint satisfaction rate | 1.000 |
| Model checksum | `8710de2d...` |
| Candidate-policy fingerprint | `30b325ed...` |
| Feature schema fingerprint (v2) | `4f4c4a4b...` |

The 30-user NDCG difference is not statistically meaningful (wide bootstrap
interval at this sample size) and is not a gate; it only proves pipeline
stability.

## Bundle integrity (run 1)

- Bundle schema: `lambdamart-bundle/v2`
- Artifact schema: `lambdamart-artifact/v2`
- Evidence schema: `lambdamart-validation/v2`
- `bundle.latent_sha256` == artifact/evidence `latent_artifact_checksum` ==
  SHA-256 of `latent.npz` (`72c472db...`)

## Determinism / replay gate (G5)

Two runs into separate artifact directories (`-30` and `-30b`) produced:

| Check | Result |
| --- | --- |
| `per_user_rows` identical | yes |
| Model checksum identical (`8710de2d...`) | yes |
| Latent artifact SHA-256 identical (`72c472db...`) | yes |
| Constraint satisfaction 100% in both runs | yes |

The validation `evidence_fingerprint` values differ between the two runs by
design: the fingerprint payload includes the latent `artifact_path` (run-local
identity) and the latent manifest embeds `created_at`. Ranking-determinism
holds at the row, model, and latent-factor level, which is what the replay
path needs.

## Interpretation

The full v2 pipeline (dense + ItemCF + ALS fold-in + route-balanced hard
negatives + schema-v2 bundle) is stable end to end in the torch-loaded dense
process, with the latent artifact persisted atomically as a bundle member.
The wall-time benchmark (≈61 s for 30 users) is recorded for the E4 time
budget; latent fit itself was 2.34 s in the 500-user diagnosis. Proceed to E4.

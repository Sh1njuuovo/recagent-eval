# BPR-MF and LightGCN post-hoc robustness protocol

**Protocol fixed:** 2026-08-24, before running either additional seed.

## Evidence status

The seed-42 Confirmation-B run remains the sole formal main result. Seeds 7 and
2026 are post-hoc robustness checks of stochastic variation in the project-local
BPR-MF and LightGCN implementations. They do not complete an originally
preregistered three-seed evaluation and do not change Confirmation-B's identity.

## Locked inputs

- Cohort: `confirmation_b`, in the exact ordered-user sequence from cohort
  ledger fingerprint
  `7153c15e78540fead8059412ac0e37d2a1eb02f60e17152c9dca86ebd2e0186f`.
- Main seed: 42.
- Additional post-hoc seeds: 7 and 2026.
- Bootstrap: seed 42, 2,000 user-level paired resamples where a comparison is
  reported.
- Hyperparameters: exactly the parameters selected for seed 42.
- Metrics: Recall@10, NDCG@10, MRR@10, candidate recall, and constraint
  satisfaction.

The parameter grid will not be rerun. If the seed-42 selected parameters are
reconstructed from the deterministic selection path, the evidence will label
them `recovered_after_run` and bind the recovery command, source artifact
SHA-256, input fingerprint, output fingerprint, and recovery commit.

## Execution and reporting rules

Each method/seed runs in its own subprocess so `process_peak_rss_mib` describes
that process lifetime. Outputs use fresh seed-specific paths and refuse
overwrite. Every seed is retained and reported; no best-seed selection is
allowed. The final table reports each seed, arithmetic mean, sample standard
deviation, and worst seed for every metric.

The robustness outcome cannot alter current_v2b, its configuration, Success A
thresholds, or the seed-42 certification. LightGCN claims are limited to the
project-local fixed CPU protocol. If LightGCN remains below Popularity, the
report will discuss CPU budget, training epochs, and implementation differences
from canonical libraries.

No frozen case, LLM API, Qwen/vLLM process, merge, push, or PR is authorized by
this protocol.

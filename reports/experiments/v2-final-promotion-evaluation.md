# current_v2b final promotion evaluation

## Evidence identity

This report separates three evidence roles:

1. **Confirmation-B is the primary algorithm comparison.** It contains 1,000
   users that were not used during model selection and supports paired
   significance tests against ItemCF and ALS.
2. **The 50-case run is a one-time final promotion evaluation.** current_v2b,
   its package, and canonical identity were locked before the run. Identity
   `7739942c...` was consumed exactly once and completed successfully.
3. **The same 50-case suite had historical use.** The earlier
   `deepseek-constraint-aware` system experiment records the same case
   fingerprint, `bc2f622c...`. The suite is therefore not described as a
   project-history-clean or previously unused holdout.

## Primary algorithm result: Confirmation-B

| Method | Users | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| current_v2b | 1,000 | **0.118** | **0.0555** |
| ItemCF direct | 1,000 | 0.064 | 0.0323 |
| ALS direct | 1,000 | 0.069 | 0.0323 |

The current_v2b minus ItemCF NDCG@10 delta is +0.0231 with 95% paired-bootstrap
CI [0.0118, 0.0346]. Against ALS direct, the delta is +0.0232 with CI
[0.0111, 0.0346]. Both use 2,000 user-level resamples and seed 42. These are the
project's main algorithm and resume claims.

## One-time 50-case promotion result

| Metric | Value | Count interpretation |
| --- | ---: | ---: |
| Recall@10 / HitRate@10 | 0.0800 | 4/50 Top-10 hits |
| NDCG@10 | 0.03964 | — |
| ItemCF candidate recall | 0.7800 | 39/50 |
| Dense candidate recall | 0.6000 | 30/50 |
| Latent candidate recall | 0.8400 | 42/50 |
| Union candidate recall | 0.9400 | 47/50 |

The 50 cases provide a small-sample generalization check after identity lock.
They do not carry a baseline-win or significance claim: the promotion run did
not execute current_v2b-matched ItemCF and ALS baselines on this suite. The
lower point estimates are reported as observed and do not invalidate the
independent 1,000-user Confirmation-B inference.

The gap between 47/50 targets present in the candidate union and only 4/50
Top-10 hits continues to identify ranking depth as the main bottleneck. Candidate
coverage is high; the second stage often fails to move the target into the top
ten.

## Relationship to historical DeepSeek evidence

The historical DeepSeek experiment used the same formal case fingerprint. It
tested agent planning, retrieval-route compliance, tool execution, constraints,
and an earlier ranking stack. It is retained as a historical system experiment,
not treated as an untouched holdout or as a matched baseline for current_v2b.

## Governance after consumption

- The consumed identity can never be rerun or bypassed.
- No current_v2b or successor parameter may be selected from these 50 labels.
- Further algorithm work uses development/validation cohorts only.
- A v3 project must define, fingerprint, and preregister a new unused holdout
  before any v3 model development or result inspection.

Machine-readable evidence is in
`reports/experiments/v2-final-promotion-evaluation.json`; immutable metrics,
marker, command log, and environment are under
`artifacts/frozen/7739942cdb8d3c58-6cf9d5bfd8d4eb3f/`.


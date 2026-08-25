# Final Promotion Narrative Design

**Date:** 2026-08-25
**Status:** approved for documentation implementation

## Evidence layers

1. **Confirmation-B is the primary algorithm-comparison evidence.** It contains
   1,000 users that were not used during model selection. Seed-42 current_v2b
   achieved Recall@10 0.118 and NDCG@10 0.0555 versus ItemCF 0.064/0.0323 and
   ALS 0.069/0.0323. User-level paired-bootstrap lower bounds were positive.
2. **The 50-case result is a one-time final promotion evaluation.** It was run
   once after current_v2b, its package, and canonical identity were locked. It
   achieved Recall@10 0.08 and NDCG@10 0.03964; union candidate recall was 0.94
   (47/50) and Top-10 hit count was 4/50.
3. **The historical DeepSeek system experiment used the same case suite.** Its
   case fingerprint is the same `bc2f622c...` identity. The 50 cases therefore
   cannot be called a previously unseen or historically unused holdout.

## Claim boundaries

- Resume headline metrics continue to come from Confirmation-B.
- Describe the 50 cases as a locked, one-time final promotion evaluation and a
  small-sample generalization check.
- Do not claim a significant baseline win on the 50 cases: current_v2b-matched
  ItemCF and ALS baselines were not rerun there.
- Show the lower 50-case point estimates without using them to negate the
  independent Confirmation-B statistical result.
- Explain 47/50 union coverage versus 4/50 Top-10 hits as continued evidence of
  a candidate-depth-to-ranking-depth bottleneck.
- Do not tune against these 50 cases. Any v3 work uses development/validation
  cohorts and pre-registers a new, unused holdout before development begins.

## Files in scope

- Current project state: `README.md`, `docs/HANDOFF-2026-08-22.md`,
  `docs/project-methodology.md`, and `docs/demo-script.md`.
- Final report: `reports/experiments/v2-final-promotion-evaluation.md` and a
  machine-readable JSON summary derived from committed metrics/marker bytes.
- Job materials: STAR, interview pack/Q&A, PPT prompt, and application
  checklist under `reports/interview-pack/`.
- Historical specs, plans, and contemporaneous experiment reports retain their
  original stage-specific statements unless they are presented as current
  project status.

## Verification

- Copy every metric and fingerprint from committed JSON artifacts.
- Search current-state documents for stale `frozen remains unconsumed` claims.
- Search for forbidden descriptions such as pure/never-used frozen holdout and
  unsupported frozen baseline/significance claims.
- Validate the new JSON report and run Ruff, lock, diff, and shell syntax gates.


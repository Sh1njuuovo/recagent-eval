# RecAgent-Eval v2 — Phase 0 read-only audit (2026-08-23)

## 1. Repository and environment (verified, not assumed)

- Worktree: `/Users/shinjuu/intern/recagent-eval/.worktrees/agent-search-v2`
- Branch: `feat/dense-recall-v2`, `git status` clean, **24 commits ahead** of
  `origin/feat/dense-recall-v2` (HEAD `ac8626c`).
- Other worktrees untouched: `main` at `34c3a12`, `feat/v2-offline-ranking` at
  `5bf48bf`.
- Python 3.13.11 (worktree `.venv` only); torch 2.13.0, lightgbm 4.7.0,
  scikit-learn 1.9.0, numpy 2.5.1.
- Test suite: **303 tests**, line coverage **90.2%** (4099 statements, 400
  missed). Ruff / uv lock / git diff / bash -n all green at the last gate.

## 2. JSON re-verification of the prompt's numbers

All numbers below were re-read from machine artifacts in this worktree.

| Claim | Artifact | Measured |
| --- | --- | --- |
| latent recall@500 | `v2-latent-diagnostics/diagnostics.json` | 0.838 |
| latent target median rank | same | p50 93 (p25 37, p75 191) |
| three-route union recall | same | 0.928 |
| latent-only coverage | same | 0.050 |
| latent recall@10 (all 500) | same | 0.084 |
| E6 NDCG@10 / ItemCF | `v2-latent-bfeat-500/validation.json` | 0.044603 / 0.033388 |
| E6 recall@10 (learned / ItemCF) | same | 0.102 / 0.064 |
| E6 CI (2,000 paired bootstrap) | same | [−0.002396, 0.027993] |
| E6 constraints | same | 1.0 |
| E4 (hard neg) NDCG / CI | `v2-latent-hardneg-500/validation.json` | 0.0 / [−0.0461, −0.0213] |
| E5 (all neg) NDCG / CI | `v2-latent-allneg-500/validation.json` | 0.025837 / [−0.0214, 0.0073] |

The prompt's numbers are accurate. Current best learned variant is
`v2b + all negatives`: mean NDCG@10 above ItemCF but the pre-registered
paired-bootstrap 95% CI lower bound (−0.0024) still crosses zero.

## 3. Users previously used for diagnosis / tuning / model selection

Every prior v2 experiment (recall sweep, `diagnose-ranker`,
`diagnose-latent`, percentile, raw/recall-1500, E3 smoke, E4/E5/E6) operated
on the **first 500 eligible users** (sorted user IDs 1–500) for validation
targets and the same users' ranker targets for training/CV. They are the
**historical selection cohort** and must never appear in confirmation cohorts.

- Eligible users (3 disjoint targets): **6035**
- Historical selection users: **500** (IDs 1–500)
- Remaining unused eligible users: **5535** (IDs 501–6040)

## 4. Frozen test status

- No `*.consumed.json` marker exists anywhere under `artifacts/`; the
  `artifacts/frozen-consumption/` directory does not exist.
- Conclusion: **the frozen test has never been consumed**; it stays locked.

## 5. Per-user target semantics (current split)

`leakage_safe_ranking_split` assigns each eligible user three disjoint targets
from the three latest distinct positive ratings:

| Item | Definition |
| --- | --- |
| legal history | rows before the ranker target, excluding all three target movie IDs |
| train target | ranker target (2nd latest distinct positive, used for LambdaMART queries) |
| validation target | 2nd latest of the three distinct positives (evaluation labels in all prior runs) |
| frozen/test target | latest distinct positive; used only by locked frozen evaluation |
| `legal_retrieval_train` | rows before the validation target excluding validation+test movie IDs (includes the ranker target); used to fit retrieval |

Confirmation cohorts in this phase use **validation targets only**; frozen/test
targets are never read.

## 6. Frozen cases and overlap

- `cases/fixed_cases.json`: 50 cases, fingerprint `bc2f622c...`, each with a
  user ID and `relevant_movie_ids` (the frozen evaluation labels).
- All 50 case users are eligible; 6 fall inside the historical-500 cohort and
  44 fall in the remaining pool. Frozen case users take precedence: they are
  excluded from development/confirmation draws so all cohorts are mutually
  exclusive.

## 7. Replay of the current best artifact

The E6 v2b bundle (`v2-latent-bfeat-500`) was replayed under the current code:

- bundle v2 load with latent member: OK
- artifact parse + dataset/config/candidate-policy/case fingerprints + latent
  checksum: OK
- `build_validation_rows` with the persisted latent retriever reproduces the
  stored 500 per-user rows byte-for-byte: **PASS**

## 8. Cohort capacity

Available pool after excluding historical-500 and the 44 remaining frozen case
users: **5491 users**. Pre-registered cohort sizes (fixed seed 42, seeded
shuffle, mutually exclusive):

| Cohort | Size | Purpose |
| --- | ---: | --- |
| development | 600 | baseline hyperparameter selection (user-grouped CV on validation targets) |
| confirmation-A | 1000 | first untouched comparison of all methods |
| confirmation-B | 1000 | final untouched confirmation if the algorithm changes after A |
| reserve (untouched) | 2891 | future use; never read in this phase |
| frozen-test | 50 cases | locked; never read |

## 9. Findings

1. The current learned method is a point estimate only: the formal gate
   (CI lower bound > 0) is not met and no strong baseline comparison exists.
2. No BPR-MF / LightGCN / SASRec implementation or evidence exists yet.
3. Percentile calibration and route-balanced hard negatives are falsified and
   must not be re-packaged.
4. `torch` 2.13.0 is already available in the worktree venv (transitive via
   sentence-transformers) and present in `uv.lock`; it will be declared
   explicitly for the BPR-MF / LightGCN baselines.
5. Candidate signal is strong (union recall 0.928, latent median rank 93) and
   the remaining gap is Top-10 statistical confidence; this phase tests
   whether the gap is competitive vs strong baselines or is a real limitation.

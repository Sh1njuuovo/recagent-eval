# Constraint-aware DeepSeek evaluation

## Run status

- Provider: `deepseek-chat`, temperature 0
- Formal cases: 40 single-turn + 10 multi-turn
- Stability subset: 16 single-turn + 4 multi-turn
- Formal case fingerprint:
  `bc2f622cd9311bca8509a46f0ee516355bc64db7d91f809273a35d97ce304d88`
- Stability fingerprint:
  `73f3115a760641dd7a3a983fef3db097c8f4f891cca70eb5cc2cb1383af5d4bf`
- Frozen candidate depth: 500 for every formal variant
- Frozen semantic history cap: 50
- Validation-selected hybrid weights: ItemCF 0.7, TF-IDF 0.3,
  explicit affinity 0.0
- Total API usage: 238 calls and 167,621 tokens
- Machine-readable aggregate:
  `reports/experiments/deepseek-constraint-aware.json`

Per-episode run directories are intentionally ignored by Git and are regenerated
with `scripts/run_deepseek_matrix.sh`. The fixed cases, configuration,
validation ablation, tuned weights, and Markdown/JSON aggregate reports are
versioned.

## Evaluation-integrity changes

The archived first formal run exposed three confounded failures:

1. four multi-turn targets violated the generated negative-genre constraint;
2. a nominal full-hybrid plan could omit semantic retrieval;
3. a plan-level `top_k` could override the validation-frozen candidate budget.

The revised evaluator now:

- chooses a negative genre absent from the held-out target;
- rejects an inconsistent case before any paid provider call;
- makes required retrieval routes part of profile-aware plan validation;
- stores ordered candidate IDs for ItemCF, semantic retrieval, and reranking;
- reports target eligibility and per-route candidate recall;
- forces all formal variants to use the configured Top-500 budget;
- phrases genre history as a soft preference rather than an ambiguous hard
  “matching A, B” constraint.

The archived and revised case fingerprints differ, so their results are reported
separately rather than merged.

## Validation-only retrieval selection

The following study used 500 validation users. No test target selected candidate
depth, profile cap, or ranker weights.

| Top-K | History cap | ItemCF candidate recall | Semantic candidate recall | Union recall | NDCG@10 | ms/user |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 10 | 0.336 | 0.116 | 0.408 | 0.0318 | 27.92 |
| 100 | 20 | 0.336 | 0.130 | 0.424 | 0.0324 | 26.54 |
| 100 | 50 | 0.336 | 0.140 | 0.424 | 0.0341 | 26.16 |
| 200 | 10 | 0.468 | 0.186 | 0.556 | 0.0331 | 27.25 |
| 200 | 20 | 0.468 | 0.206 | 0.574 | 0.0339 | 26.90 |
| 200 | 50 | 0.468 | 0.230 | 0.584 | 0.0351 | 27.01 |
| 500 | 10 | 0.696 | 0.362 | 0.792 | 0.0355 | 28.26 |
| 500 | 20 | 0.696 | 0.386 | 0.806 | 0.0327 | 27.98 |
| **500** | **50** | **0.696** | **0.408** | **0.818** | **0.0366** | **28.54** |

Top-500 with a 50-item profile won on validation NDCG@10 and union candidate
recall. Compared with Top-100/cap-20, union coverage increased by 39.4
percentage points while measured local compute increased by about 2 ms/user.

## Formal results

| Variant | Recall@10 | NDCG@10 | HitRate@10 | ItemCF cand. | Semantic cand. | Union cand. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstructured, no memory | **0.0600** | **0.0486** | **0.0600** | 0.5800 | 0.0000 | 0.5800 |
| Structured + memory, ItemCF | **0.0600** | 0.0360 | **0.0600** | 0.7800 | 0.0000 | 0.7800 |
| Full hybrid | 0.0400 | 0.0149 | 0.0400 | 0.7800 | 0.4400 | **0.8800** |
| Full hybrid, 20-case stability | 0.0000 | 0.0000 | 0.0000 | 0.6500 | 0.3500 | 0.7500 |

The full system improves candidate union recall by 10 percentage points over the
same-depth structured ItemCF variant, but it does not improve top-10 quality.
Recall falls from 6% to 4%, and NDCG@10 falls from 0.0360 to 0.0149. This is a
negative ranking result: the lightweight TF-IDF route adds relevant candidates,
but the fixed linear score does not place them high enough.

The no-memory baseline drops the initial history before retrieval, so its
ItemCF route uses the deterministic popularity fallback. Its stronger NDCG is
therefore not evidence that forgetting user history is generally preferable;
it shows that this small fixed test matrix rewards popular items.

## Agent reliability and constraints

| Variant | Plan valid | Fallback | Pipeline compliant | Tool success | Label eligible | Final target eligible | Constraints | Excluded violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstructured, no memory | N/A | 0% | 100% | 100% | 100% | 100% | 100% | 0% |
| Structured + memory | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 0% |
| Full hybrid | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 0% |
| Full stability subset | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 0% |

Preference retention is 90% for structured memory and 100% for both full runs.
The unstructured baseline intentionally has no structured-plan denominator; its
raw `plan_valid_rate=0` means “not applicable.”

All acceptance thresholds for plan legality, pipeline compliance, tool
execution, hard constraints, and excluded-item safety are met. The aspirational
ranking criterion is not met, and no test cases or outputs were removed to hide
that result.

## Latency and cost

| Variant | p50 episode | p95 episode | Calls | Tokens | Episode failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unstructured, no memory | 11.11 s | 13.91 s | 70 | 41,781 | 0% |
| Structured + memory | 3.50 s | 7.11 s | 70 | 56,403 | 0% |
| Full hybrid | 2.45 s | 7.70 s | 70 | 50,430 | 0% |
| Full stability subset | 2.38 s | 6.55 s | 28 | 19,007 | 0% |

Latency is wall-clock episode latency and includes all turns in a multi-turn
case. API conditions varied during the run, so latency should not be interpreted
as a controlled model-speed benchmark.

## Failure-motivated conclusion

The strongest evidence from this iteration is a decomposition:

- evaluation-label eligibility: fixed and verified at 100%;
- Agent planning and execution: 100% legal and compliant, with zero fallback;
- hard-constraint safety: 100%, with zero excluded-item violations;
- candidate retrieval: improved materially to 88% union recall;
- final ranking: still weak and worse than the ItemCF-only variant.

The next justified experiment is a validation-only calibrated or learned
second-stage ranker using route ranks, raw scores, popularity, and genre
features. It should reuse this exact frozen test matrix. Another full LLM run is
not justified until offline ranking improves.

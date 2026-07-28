# DeepSeek formal evaluation

## Run status

- Provider: `deepseek-chat`, temperature 0
- Formal cases: 40 single-turn + 10 multi-turn
- Stability repeat: stratified 16 single-turn + 4 multi-turn
- Total API calls: 255, including repair requests
- Total tokens: 145,919
- Original formal-case fingerprint:
  `5e242d292164fee7755483b77cb66008b9e02cc06bb5e10627e363787d8cb3cf`
- Raw local artifacts: `artifacts/runs/*-deepseek/`

## Results

| Variant | Recall@10 | NDCG@10 | Plan valid | Fallback | Tool success | Constraints | p50 episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstructured/no memory | 0.0600 | **0.0486** | N/A | 0% | 100% | 100% | 10.13 s |
| Structured + memory | 0.0400 | 0.0326 | 84% | 16% | 100% | 100% | 1.33 s |
| Full hybrid | 0.0400 | 0.0226 | 86% | 14% | 100% | 100% | 1.66 s |
| Full, 20-case repeat | 0.0000 | 0.0000 | 90% | 10% | 100% | 100% | 1.89 s |

The baseline intentionally uses free text and therefore has no structured-plan
validity denominator. Its `plan_valid_rate=0` in raw JSON means “not
applicable”, not 50 malformed plans.

## Findings

1. Structured execution is safe after validation: all four runs achieved 100%
   tool execution and hard-constraint satisfaction with zero excluded-item
   violations.
2. The 95% plan-validity target was not met. All invalid structured episodes
   were multi-turn: 8/10 for structured-memory, 7/10 for full, and 2/4 in the
   repeat. All 40 formal single-turn episodes were valid.
3. Because an episode aggregates all turns, an invalid preference-only turn
   marks the whole episode invalid even when the final recommendation turn is
   valid. The current artifacts retain the error/fallback but not the raw
   intermediate LLM response, so exact schema categories cannot be recovered
   post hoc.
4. The unstructured baseline was expensive: it consumed 49,850 completion
   tokens versus 9,283 for the full structured run. Constraining JSON output
   cut completion-token volume substantially.
5. The full configuration did not beat ItemCF on ranking quality. This is a
   negative result, not an improvement claim.

## Metric caveat

`preference_retention_rate=0` is not used as a project claim. The generated
labels encode “Please avoid X” as soft `disliked_genres`, while DeepSeek often
maps it to the more appropriate hard `excluded_genres`. Several cases also
move known watched IDs into `excluded_movie_ids`. The next evaluator revision
must compare semantic constraint satisfaction rather than exact field choice.

## Targeted multi-turn fix

Root-cause reproduction showed that DeepSeek correctly extracted preferences,
but the final “exclude watched movies” turn sometimes emitted
`itemcf_retrieve → rerank → explain` without the mandatory `hard_filter`. The
repair prompt did not repeat the ordering constraint and often reproduced the
same invalid plan.

The fix:

- repeats `MUST include hard_filter before retrieval` in both initial and repair
  prompts;
- retains strict validation instead of relaxing the safety invariant;
- stores sanitized validation/fallback diagnostics for every turn;
- labels “avoid” as `excluded_genres` and accepts a hard exclusion as satisfying
  a soft negative preference.

The label-only case revision changed the current 50-case fingerprint to
`351b1d23b05cd993287f0598ce35a3fdb8f8b03a99987c3d351ac9d145e9c836`.
User turns, user histories, and held-out targets were unchanged.

Only the 10 multi-turn cases were re-run:

| Run | Plan valid | Fallback | Preference retention | Tool success | Constraints |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full, before fix | 30% | 70% | unreliable | 100% | 100% |
| Full, after fix | **100%** | **0%** | **90%** | 100% | 100% |

The follow-up used 30 calls and 18,759 tokens. Ranking remained at
Recall/NDCG@10 = 0 on this multi-turn-only subset, isolating the remaining
problem to candidate retrieval/ranking rather than Agent plan execution.

## Next engineering experiment

- Inspect the one retention miss (`multi-007`) without weakening constraints.
- Add source-score calibration or a small validation-only learning-to-rank
  model.
- Re-run ranking on the same 10 cases; do not spend on another full LLM matrix
  until offline ranking improves.

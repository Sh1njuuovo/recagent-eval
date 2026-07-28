# DeepSeek formal evaluation

## Run status

- Provider: `deepseek-chat`, temperature 0
- Formal cases: 40 single-turn + 10 multi-turn
- Stability repeat: stratified 16 single-turn + 4 multi-turn
- Total API calls: 255, including repair requests
- Total tokens: 145,919
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

## Next engineering experiment

- Treat preference-only turns as valid state-update plans instead of requiring a
  full retrieval/rerank chain on every turn.
- Persist sanitized per-turn plan-validation diagnostics.
- Make retention labels accept equivalent soft/hard negative preferences.
- Re-run only the 10 multi-turn cases before spending on another full matrix.

# Offline rule-based experiment

## Status

This run validates data splitting, tools, retrieval, ranking, constraints,
metrics, and artifact generation. It does **not** measure LLM extraction
quality. The formal DeepSeek run is now reported separately in
`deepseek-formal.md`; remote Qwen remains pending host access.

- Date: 2026-07-28
- Platform: macOS 15.7.4 arm64
- Python: 3.13.11 (project supports 3.11+)
- Data: MovieLens 1M, 3,883 movies, 988,139 training ratings after holdout
- Cases: 40 single-turn + 10 multi-turn
- Provider calls: 70 per variant after aggregating all multi-turn requests
- Seed: 42
- Stable case fingerprint:
  `351b1d23b05cd993287f0598ce35a3fdb8f8b03a99987c3d351ac9d145e9c836`

## Result

| Variant | Recall@10 | NDCG@10 | HitRate@10 | Plan valid | Constraints | Excluded violations | Tool success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstructured/no memory | 0.0600 | **0.0486** | 0.0600 | 0% | 100% | 0% | 100% |
| Structured + memory | 0.0600 | 0.0360 | 0.0600 | 100% | 100% | 0% | 100% |
| Full hybrid | **0.0800** | 0.0418 | **0.0800** | 100% | 100% | 0% | 100% |

## Interpretation

- The full configuration retrieves the held-out item for one additional case,
  raising hit/recall from 6% to 8%.
- It does not meet the target `NDCG@10 >= ItemCF baseline`: the relevant item is
  often placed lower in the top 10.
- Validation selected `(0.7, 0.3, 0.0)`. The handcrafted preference score added
  no validation gain, so it was not assigned an arbitrary positive weight.
- Multi-turn preference retention is 0 in this offline run because the
  rule-based provider intentionally returns an empty patch. This metric is only
  meaningful in the DeepSeek run.
- Next ranking experiment: calibrate per-source scores or train a small
  validation-only linear/listwise reranker while keeping the same test split.

## Debug evidence

1. Semantic retrieval originally admitted zero-similarity items. A failing
   integration test exposed the issue; candidates now require positive cosine
   similarity.
2. Case fingerprints originally differed across processes because Pydantic
   serialized sets in hash order. The manifest now hashes a recursively sorted
   canonical payload, and all three variants share the same fingerprint.

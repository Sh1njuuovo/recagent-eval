# Ten-minute demo script

1. **0:00–1:00 — Problem.** Show why conversational recommendation needs both
   LLM flexibility and deterministic ranking/measurement.
2. **1:00–2:00 — Upstream choice.** Explain RecAI’s tool workflow, its legacy
   dependency risk, and why RecBole remained a backup.
3. **2:00–4:00 — Architecture.** Walk from utterance to preference state,
   validated plan, hard filters, dual retrieval, reranking, and traces.
4. **4:00–5:30 — Failure behavior.** Feed an invalid provider response in the
   test and show one repair followed by deterministic fallback.
5. **5:30–7:00 — Reproduction.** Run `recagent-eval smoke`, inspect
   `metrics.json` and `run_manifest.json`, then show the stable case fingerprint.
6. **7:00–8:30 — Results.** Present the three-row table. Explicitly state that
   recall rose from 0.06 to 0.08 while NDCG stayed below baseline.
7. **8:30–9:30 — Debug story.** Explain the zero-similarity retrieval bug and
   cross-process set-order fingerprint bug found by tests.
8. **9:30–10:00 — Next step.** Run DeepSeek formal evaluation and Qwen/vLLM
   smoke on the 4090; then improve head ranking rather than hiding the negative
   NDCG result.

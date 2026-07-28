# Implementation plan

Build a Python 3.11 conversational movie recommendation agent using MovieLens
1M. The LLM produces a validated tool plan and updates structured preferences;
deterministic modules apply hard filters, ItemCF and semantic retrieval,
validation-tuned hybrid reranking, and reproducible evaluation.

Formal evaluation uses a DeepSeek OpenAI-compatible endpoint. A Qwen 7B/8B
model served through vLLM uses the same provider interface for a 10–20 case
remote RTX 4090 smoke test.

The formal comparison contains:

1. unstructured planning without memory;
2. structured planning with preference memory;
3. structured planning, memory, dual retrieval, and hybrid reranking.

The implementation must expose CLI batch evaluation, retain a lightweight
Gradio demo, avoid secrets in logs, and report negative results honestly.

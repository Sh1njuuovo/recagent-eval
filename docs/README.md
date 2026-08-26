# Documentation guide

Start with the documents that explain the shipped system:

| Document | Purpose |
| --- | --- |
| [Core code walkthrough](core-code-walkthrough.md) | Reading order for Agent, retrieval, v2b ranking, and evidence code |
| [Demo script](demo-script.md) | Ten-minute local presentation flow |
| [Project methodology](project-methodology.md) | Leakage prevention, experiment gates, and claim-to-evidence rules |
| [Remote 4090 runbook](remote-4090.md) | Optional Qwen3-8B/vLLM compatibility procedure; no completed GPU claim |
| [Final experiment report](../reports/experiments/v2-final-promotion-evaluation.md) | Confirmation-B and one-shot promotion claim boundaries |

The root [README](../README.md) summarizes the current public status. Historical
implementation plans and session handoffs are intentionally excluded from the
portfolio branch; the shipped methodology and evidence contracts are documented
in the files above and enforced by tests.

# Documentation guide

Start with the documents that explain the shipped system:

| Document | Purpose |
| --- | --- |
| [Core code walkthrough](core-code-walkthrough.md) | Reading order for Agent, retrieval, v2b ranking, and evidence code |
| [Demo script](demo-script.md) | Ten-minute local presentation flow |
| [Project methodology](project-methodology.md) | Leakage prevention, experiment gates, and claim-to-evidence rules |
| [Remote 4090 runbook](remote-4090.md) | Optional Qwen3-8B/vLLM compatibility procedure; no completed GPU claim |
| [Final experiment report](../reports/experiments/v2-final-promotion-evaluation.md) | Confirmation-B and one-shot promotion claim boundaries |

## Historical engineering record

`superpowers/specs/` and `superpowers/plans/` preserve the approved designs and
implementation sequences used during development. They are useful for auditing
decisions such as leakage-safe splitting, latent retrieval, evidence hygiene,
and one-shot promotion hardening, but are not required to run the project.

`HANDOFF-2026-08-22.md` is a detailed chronological handoff. The current public
status is summarized more directly in the root [README](../README.md).

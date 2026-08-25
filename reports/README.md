# Evidence index

This directory contains the project evidence trail. The landing page promotes
only results with a clear evaluation role; intermediate and negative experiments
remain available for auditability.

## Primary evidence

| Evidence | Role | Result |
| --- | --- | --- |
| [Confirmation-B strong baselines](experiments/v2-strong-baselines-confirmation-b.md) | Sole certification cohort; 1,000 users | current_v2b Recall@10 0.118, NDCG@10 0.0555 |
| [Compact Confirmation-B bundle](evidence/confirmation-b.compact.json) | Fail-closed aggregate and paired-bootstrap replay | Bound to cohort, method, config, and source artifacts |
| [Final promotion evaluation](experiments/v2-final-promotion-evaluation.md) | One-time 50-case generalization supplement | Recall@10 0.08, NDCG@10 0.03964, union recall 0.94 |
| [Cohort ledger](audit/2026-08-23-cohort-ledger.json) | Reproducible disjoint user allocation | development / A / B / reserve |

Confirmation-A is retained as development and replication evidence because it
was inspected before baseline implementation and metric corrections. It is not
used as the final certification claim.

## Method-development trail

These reports show how the final method emerged and preserve unsuccessful
hypotheses instead of filtering them out:

1. [Initial dense LambdaMART result](experiments/v2-dense-lambdamart-500user.md)
2. [Candidate-depth diagnosis](experiments/v2-ranker-diagnostics.md)
3. [Dense recall-1500 follow-up](experiments/v2-dense-lambdamart-recall1500.md)
4. [Percentile calibration falsification](experiments/v2-dense-lambdamart-recall1500-percentile.md)
5. [ALS latent-route diagnostics](experiments/v2-latent-diagnostics.md)
6. [v2b feature result](experiments/v2-dense-lambdamart-latent-bfeat.md)
7. [Confirmation-A replication](experiments/v2-strong-baselines-confirmation-a.md)
8. [Confirmation-B certification](experiments/v2-strong-baselines-confirmation-b.md)

The most consequential negative result was route-balanced hard-negative
sampling: it destroyed held-out ranking performance. All-negatives training
preserved the value of the latent route and enabled the final v2b result.

## Agent and provider evidence

- [DeepSeek constraint-aware system evaluation](experiments/deepseek-constraint-aware.md)
- [Rule-based offline evaluation](experiments/offline-rule-based.md)
- [Local Gradio screenshot](demo/v2-demo-lambdamart-rule-based.png)

These evaluate planning, tool execution, constraints, and the interactive path.
They are kept separate from the 1,000-user offline algorithm comparison.

## Integrity and replay

- [`audit/`](audit) — cohort ledger and repository audit material
- [`evidence/`](evidence) — compact provenance-bound replay bundles
- [`promotion/`](promotion) — canonical one-shot promotion manifest and receipt
- [`experiments/`](experiments) — human-readable reports and machine JSON

The historical `obsolete-*` promotion files are retained only as immutable
identity history. The canonical current result and the consumed marker are
identified in the final promotion report.

## Reproduction entry points

```bash
# Fast evidence replay
uv run recagent-eval replay-evidence \
  --bundle reports/evidence/confirmation-b.compact.json \
  --ledger reports/audit/2026-08-23-cohort-ledger.json \
  --summary reports/experiments/v2-strong-baselines-confirmation-b.json

# Local quality gate
uv run ruff check .
uv run pytest
```

Detailed training, artifact, and frozen-promotion history lives in the linked
experiment reports and [`docs/superpowers`](../docs/superpowers). Those files
are engineering records, not the recommended first reading path.

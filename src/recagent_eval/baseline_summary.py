from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from recagent_eval.baseline_eval import paired_bootstrap_deltas

SUMMARY_SCHEMA_VERSION = "baseline-summary/v1"
RESAMPLES = 2000
SEED = 42


def summarize_baselines(
    artifacts: Mapping[str, dict[str, object]],
    *,
    cohort: str,
    reference: str = "itemcf_direct",
) -> dict[str, object]:
    per_method_rows = {
        name: _per_user_rows(artifact) for name, artifact in artifacts.items()
    }
    user_sets = {
        name: {row["user_id"] for row in rows}
        for name, rows in per_method_rows.items()
    }
    common = set.intersection(*user_sets.values()) if user_sets else set()
    counts = {len(users) for users in user_sets.values()}
    if len(counts) != 1 or len(common) != counts.pop():
        raise ValueError("per-user rows are not aligned across methods")
    ordered = sorted(common)
    by_user = {
        name: {row["user_id"]: row for row in rows}
        for name, rows in per_method_rows.items()
    }
    aggregates: dict[str, dict[str, float]] = {}
    for name, rows in per_method_rows.items():
        aggregates[name] = {
            "recall_at_10": _mean([row["recall_at_10"] for row in rows]),
            "ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows]),
            "mrr_at_10": _mean([row["mrr_at_10"] for row in rows]),
            "candidate_recall": _mean([row["candidate_recall"] for row in rows]),
            "constraint_satisfaction_rate": _mean(
                [float(row["constraint_satisfied"]) for row in rows]
            ),
        }
    pairwise: dict[str, dict[str, float]] = {}
    for left in sorted(per_method_rows):
        for right in sorted(per_method_rows):
            if left >= right:
                continue
            left_values = [by_user[left][user]["ndcg_at_10"] for user in ordered]
            right_values = [by_user[right][user]["ndcg_at_10"] for user in ordered]
            pairwise[f"{right}_vs_{left}"] = paired_bootstrap_deltas(
                left_values, right_values, seed=SEED, resamples=RESAMPLES
            )
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "cohort": cohort,
        "reference": reference,
        "user_count": len(ordered),
        "ordered_user_ids": ordered,
        "aggregates": aggregates,
        "pairwise_ndcg_bootstrap": pairwise,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _per_user_rows(artifact: Mapping[str, object]) -> list[dict[str, object]]:
    rows = artifact.get("per_user_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline artifact has no per_user_rows")
    return [dict(row) for row in rows]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summary_to_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Strong-baseline comparison — cohort `{summary['cohort']}`",
        "",
        f"- Users: {summary['user_count']}",
        "- Pairwise: 2,000 paired bootstrap, seed 42, NDCG@10 deltas",
        "",
        "| Method | Recall@10 | NDCG@10 | MRR@10 | Candidate recall | Constraints |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregates = summary["aggregates"]
    for name in sorted(aggregates):
        row = aggregates[name]
        lines.append(
            f"| {name} | {row['recall_at_10']:.4f} | {row['ndcg_at_10']:.4f} | "
            f"{row['mrr_at_10']:.4f} | {row['candidate_recall']:.4f} | "
            f"{row['constraint_satisfaction_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise NDCG@10 (delta, 95% CI)",
            "",
            "| Pair | Mean | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for pair in sorted(summary["pairwise_ndcg_bootstrap"]):
        row = summary["pairwise_ndcg_bootstrap"][pair]
        lines.append(
            f"| {pair} | {row['mean_delta']:.4f} | "
            f"[{row['lower']:.4f}, {row['upper']:.4f}] |"
        )
    return "\n".join(lines) + "\n"

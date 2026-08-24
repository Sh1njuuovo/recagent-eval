from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping

from recagent_eval.evidence import artifact_fingerprint, canonical_digest, provenance_value

POSTHOC_SEEDS = (42, 7, 2026)
ROBUSTNESS_METRICS = (
    "recall_at_10",
    "ndcg_at_10",
    "mrr_at_10",
    "candidate_recall",
    "constraint_satisfaction_rate",
)


def build_posthoc_robustness_input(
    *,
    source_artifacts: Mapping[str, Mapping[int, bytes]],
    cohort: str,
) -> dict[str, object]:
    """Normalize immutable v1 and new v2 artifacts into one explicit derived schema."""
    methods: dict[str, object] = {}
    for method, by_seed in source_artifacts.items():
        rows: list[dict[str, object]] = []
        for expected_seed, source_bytes in by_seed.items():
            try:
                artifact = json.loads(source_bytes)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source artifact for {method}: {exc}") from exc
            if artifact.get("method") != method:
                raise ValueError(f"source method does not match slot {method}")
            if artifact.get("cohort") != cohort:
                raise ValueError(f"source cohort does not match {cohort}")
            schema = artifact.get("schema_version")
            if schema not in {"baseline-evaluation/v1", "baseline-evaluation/v2"}:
                raise ValueError(f"unknown source schema for {method}: {schema}")
            if artifact.get("fingerprint") != artifact_fingerprint(artifact):
                raise ValueError(f"source fingerprint drift for {method} seed {expected_seed}")
            if schema == "baseline-evaluation/v2":
                seed_record = artifact.get("seed")
                if (
                    not isinstance(seed_record, Mapping)
                    or seed_record.get("value") != expected_seed
                ):
                    raise ValueError(f"source seed does not match slot {expected_seed}")
            elif expected_seed != 42:
                raise ValueError("legacy v1 source is allowed only for formal seed 42")
            aggregates = artifact.get("aggregates")
            if not isinstance(aggregates, Mapping):
                raise ValueError(f"source aggregates missing for {method} seed {expected_seed}")
            metrics: dict[str, float] = {}
            for metric in ROBUSTNESS_METRICS:
                value = aggregates.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"source has invalid {metric} for {method} seed {expected_seed}"
                    )
                metrics[metric] = float(value)
            rows.append(
                {
                    "seed": expected_seed,
                    "source_schema": schema,
                    "source_artifact_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "source_artifact_fingerprint": artifact["fingerprint"],
                    "metrics": {"source": "derived", "value": metrics},
                }
            )
        rows.sort(key=lambda row: POSTHOC_SEEDS.index(int(row["seed"])))
        if tuple(int(row["seed"]) for row in rows) != POSTHOC_SEEDS:
            raise ValueError(f"{method} robustness requires exact seeds {POSTHOC_SEEDS}")
        methods[method] = rows
    payload: dict[str, object] = {
        "schema_version": "posthoc-robustness-input/v1",
        "cohort": cohort,
        "methods": methods,
    }
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def build_parameter_recovery_manifest(
    *,
    selections: Mapping[str, Mapping[str, object]],
    source_artifacts: Mapping[str, Mapping[str, bytes]],
    command: str,
    commit_sha: str,
) -> dict[str, object]:
    cohorts: dict[str, object] = {}
    for cohort, sources in source_artifacts.items():
        methods: dict[str, object] = {}
        if set(sources) != set(selections):
            raise ValueError(f"source methods do not match recovered selections for {cohort}")
        for method, source_bytes in sources.items():
            try:
                artifact = json.loads(source_bytes)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source artifact for {method}: {exc}") from exc
            selection = selections[method]
            selection_fingerprint = selection.get("fingerprint")
            if artifact.get("config_fingerprint") != selection_fingerprint:
                raise ValueError(f"selection fingerprint drift for {method} on {cohort}")
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            input_fingerprint = str(artifact.get("fingerprint"))
            records: dict[str, object] = {
                "selection_fingerprint": selection_fingerprint,
            }
            for source_field, output_field in (
                ("selected_params", "selected_params"),
                ("grid", "parameter_grid"),
                ("seed", "seed"),
            ):
                value = selection[source_field]
                records[output_field] = provenance_value(
                    value,
                    source="recovered",
                    recovery={
                        "status": "recovered_after_run",
                        "command": command,
                        "source_artifact_sha256": source_sha,
                        "input_fingerprint": input_fingerprint,
                        "output_fingerprint": canonical_digest(value),
                        "commit_sha": commit_sha,
                    },
                )
            methods[method] = records
        cohorts[cohort] = methods
    payload: dict[str, object] = {
        "schema_version": "baseline-parameter-recovery/v1",
        "status": "recovered_after_run",
        "command": command,
        "commit_sha": commit_sha,
        "cohorts": cohorts,
    }
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def summarize_posthoc_robustness(
    seed_metrics: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> dict[str, object]:
    methods: dict[str, object] = {}
    for method, by_seed in seed_metrics.items():
        if set(by_seed) != set(POSTHOC_SEEDS):
            raise ValueError(f"{method} robustness requires exact seeds {POSTHOC_SEEDS}")
        seed_rows: list[dict[str, object]] = []
        for seed in POSTHOC_SEEDS:
            metrics = by_seed[seed]
            row: dict[str, object] = {"seed": seed}
            for metric in ROBUSTNESS_METRICS:
                value = metrics.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{method} seed {seed} has invalid {metric}")
                row[metric] = float(value)
            seed_rows.append(row)
        summary: dict[str, object] = {}
        for metric in ROBUSTNESS_METRICS:
            values = [float(row[metric]) for row in seed_rows]
            worst_index = min(range(len(values)), key=values.__getitem__)
            summary[metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "worst_seed": POSTHOC_SEEDS[worst_index],
                "worst_value": values[worst_index],
            }
        methods[method] = {"seeds": seed_rows, "summary": summary}
    payload: dict[str, object] = {
        "schema_version": "posthoc-robustness-summary/v1",
        "evidence_status": "post-hoc robustness",
        "formal_main_seed": 42,
        "posthoc_seeds": [7, 2026],
        "methods": methods,
    }
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def summarize_posthoc_robustness_input(value: Mapping[str, object]) -> dict[str, object]:
    if value.get("schema_version") != "posthoc-robustness-input/v1":
        raise ValueError("unknown post-hoc robustness input schema")
    recorded = value.get("fingerprint")
    if not isinstance(recorded, str) or recorded != canonical_digest(
        {key: item for key, item in value.items() if key != "fingerprint"}
    ):
        raise ValueError("post-hoc robustness input fingerprint drift")
    methods = value.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("post-hoc robustness input methods are missing")
    seed_metrics: dict[str, dict[int, Mapping[str, float]]] = {}
    for method, raw_rows in methods.items():
        if not isinstance(method, str) or not isinstance(raw_rows, list):
            raise ValueError("post-hoc robustness input method is malformed")
        by_seed: dict[int, Mapping[str, float]] = {}
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("post-hoc robustness input row is malformed")
            seed = raw_row.get("seed")
            metrics_record = raw_row.get("metrics")
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not isinstance(metrics_record, Mapping)
                or metrics_record.get("source") != "derived"
                or not isinstance(metrics_record.get("value"), Mapping)
                or seed in by_seed
            ):
                raise ValueError("post-hoc robustness input seed row is malformed")
            by_seed[seed] = metrics_record["value"]  # type: ignore[assignment]
        seed_metrics[method] = by_seed
    summary = summarize_posthoc_robustness(seed_metrics)
    summary["source_input_fingerprint"] = recorded
    summary["fingerprint"] = canonical_digest(
        {key: item for key, item in summary.items() if key != "fingerprint"}
    )
    return summary

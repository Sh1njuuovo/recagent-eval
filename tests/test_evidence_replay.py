from __future__ import annotations

import copy
import hashlib
import json

import pytest

from recagent_eval.baseline_eval import paired_bootstrap_deltas
from recagent_eval.evidence import artifact_fingerprint, canonical_digest
from recagent_eval.evidence_replay import (
    EvidenceReplayError,
    build_compact_bundle,
    replay_compact_bundle,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _rows(users: list[int], ndcgs: list[float]) -> list[dict[str, object]]:
    return [
        {
            "user_id": user,
            "recall_at_10": float(ndcg > 0),
            "ndcg_at_10": ndcg,
            "mrr_at_10": ndcg,
            "candidate_recall": 1.0,
            "constraint_satisfied": True,
            "latency_ms": 1.0,
        }
        for user, ndcg in zip(users, ndcgs, strict=True)
    ]


def _artifact(method: str, users: list[int], ndcgs: list[float]) -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": "baseline-evaluation/v1",
        "method": method,
        "cohort": "confirmation_b",
        "config_fingerprint": "c" * 64,
        "dataset_fingerprint": "d" * 64,
        "model_fingerprint": (method[0] * 64),
        "user_count": len(users),
        "per_user_rows": _rows(users, ndcgs),
    }
    artifact["fingerprint"] = artifact_fingerprint(artifact)
    return artifact


def _summary(users: list[int]) -> dict[str, object]:
    itemcf = [0.0, 1.0]
    current = [1.0, 1.0]
    payload: dict[str, object] = {
        "schema_version": "baseline-summary/v1",
        "cohort": "confirmation_b",
        "reference": "itemcf_direct",
        "user_count": len(users),
        "ordered_user_ids": users,
        "aggregates": {
            "itemcf_direct": {
                "recall_at_10": 0.5,
                "ndcg_at_10": 0.5,
                "mrr_at_10": 0.5,
                "candidate_recall": 1.0,
                "constraint_satisfaction_rate": 1.0,
            },
            "current_v2b": {
                "recall_at_10": 1.0,
                "ndcg_at_10": 1.0,
                "mrr_at_10": 1.0,
                "candidate_recall": 1.0,
                "constraint_satisfaction_rate": 1.0,
            },
        },
        "pairwise_ndcg_bootstrap": {
            "itemcf_direct_vs_current_v2b": paired_bootstrap_deltas(
                current, itemcf, seed=42, resamples=2000
            )
        },
    }
    payload["fingerprint"] = canonical_digest(payload)
    return payload


def _recovery(methods: list[str]) -> dict[str, object]:
    binding = {
        "status": "recovered_after_run",
        "command": ".venv/bin/recagent-eval recover-baseline-params",
        "source_artifact_sha256": "a" * 64,
        "input_fingerprint": "b" * 64,
        "output_fingerprint": "c" * 64,
        "commit_sha": "d" * 40,
    }
    return {
        method: {
            "selected_params": {
                "source": "recovered",
                "value": {},
                "recovery": dict(binding),
            },
            "parameter_grid": {
                "source": "recovered",
                "value": [],
                "recovery": dict(binding),
            },
            "seed": {
                "source": "recovered",
                "value": 42,
                "recovery": dict(binding),
            },
        }
        for method in methods
    }


def _inputs() -> tuple[dict[str, bytes], bytes, bytes, dict[str, object]]:
    users = [1, 2]
    sources = {
        "itemcf_direct": _json_bytes(_artifact("itemcf_direct", users, [0.0, 1.0])),
        "current_v2b": _json_bytes(_artifact("current_v2b", users, [1.0, 1.0])),
    }
    ledger = {
        "schema_version": "cohort-ledger/v1",
        "cohorts": {"confirmation_b": users},
        "fingerprint": "l" * 64,
    }
    summary = _summary(users)
    recovery = _recovery(list(sources))
    for method, source_bytes in sources.items():
        for field in ("selected_params", "parameter_grid", "seed"):
            recovery[method][field]["recovery"]["source_artifact_sha256"] = (
                hashlib.sha256(source_bytes).hexdigest()
            )
    return sources, _json_bytes(ledger), _json_bytes(summary), recovery


def test_build_and_replay_compact_bundle_verifies_full_chain() -> None:
    sources, ledger_bytes, summary_bytes, recovery = _inputs()
    bundle = build_compact_bundle(
        source_artifacts=sources,
        ledger_bytes=ledger_bytes,
        summary_bytes=summary_bytes,
        recovery=recovery,
        cohort="confirmation_b",
        commit_sha="e" * 40,
    )
    assert bundle["generator"]["schema_version"] == "evidence-bundle-generator/v1"
    assert bundle["bootstrap"] == {"seed": 42, "resamples": 2000}
    assert len(bundle["methods"]["itemcf_direct"]["source_artifact_sha256"]) == 64
    replayed = replay_compact_bundle(
        bundle,
        ledger_bytes=ledger_bytes,
        summary_bytes=summary_bytes,
    )
    assert replayed["fingerprint"] == bundle["summary_fingerprint"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda bundle: bundle["methods"]["itemcf_direct"].update(
                canonical_per_user_digest="0" * 64
            ),
            "per-user digest",
        ),
        (
            lambda bundle: bundle["methods"]["itemcf_direct"]["per_user_rows"].append(
                copy.deepcopy(bundle["methods"]["itemcf_direct"]["per_user_rows"][0])
            ),
            "duplicate",
        ),
        (
            lambda bundle: bundle["methods"]["itemcf_direct"]["per_user_rows"][0].update(
                ndcg_at_10=float("nan")
            ),
            "finite",
        ),
        (lambda bundle: bundle.update(schema_version="unknown"), "unknown bundle schema"),
    ],
)
def test_replay_rejects_tampered_bundle(mutation, message: str) -> None:
    sources, ledger_bytes, summary_bytes, recovery = _inputs()
    bundle = build_compact_bundle(
        source_artifacts=sources,
        ledger_bytes=ledger_bytes,
        summary_bytes=summary_bytes,
        recovery=recovery,
        cohort="confirmation_b",
        commit_sha="e" * 40,
    )
    mutation(bundle)
    with pytest.raises(EvidenceReplayError, match=message):
        replay_compact_bundle(bundle, ledger_bytes=ledger_bytes, summary_bytes=summary_bytes)


def test_replay_rejects_ledger_summary_and_bundle_fingerprint_drift() -> None:
    sources, ledger_bytes, summary_bytes, recovery = _inputs()
    bundle = build_compact_bundle(
        source_artifacts=sources,
        ledger_bytes=ledger_bytes,
        summary_bytes=summary_bytes,
        recovery=recovery,
        cohort="confirmation_b",
        commit_sha="e" * 40,
    )
    with pytest.raises(EvidenceReplayError, match="ledger SHA"):
        replay_compact_bundle(bundle, ledger_bytes=ledger_bytes + b" ", summary_bytes=summary_bytes)
    with pytest.raises(EvidenceReplayError, match="summary SHA"):
        replay_compact_bundle(bundle, ledger_bytes=ledger_bytes, summary_bytes=summary_bytes + b" ")
    bundle["fingerprint"] = "0" * 64
    with pytest.raises(EvidenceReplayError, match="bundle fingerprint"):
        replay_compact_bundle(bundle, ledger_bytes=ledger_bytes, summary_bytes=summary_bytes)

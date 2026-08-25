from __future__ import annotations

import hashlib
import json

from recagent_eval.cohorts import build_cohort_ledger, ledger_fingerprint


def test_ledger_cohorts_are_mutually_exclusive_and_cover_the_pool() -> None:
    eligible = list(range(1, 21))
    historical = set(range(1, 6))
    excluded = {9}
    ledger = build_cohort_ledger(
        eligible,
        historical=historical,
        excluded=excluded,
        sizes={"development": 4, "confirmation_a": 5, "confirmation_b": 5},
        seed=42,
    )
    cohorts = {
        name: set(ledger["cohorts"][name])
        for name in ("development", "confirmation_a", "confirmation_b")
    }
    assert cohorts["development"].isdisjoint(cohorts["confirmation_a"])
    assert cohorts["development"].isdisjoint(cohorts["confirmation_b"])
    assert cohorts["confirmation_a"].isdisjoint(cohorts["confirmation_b"])
    assert all(
        user not in historical and user not in excluded
        for cohort in cohorts.values()
        for user in cohort
    )
    assert ledger["seed"] == 42
    assert ledger["fingerprint"] == ledger_fingerprint(ledger["cohorts"])
    assert all(
        len(ledger["cohorts"][name]) == size
        for name, size in ledger["sizes"].items()
    )


def test_ledger_is_deterministic_and_hashes_the_lists() -> None:
    eligible = list(range(1, 101))
    first = build_cohort_ledger(
        eligible,
        historical={1},
        excluded=set(),
        sizes={"development": 10, "confirmation_a": 20, "confirmation_b": 20},
        seed=7,
    )
    second = build_cohort_ledger(
        eligible,
        historical={1},
        excluded=set(),
        sizes={"development": 10, "confirmation_a": 20, "confirmation_b": 20},
        seed=7,
    )
    assert first["cohorts"] == second["cohorts"]
    assert first["fingerprint"] == second["fingerprint"]
    canonical = json.dumps(
        first["cohorts"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert first["fingerprint"] == hashlib.sha256(canonical).hexdigest()

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence

COHORT_SCHEMA_VERSION = "cohort-ledger/v1"


def build_cohort_ledger(
    eligible: Sequence[int],
    *,
    historical: Iterable[int],
    excluded: Iterable[int],
    sizes: Mapping[str, int],
    seed: int,
) -> dict[str, object]:
    """Deterministically assign disjoint cohorts from the eligible pool."""
    blocked = set(historical) | set(excluded)
    pool = sorted(user for user in eligible if user not in blocked)
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    cohorts: dict[str, list[int]] = {}
    cursor = 0
    for name in ("development", "confirmation_a", "confirmation_b"):
        size = int(sizes[name])
        if cursor + size > len(shuffled):
            raise ValueError(f"pool too small for cohort {name}")
        cohorts[name] = sorted(shuffled[cursor : cursor + size])
        cursor += size
    cohorts["reserve"] = sorted(shuffled[cursor:])
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "seed": seed,
        "sizes": dict(sizes),
        "blocked_historical_count": len(set(historical) & set(eligible)),
        "blocked_excluded_count": len(set(excluded) & set(eligible)),
        "pool_size": len(pool),
        "cohorts": cohorts,
        "fingerprint": ledger_fingerprint(cohorts),
    }


def ledger_fingerprint(cohorts: Mapping[str, Sequence[int]]) -> str:
    canonical = json.dumps(cohorts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()

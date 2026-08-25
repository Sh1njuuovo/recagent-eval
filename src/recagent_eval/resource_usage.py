from __future__ import annotations

import math
import os
import platform
import resource

_SOURCE = "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"


def normalize_process_peak_rss(
    raw_value: int | float,
    *,
    system: str,
    process_id: int | None = None,
) -> dict[str, object]:
    """Normalize a single process lifetime peak RSS to MiB."""
    value = float(raw_value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("peak RSS raw value must be finite non-negative")
    if system == "Darwin":
        raw_unit = "bytes"
        normalized_mib = value / (1024.0 * 1024.0)
    elif system == "Linux":
        raw_unit = "KiB"
        normalized_mib = value / 1024.0
    else:
        raise ValueError(f"unsupported platform for peak RSS: {system}")
    return {
        "metric_name": "process_peak_rss_mib",
        "source": _SOURCE,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "normalized_mib": normalized_mib,
        "platform": system,
        "measurement_scope": "single_process_lifetime_peak",
        "process_id": os.getpid() if process_id is None else process_id,
    }


def read_process_peak_rss() -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return normalize_process_peak_rss(
        usage.ru_maxrss,
        system=platform.system(),
        process_id=os.getpid(),
    )

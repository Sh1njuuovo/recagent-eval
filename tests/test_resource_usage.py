from __future__ import annotations

import pytest

from recagent_eval.resource_usage import normalize_process_peak_rss


def test_peak_rss_darwin_bytes_to_mib() -> None:
    record = normalize_process_peak_rss(104_857_600, system="Darwin")
    assert record["metric_name"] == "process_peak_rss_mib"
    assert record["source"] == "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    assert record["raw_value"] == 104_857_600
    assert record["raw_unit"] == "bytes"
    assert record["normalized_mib"] == 100.0
    assert record["platform"] == "Darwin"
    assert record["measurement_scope"] == "single_process_lifetime_peak"
    assert isinstance(record["process_id"], int)


def test_peak_rss_linux_kib_to_mib() -> None:
    record = normalize_process_peak_rss(102_400, system="Linux")
    assert record["raw_unit"] == "KiB"
    assert record["normalized_mib"] == 100.0


@pytest.mark.parametrize("raw_value", [-1, float("nan"), float("inf")])
def test_peak_rss_rejects_invalid_raw_values(raw_value: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        normalize_process_peak_rss(raw_value, system="Linux")


def test_peak_rss_rejects_unsupported_platform() -> None:
    with pytest.raises(ValueError, match="unsupported platform"):
        normalize_process_peak_rss(1, system="Windows")

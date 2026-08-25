# Strong-baseline evidence corrections and addendum

**Issued:** 2026-08-24

This addendum changes evidence interpretation and resource-field validity. It
does not alter the recommendation methods, cohorts, metrics, thresholds, JSON
rows, or historical artifacts.

## Confirmation identity

Confirmation-A was read before fixes to BPR-MF/LightGCN batch-sum loss and
initialization behavior and the ALS/dev-CV metric denominator. It is therefore
development/debugging/replication evidence. Confirmation-B seed 42 is the sole
final certification result and independently satisfies Success A.

## Historical peak RSS field

The v1 baseline writers divided `resource.getrusage(RUSAGE_SELF).ru_maxrss` by
1024 on every platform. Darwin reports the raw value in bytes; Linux reports it
in KiB. Consequently the old `peak_memory_mb` values are labeled:

```text
invalid_due_to_platform_unit_bug
```

They must not be used for external comparison, Pareto claims, or conversion
unless the original platform and raw value are independently proven. The v1
files remain byte-identical.

New evidence names the metric `process_peak_rss_mib` and records metric source,
raw value, raw unit, normalized MiB, platform, process ID, and measurement
scope. Darwin divides bytes by 1,048,576; Linux divides KiB by 1,024. Each
formal method must run in an independent process because `ru_maxrss` is the
process-lifetime peak.

The A/B models are not rerun solely to repair this field. Post-hoc robustness
and any later separately authorized frozen run use the corrected measurement
path. Until such evidence exists, public cost tables omit peak RSS.

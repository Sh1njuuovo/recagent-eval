# Evidence Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RecAgent-Eval's baseline evidence independently replayable,
strictly provenance-bound, correctly measured, factually documented, and above
the real 90% coverage gate without consuming frozen data.

**Architecture:** Keep historical v1 JSON immutable, add focused resource and
evidence modules for strict v2 writing, normalize all validation through one
version dispatcher, and commit compact replay bundles generated from existing
per-user rows. Run post-hoc seeds in isolated subprocesses with locked recovered
parameters, then reconcile reports from replayed JSON.

**Tech Stack:** Python 3.13 in `.venv`, Typer, Pydantic-compatible validation,
NumPy, PyTorch, pytest/pytest-cov, Ruff, uv, canonical JSON/SHA-256.

---

## File map

- `src/recagent_eval/resource_usage.py`: platform-aware process peak RSS record.
- `src/recagent_eval/evidence.py`: schema dispatch, provenance types, strict
  validation, canonical digests, and compact bundles.
- `src/recagent_eval/evidence_replay.py`: aggregate/bootstrap replay and chain
  verification.
- `src/recagent_eval/baseline_summary.py`: validated summaries using evidence
  readers.
- `src/recagent_eval/cli.py`: evidence generation/replay and isolated baseline
  execution commands.
- `tests/test_resource_usage.py`, `tests/test_evidence.py`,
  `tests/test_evidence_replay.py`: focused RED/GREEN coverage.
- Existing baseline/CLI/summary tests: regression and refusal-path coverage.
- `reports/evidence/`: committed A/B compact bundles.
- `reports/experiments/`: correction, identity, and robustness addenda.
- README/HANDOFF/methodology/demo/interview-pack: factual reconciliation.

### Task 1: Coverage-first baseline and summary error paths

**Files:**
- Modify: `tests/test_baseline_eval.py`
- Modify: `tests/test_baseline_summary.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add tests for empty metrics, duplicate registration, invalid
  bootstrap shape/non-finite values, markdown serialization, bad cohort,
  missing/corrupt artifact, and output refusal.**

```python
with pytest.raises(ValueError, match="finite"):
    paired_bootstrap_deltas([float("nan")], [0.0], seed=42)

with pytest.raises(ValueError, match="aligned"):
    summarize_baselines({"itemcf_direct": itemcf, "current_v2b": duplicate})
```

- [ ] **Step 2: Run RED and confirm targeted missing/error branches execute.**

```bash
.venv/bin/pytest tests/test_baseline_eval.py tests/test_baseline_summary.py tests/test_cli.py -q
```

- [ ] **Step 3: Add only validation needed by the failing tests, preserving v1
  successful behavior.**

- [ ] **Step 4: Run GREEN and focused Ruff.**

```bash
.venv/bin/pytest tests/test_baseline_eval.py tests/test_baseline_summary.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/baseline_eval.py src/recagent_eval/baseline_summary.py tests/test_baseline_eval.py tests/test_baseline_summary.py tests/test_cli.py
```

- [ ] **Step 5: Commit.**

```bash
git add tests/test_baseline_eval.py tests/test_baseline_summary.py tests/test_cli.py src/recagent_eval/baseline_eval.py src/recagent_eval/baseline_summary.py
git commit -m "test: cover baseline evidence error paths"
```

### Task 2: Cross-platform subprocess peak RSS

**Files:**
- Create: `src/recagent_eval/resource_usage.py`
- Create: `tests/test_resource_usage.py`
- Modify: all six files under `src/recagent_eval/baselines/`
- Modify: `src/recagent_eval/cli.py`

- [ ] **Step 1: Write RED tests for Darwin bytes, Linux KiB, unsupported
  systems, and the serialized resource record.**

```python
def test_peak_rss_darwin_bytes_to_mib() -> None:
    record = normalize_process_peak_rss(104857600, system="Darwin")
    assert record.raw_unit == "bytes"
    assert record.process_peak_rss_mib == 100.0

def test_peak_rss_linux_kib_to_mib() -> None:
    record = normalize_process_peak_rss(102400, system="Linux")
    assert record.raw_unit == "KiB"
    assert record.process_peak_rss_mib == 100.0
```

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/pytest tests/test_resource_usage.py -q
```

- [ ] **Step 3: Implement immutable `ProcessPeakRss` plus
  `normalize_process_peak_rss` and `read_process_peak_rss`; replace direct
  `resource.getrusage` divisions.**

- [ ] **Step 4: Add an internal one-method child command and make formal
  baseline orchestration spawn `.venv/bin/recagent-eval` once per method,
  capturing a fresh JSON output and rejecting overwrite.**

- [ ] **Step 5: Run GREEN, baseline scorer tests, and Ruff.**

```bash
.venv/bin/pytest tests/test_resource_usage.py tests/test_baselines_*.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/resource_usage.py src/recagent_eval/baselines src/recagent_eval/cli.py tests/test_resource_usage.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/recagent_eval/resource_usage.py src/recagent_eval/baselines src/recagent_eval/cli.py tests/test_resource_usage.py tests/test_baselines_*.py tests/test_cli.py
git commit -m "fix: measure process peak RSS per baseline subprocess"
```

### Task 3: Strict versioned evidence and provenance

**Files:**
- Create: `src/recagent_eval/evidence.py`
- Create: `tests/test_evidence.py`
- Modify: `src/recagent_eval/baseline_eval.py`
- Modify: `src/recagent_eval/baseline_summary.py`

- [ ] **Step 1: Write RED tests for v1 compatibility and v2 strict fields,
  source classes, artifact fingerprints, method-slot mismatch, cohort drift,
  duplicate/missing users, empty fingerprints, non-finite values, unknown
  schema, and mixed v1/v2.**

```python
with pytest.raises(EvidenceValidationError, match="method"):
    validate_evidence_set({"itemcf_direct": artifact | {"method": "als_direct"}}, ledger)
with pytest.raises(EvidenceValidationError, match="mixed schema"):
    validate_evidence_set({"a": v1, "b": v2}, ledger)
```

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/pytest tests/test_evidence.py -q
```

- [ ] **Step 3: Implement canonical JSON/SHA helpers, known-schema dispatch,
  `observed|derived|recovered` provenance records, strict v2 creation, v1
  immutable reader, and set-level identity validation.**

- [ ] **Step 4: Route `summarize_baselines` through the validator and include
  ledger identity in new summary schema.**

- [ ] **Step 5: Run GREEN plus legacy summary tests.**

```bash
.venv/bin/pytest tests/test_evidence.py tests/test_baseline_eval.py tests/test_baseline_summary.py -q
.venv/bin/ruff check src/recagent_eval/evidence.py src/recagent_eval/baseline_eval.py src/recagent_eval/baseline_summary.py tests/test_evidence.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/recagent_eval/evidence.py src/recagent_eval/baseline_eval.py src/recagent_eval/baseline_summary.py tests/test_evidence.py tests/test_baseline_eval.py tests/test_baseline_summary.py
git commit -m "feat: add strict versioned baseline evidence"
```

### Task 4: Compact bundles and independent replay

**Files:**
- Create: `src/recagent_eval/evidence_replay.py`
- Create: `tests/test_evidence_replay.py`
- Modify: `src/recagent_eval/cli.py`
- Create: `reports/evidence/confirmation-a.compact.json`
- Create: `reports/evidence/confirmation-b.compact.json`

- [ ] **Step 1: Write RED tests covering canonical row digest, aggregate
  replay, all pairwise bootstraps, source/ledger/summary digest drift,
  duplicate/misaligned users, non-finite values, unknown generator schema, and
  overwrite refusal.**

```python
replayed = replay_bundle(bundle, ledger_bytes=ledger_bytes, summary_bytes=summary_bytes)
assert replayed.summary_fingerprint == bundle["summary_fingerprint"]
with pytest.raises(EvidenceReplayError, match="row digest"):
    replay_bundle(tampered_bundle, ledger_bytes=ledger_bytes, summary_bytes=summary_bytes)
```

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/pytest tests/test_evidence_replay.py -q
```

- [ ] **Step 3: Implement compact generation and chain replay; add
  `build-evidence-bundle` and `replay-evidence` commands with fresh-output
  refusal.**

- [ ] **Step 4: Run GREEN and CLI tests.**

```bash
.venv/bin/pytest tests/test_evidence_replay.py tests/test_cli.py -q
.venv/bin/ruff check src/recagent_eval/evidence_replay.py src/recagent_eval/cli.py tests/test_evidence_replay.py tests/test_cli.py
```

- [ ] **Step 5: Generate A/B bundles from ignored source artifacts, replay
  them, and compare the emitted aggregates/bootstrap/fingerprint with committed
  summaries.**

```bash
.venv/bin/recagent-eval build-evidence-bundle --cohort confirmation_a --ledger reports/audit/2026-08-23-cohort-ledger.json --artifact-dir artifacts/experiments/v2-baselines --summary reports/experiments/v2-strong-baselines-confirmation-a.json --output reports/evidence/confirmation-a.compact.json
.venv/bin/recagent-eval replay-evidence --bundle reports/evidence/confirmation-a.compact.json --ledger reports/audit/2026-08-23-cohort-ledger.json --summary reports/experiments/v2-strong-baselines-confirmation-a.json
.venv/bin/recagent-eval build-evidence-bundle --cohort confirmation_b --ledger reports/audit/2026-08-23-cohort-ledger.json --artifact-dir artifacts/experiments/v2-baselines --summary reports/experiments/v2-strong-baselines-confirmation-b.json --output reports/evidence/confirmation-b.compact.json
.venv/bin/recagent-eval replay-evidence --bundle reports/evidence/confirmation-b.compact.json --ledger reports/audit/2026-08-23-cohort-ledger.json --summary reports/experiments/v2-strong-baselines-confirmation-b.json
```

- [ ] **Step 6: Commit.**

```bash
git add src/recagent_eval/evidence_replay.py src/recagent_eval/cli.py tests/test_evidence_replay.py tests/test_cli.py reports/evidence
git commit -m "feat: commit replayable compact baseline evidence"
```

### Task 5: Evidence identity and peak-RSS correction reports

**Files:**
- Modify: `reports/experiments/v2-strong-baselines-confirmation-a.md`
- Modify: `reports/experiments/v2-strong-baselines-confirmation-b.md`
- Create: `reports/experiments/v2-baseline-evidence-corrections.md`

- [ ] **Step 1: Add tests or a factual-consistency script asserting A's
  development identity, B's sole-certification identity, and absence of active
  peak-memory Pareto claims.**

- [ ] **Step 2: Run RED against current reports.**

```bash
.venv/bin/pytest tests/test_scripts.py -q
```

- [ ] **Step 3: Write the correction addendum, mark old `peak_memory_mb` as
  `invalid_due_to_platform_unit_bug`, and revise A/B reports without altering
  their JSON or historical bug artifacts.**

- [ ] **Step 4: Run GREEN and commit.**

```bash
.venv/bin/pytest tests/test_scripts.py -q
git diff --check
git add reports/experiments tests/test_scripts.py
git commit -m "docs: correct confirmation identity and peak RSS evidence"
```

### Task 6: Reconcile stale project documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/HANDOFF-2026-08-22.md`
- Modify: `docs/project-methodology.md`
- Modify: `docs/demo-script.md`
- Modify: `reports/interview-pack/resume_star.md`
- Modify: `reports/interview-pack/interview-pack.md`
- Modify: `reports/interview-pack/interview_qa.md`
- Modify: `reports/interview-pack/ppt_prompt.md`
- Modify: `reports/interview-pack/application_checklist.md`

- [ ] **Step 1: Add stale-claim and machine-JSON consistency assertions to
  `tests/test_scripts.py`; run RED.**

- [ ] **Step 2: Rewrite the current-state narrative with values copied from
  replayed Confirmation-B JSON while retaining early negative results and Qwen
  pending status.**

- [ ] **Step 3: Run GREEN, search forbidden claims, and commit.**

```bash
.venv/bin/pytest tests/test_scripts.py -q
rg -n "both untouched|both confirmation cohorts|peak.memory.*Pareto|frozen.*rerun" README.md docs reports/interview-pack reports/experiments
git diff --check
git add README.md docs/HANDOFF-2026-08-22.md docs/project-methodology.md docs/demo-script.md reports/interview-pack tests/test_scripts.py
git commit -m "docs: reconcile project evidence with confirmation B"
```

### Task 7: Locked-parameter post-hoc robustness

**Files:**
- Create: `src/recagent_eval/robustness.py`
- Create: `tests/test_robustness.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `reports/experiments/v2-posthoc-robustness-protocol.md`
- Create after runs: `reports/experiments/v2-posthoc-robustness.json`
- Create after runs: `reports/experiments/v2-posthoc-robustness.md`

- [ ] **Step 1: Write RED tests that lock seeds `(42, 7, 2026)`, reject grid
  search or parameter drift, validate `recovered_after_run`, compute mean/sample
  std/worst seed, and preserve every seed.**

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/pytest tests/test_robustness.py -q
```

- [ ] **Step 3: Implement deterministic parameter recovery and a subprocess
  runner that accepts selected parameters directly and has no grid-selection
  path.**

- [ ] **Step 4: Run GREEN and a seed-7 smoke with a fresh temporary output.**

```bash
.venv/bin/pytest tests/test_robustness.py tests/test_baselines_bpr_mf.py tests/test_baselines_lightgcn.py tests/test_cli.py -q
```

- [ ] **Step 5: Hash existing seed-42 artifacts, run BPR seeds 7/2026, then
  run LightGCN seeds 7/2026 on Confirmation-B in separate processes.**

- [ ] **Step 6: Generate the machine and Markdown robustness reports, verify
  mean/std/worst-seed calculations, and confirm seed-42 source hashes did not
  change.**

- [ ] **Step 7: Commit.**

```bash
git add src/recagent_eval/robustness.py src/recagent_eval/cli.py tests/test_robustness.py tests/test_cli.py reports/experiments/v2-posthoc-robustness-protocol.md reports/experiments/v2-posthoc-robustness.json reports/experiments/v2-posthoc-robustness.md
git commit -m "exp: add post-hoc BPR and LightGCN robustness"
```

### Task 8: Final coverage hardening and P0 verification

**Files:**
- Modify only tests exposing remaining valuable behavior branches
- Modify: `docs/frozen-promotion-checklist-draft.md`

- [ ] **Step 1: Run the full coverage command and inspect exact remaining
  misses. Add RED tests for high-value baseline/evidence/CLI failure behavior
  until the unrounded result is at least 90.00%.**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered --cov-fail-under=90
```

- [ ] **Step 2: Materialize the promotion-manifest draft identities, exact
  label-free preflight command, future one-shot command, output path, and
  identity-derived marker path without executing either command.**

- [ ] **Step 3: Run every independent verification gate.**

```bash
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing:skip-covered --cov-fail-under=90
.venv/bin/ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv lock --check
git diff --check
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
.venv/bin/recagent-eval replay-evidence --bundle reports/evidence/confirmation-a.compact.json --ledger reports/audit/2026-08-23-cohort-ledger.json --summary reports/experiments/v2-strong-baselines-confirmation-a.json
.venv/bin/recagent-eval replay-evidence --bundle reports/evidence/confirmation-b.compact.json --ledger reports/audit/2026-08-23-cohort-ledger.json --summary reports/experiments/v2-strong-baselines-confirmation-b.json
```

- [ ] **Step 4: Verify cohort disjointness, frozen marker absence, stale claims,
  Markdown/JSON metrics, historical hashes, branch/HEAD, and clean worktree.**

- [ ] **Step 5: Commit final test/checklist changes, rerun affected gates, and
  stop before frozen preflight or consumption.**

```bash
git add tests docs/frozen-promotion-checklist-draft.md README.md docs reports
git commit -m "test: close evidence hygiene quality gates"
```

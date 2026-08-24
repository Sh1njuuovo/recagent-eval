# Final Promotion Narrative Implementation Plan

> **For agentic workers:** Execute inline in the existing isolated worktree.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all current project and interview materials accurately describe
the consumed 50-case promotion evaluation and its relationship to
Confirmation-B and historical DeepSeek evidence.

**Architecture:** One machine-readable final-evaluation summary is the numeric
source for a human-readable report. Current-state documents consume the same
three-layer evidence narrative; historical stage documents remain immutable.

**Tech Stack:** Markdown, JSON, repository evidence replay, `rg`, pytest/Ruff.

---

### Task 1: Publish the final evaluation evidence summary

**Files:**

- Create: `reports/experiments/v2-final-promotion-evaluation.json`
- Create: `reports/experiments/v2-final-promotion-evaluation.md`

- [ ] Copy metrics, marker identity, SHA, case count, and route recalls from
  `artifacts/frozen/7739942cdb8d3c58-6cf9d5bfd8d4eb3f/`.
- [ ] Bind the historical DeepSeek report with the identical case fingerprint.
- [ ] State that no matched ItemCF/ALS frozen comparison was executed.
- [ ] Record `no_further_tuning_on_case_suite: true` and the v3 holdout rule.
- [ ] Validate JSON parsing and arithmetic: 0.94 × 50 = 47 and 0.08 × 50 = 4.
- [ ] Commit the report separately.

### Task 2: Correct current project documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/HANDOFF-2026-08-22.md`
- Modify: `docs/project-methodology.md`
- Modify: `docs/demo-script.md`

- [ ] Replace current `unconsumed` language with completed one-shot promotion
  status and link the new report.
- [ ] Preserve Confirmation-B as the primary comparison and significance claim.
- [ ] Add the shared DeepSeek case-suite caveat and explicit no-tuning rule.
- [ ] Retain historical locked-gate statements only when clearly time-scoped.
- [ ] Commit current documentation corrections.

### Task 3: Refresh interview materials

**Files:**

- Modify: `reports/interview-pack/resume_star.md`
- Modify: `reports/interview-pack/interview-pack.md`
- Modify: `reports/interview-pack/interview_qa.md`
- Modify: `reports/interview-pack/ppt_prompt.md`
- Modify: `reports/interview-pack/application_checklist.md`

- [ ] Keep the 4–5-line resume headline centered on Confirmation-B.
- [ ] Add final-promotion numbers as a small generalization check.
- [ ] Add Q&A for shared case history, missing matched baselines, point-estimate
  decline, one-shot policy, ranking-depth bottleneck, and v3 holdout design.
- [ ] Remove requests to seek future frozen authorization.
- [ ] Commit interview-material updates.

### Task 4: Verify claim consistency

- [ ] Parse all committed JSON and compare the report to metrics/marker/DeepSeek
  fingerprints.
- [ ] Run stale-claim and unsupported-claim searches over current-state files.
- [ ] Run `pytest`, Ruff, `uv lock --check`, `git diff --check`, and shell syntax.
- [ ] Confirm the worktree is clean and report the final HEAD without pushing or
  merging.


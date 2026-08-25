# CI portability hardening design

**Date:** 2026-08-25

**Status:** approved in conversation; awaiting written-spec review

## Context

Pull Request #2 passes locally on macOS but fails both Ubuntu GitHub Actions
runs for two environment-specific reasons:

1. `score_current_v2b()` creates its workspace under the macOS-only
   `/private/tmp` directory.
2. A CLI test searches Rich/Typer output before removing ANSI styling, so the
   assertion depends on the runner's color behavior.

The failures do not involve ranking logic, experiment evidence, the promotion
package, or the consumed frozen result.

## Selected approach

Make the smallest portable changes:

- Let `tempfile.mkdtemp()` select the operating system's standard temporary
  directory. Do not add platform branches or create `/private/tmp` in CI.
- Add a regression test that rejects passing an explicit platform-specific
  temporary directory from `score_current_v2b()`.
- Normalize captured CLI output with Click's ANSI removal before asserting the
  `locked-params` diagnostic.
- Add a focused regression assertion showing the diagnostic remains detectable
  when ANSI codes are present.

## Alternatives rejected

- Creating `/private/tmp` in the workflow would hide a production portability
  defect.
- Weakening the CLI test to check only a nonzero exit code could accept a
  failure from the wrong validation branch.
- Adding OS-specific path conditionals would increase maintenance without
  improving behavior over Python's standard temporary-directory selection.

## Scope boundaries

The fix must not change:

- ranking, retrieval, feature, training, or evaluation behavior;
- experiment configs or dependency versions;
- promotion package members, identity, marker, metrics, or frozen evidence;
- README or resume result claims.

## Verification

Use RED-to-GREEN TDD for both failures, then run:

- the two focused regression tests;
- the complete pytest suite and coverage gate;
- Ruff;
- `uv lock --check`;
- `git diff --check`;
- shell syntax checks;
- GitHub Actions on the existing Pull Request #2.

The work is complete only when the PR checks are green.

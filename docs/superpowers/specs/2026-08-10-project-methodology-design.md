# RecAgent-Eval Project Methodology Document Design

Date: 2026-08-10

## Purpose

Create one project-level methodology document that serves two audiences without
mixing their evidence standards:

- developers use it to decide what to build, how to evaluate it, and when a
  frozen test may run;
- internship reviewers use it to understand the system choices, experimental
  discipline, negative results, and evidence behind public claims.

The published methodology lives at `docs/project-methodology.md`.

## Chosen structure

The document uses a dual-loop methodology plus an application-evidence layer:

1. an online decision loop covers preference understanding, constrained tool
   planning, hard filtering, retrieval, ranking, explanation, and fallback;
2. an offline evidence loop covers falsifiable hypotheses, chronological data
   boundaries, stage metrics, ablations, validation gates, frozen testing, and
   failure-driven iteration;
3. a Claim-to-Evidence layer connects code and experiment artifacts to README,
   resume, demo, and interview statements.

This was selected over a pipeline-only document, which would understate the
experimental method, and a research-only document, which would make the product
and Agent architecture harder to understand.

## Evidence boundary

The document explicitly separates:

- v1 facts already supported by the repository;
- conclusions that v1 does not support;
- v2 targets that remain hypotheses until their gates pass.

No planned embedding model, learned ranker, statistical result, Qwen run, or
Demo capability is described as completed work.

## Content requirements

The methodology must include:

- problem decomposition and module responsibility;
- online and offline data flow;
- hard-versus-soft constraint policy;
- stage-level metrics and failure localization;
- validation selection and frozen-test rules;
- error handling and deterministic degradation;
- engineering, testing, Demo, and artifact practices;
- Claim-to-Evidence rules and an interview-ready explanation;
- an explicit application-readiness gate.

It must retain the current negative ranking result and explain why higher union
candidate recall does not imply better Top-10 ranking.

## Non-goals

- Selecting the exact v2 embedding model or ranker library.
- Implementing v2 features.
- Rerunning DeepSeek or Qwen.
- Changing checked-in metrics, cases, configurations, or source code.
- Claiming that the project is ready for applications.

## Acceptance criteria

- `docs/project-methodology.md` is understandable without reading the source.
- Every numeric v1 claim matches existing repository evidence.
- Planned work is grammatically and visibly separated from completed work.
- There are no placeholders or ambiguous permissions to tune on the test set.
- The document gives both an implementation decision rule and an interview
  narrative.
- Only documentation files are committed as part of this change.

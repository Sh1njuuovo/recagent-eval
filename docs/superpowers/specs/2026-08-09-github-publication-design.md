# GitHub Publication Design

## Goal

Publish RecAgent-Eval as a public internship portfolio repository at
`Sh1njuuovo/recagent-eval`, with `main` as the default branch and no secrets,
private logs, or redistributed MovieLens data.

## Public Story

The repository presents an evaluation-first conversational recommendation
Agent for search/recommendation and LLM internship roles. Its public claims are
limited to versioned evidence:

- schema-validated planning and deterministic recommendation tools;
- 100% formal plan, tool, pipeline, and hard-constraint reliability metrics;
- candidate union recall increasing from 0.78 to 0.88;
- retained negative top-10 ranking results;
- a validation-only RRF/percentile follow-up that kept the frozen test locked;
- 79 automated tests and 90% line coverage.

Qwen/vLLM throughput and memory numbers remain explicitly pending.

## Repository Shape

The current verified commit becomes the public `main` branch. The repository
includes source, tests, configurations, fixed cases, aggregate evidence,
reports, scripts, license, attribution, and interview/demo documentation.

The following remain local or ignored:

- MovieLens raw archives and extracted data;
- `.venv`, caches, and local worktrees;
- API keys and environment files;
- per-episode DeepSeek logs and private prompts;
- generated demo runtime outputs not already selected as aggregate evidence.

The publication process does not rewrite experiment history or squash the
evidence-motivated commits.

## README and Metadata

Before publication, the README receives only presentation-focused changes:

- a concise result-first opening;
- status badges that do not depend on unconfigured CI;
- a compact evidence table and reproducibility route;
- an explicit negative-result statement;
- links to the detailed reports, demo script, and interview pack.

Repository metadata:

- Name: `recagent-eval`
- Visibility: public
- Description: `Evaluation-first conversational movie recommendation agent with structured LLM planning, deterministic retrieval, and frozen-test gates.`
- Topics: `recommender-system`, `llm-agent`, `movielens`, `deepseek`,
  `information-retrieval`, `evaluation`, `python`

No release or package registry publication is part of this iteration.

## Identity and Attribution

The MIT license remains. The copyright line may identify `Sh1njuuovo` as the
project author while `NOTICE` preserves Microsoft RecAI/InteRecAgent attribution
and states that no upstream source files are vendored. MovieLens remains a
download-only dependency subject to GroupLens terms.

## Authentication and External Actions

Authentication uses `gh auth login --web`. A token must not be sent through the
chat, written to a repository file, or embedded in a command. After successful
authentication, the workflow creates the public repository, adds `origin`,
pushes `main`, and applies description/topics through GitHub CLI.

External mutations are limited to:

1. creating `Sh1njuuovo/recagent-eval` if it does not exist;
2. pushing the verified `main` branch;
3. setting repository description and topics.

No issue, pull request, release, discussion, webhook, secret, or collaborator is
created.

## Safety and Verification

Pre-push checks:

- clean worktree and expected branch tip;
- full Ruff and pytest/coverage verification;
- tracked-file size review;
- tracked-file and history scan for common secret patterns;
- confirmation that MovieLens raw data and `.env` files are untracked;
- README, LICENSE, NOTICE, and aggregate evidence consistency.

Post-push checks:

- remote URL and default branch;
- public visibility, description, and topics;
- README rendering and Mermaid presence;
- key report links resolve;
- local HEAD equals `origin/main`.

If authentication fails, publication pauses without asking for a token in chat.
If the target repository already exists with unrelated content, the workflow
stops before pushing and reports the conflict. No force-push is allowed.

## Acceptance Criteria

- `https://github.com/Sh1njuuovo/recagent-eval` is public and uses `main`.
- The remote contains the locally verified commit without force-push.
- No raw MovieLens data, API key, `.env`, or private episode log is tracked.
- README claims match the versioned JSON/Markdown reports.
- LICENSE and NOTICE preserve project ownership and upstream attribution.
- Repository description and topics are set.
- Local and remote `main` point to the same commit.


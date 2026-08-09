# GitHub Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the verified RecAgent-Eval repository as the public `Sh1njuuovo/recagent-eval` GitHub portfolio repository with `main` as its default branch and no secrets or raw data.

**Architecture:** First prove the tracked Git history is safe to publish, then make a small evidence-first README/attribution commit. Authenticate through GitHub's web flow, create or safely validate the target repository, push a new local `main` without force, apply metadata, and verify the public state against the local commit.

**Tech Stack:** Git, GitHub CLI, Python 3.11+, uv, Ruff, pytest, Markdown, GitHub repository metadata.

---

## File Map

- Modify `README.md`: add stable badges, result-first evidence summary, and a
  direct demo/interview navigation block.
- Modify `LICENSE`: identify `Sh1njuuovo` as the repository author.
- Modify `NOTICE`: retain RecAI attribution while naming the independent
  implementation owner.
- Create `docs/superpowers/plans/2026-08-09-github-publication.md`: preserve
  this publication procedure.
- No source code, experiment aggregate, fixed case, or configuration is changed.
- No `.github/workflows` file is added because CI is not configured in this
  publication step.

### Task 1: Prove the repository is safe to publish

**Files:**
- Inspect: all tracked files and Git history
- Inspect: `.gitignore`
- Inspect: `.env.example`

- [ ] **Step 1: Confirm clean state and publication source commit**

Run:

```bash
git status --short
git branch --show-current
git log -1 --oneline
git remote -v
```

Expected: clean worktree on `feat/recagent-eval`, one verified local HEAD, and
no configured remote.

- [ ] **Step 2: Verify tests and lint before public edits**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  --cov=recagent_eval --cov-report=term-missing
```

Expected: Ruff exits 0, 79 tests pass, and line coverage rounds to 90%.

- [ ] **Step 3: Scan tracked paths and sizes**

Run:

```bash
git ls-files
uv run python -c 'from pathlib import Path; files=[(Path(p).stat().st_size,p) for p in __import__("subprocess").check_output(["git","ls-files"],text=True).splitlines()]; large=[(n,p) for n,p in files if n>5_000_000]; assert not large, large; print(f"tracked_files={len(files)} max_bytes={max(n for n,_ in files)}")'
git ls-files 'data/**' 'artifacts/runs/**' '.env' '.env.*'
```

Expected: no tracked file exceeds 5 MB; no raw MovieLens path, run directory,
or real `.env` is listed. `.env.example` is allowed.

- [ ] **Step 4: Scan tracked content and history for credential shapes**

Run:

```bash
git grep -qEI 'sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}' -- .; test $? -eq 1
git log -p --all | rg -q 'sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}'; test $? -eq 1
git grep -n 'DEEPSEEK_API_KEY' -- .
```

Expected: the first two commands exit successfully only when they find no
credential-shaped match, without printing any candidate secret.
`DEEPSEEK_API_KEY`
appears only as an environment-variable name or placeholder, never a value.

- [ ] **Step 5: Confirm ignored private/runtime material**

Run:

```bash
git check-ignore -v data/raw/ml-1m/ratings.dat
git check-ignore -v artifacts/runs/private-check/episodes.jsonl
git check-ignore -v .venv/pyvenv.cfg
git check-ignore -v .env
```

Expected: every path is ignored by a specific `.gitignore` rule.

If any credential-shaped value, raw data file, or private log is tracked or
present in history, stop before authentication or repository creation. Do not
attempt history rewriting without a separate approved design.

### Task 2: Polish the public landing page and attribution

**Files:**
- Modify: `README.md`
- Modify: `LICENSE`
- Modify: `NOTICE`

- [ ] **Step 1: Add stable badges and a result-first summary**

Immediately below the README title, add:

```markdown
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-79%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)

An evaluation-first conversational movie recommendation Agent that separates
LLM preference understanding from deterministic filtering, retrieval, ranking,
and frozen-test evaluation.

**Verified evidence:** structured Agent reliability and hard-constraint metrics
reach 100%; hybrid retrieval raises candidate union recall from 78% to 88%, but
the retained full-system NDCG@10 is 0.0149. A validation-only RRF/percentile
follow-up did not pass the preregistered ItemCF gate, so the frozen test was not
rerun.
```

Replace the existing two-sentence opening rather than duplicating it.

- [ ] **Step 2: Add public navigation links**

Before `## Architecture`, add:

```markdown
## Start here

- [Formal DeepSeek evaluation](reports/experiments/deepseek-constraint-aware.md)
- [Offline ranker gate](reports/experiments/offline-ranker-selection.md)
- [Ten-minute demo script](docs/demo-script.md)
- [Core code walkthrough](docs/core-code-walkthrough.md)
- [Interview pack](reports/interview-pack/interview-pack.md)
```

- [ ] **Step 3: Identify the independent implementation owner**

Change the LICENSE copyright line to:

```text
Copyright (c) 2026 Sh1njuuovo
```

Prepend NOTICE with:

```text
RecAgent-Eval is independently implemented and maintained by Sh1njuuovo.

```

Do not alter the existing RecAI/InteRecAgent or MovieLens attribution text.

- [ ] **Step 4: Verify public claims against evidence**

Run:

```bash
uv run python -c 'import json; d=json.load(open("reports/experiments/deepseek-constraint-aware.json")); a=json.load(open("artifacts/ranker_ablation.json")); assert a["test_unlocked"] is False; assert len(a["rows"])==17; print("ranker gate locked and 17 rows verified")'
rg -n '0\.0149|0\.8800|79-test|90%' README.md reports/experiments/deepseek-constraint-aware.md reports/experiments/offline-ranker-selection.md
git diff --check
```

Expected: the JSON checks pass, README values have supporting report matches,
and no whitespace errors appear.

- [ ] **Step 5: Run full verification and commit the publication polish**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  --cov=recagent_eval --cov-report=term-missing
git add README.md LICENSE NOTICE
git commit -m "docs: prepare public GitHub portfolio"
```

Expected: Ruff passes, 79 tests pass with 90% line coverage, and the commit
contains only the three public-facing files.

### Task 3: Authenticate and prepare a public `main`

**Files:**
- Modify: local Git refs and GitHub CLI credential store
- External target: `github.com/Sh1njuuovo/recagent-eval`

- [ ] **Step 1: Authenticate through the browser flow**

Run:

```bash
gh auth login -h github.com -p https --web
gh auth status -h github.com
```

Expected: GitHub reports account `Sh1njuuovo` authenticated. Do not accept or
request a token through chat, a command argument, stdin captured in logs, or a
repository file.

- [ ] **Step 2: Check for a repository-name collision**

Run:

```bash
gh repo view Sh1njuuovo/recagent-eval \
  --json nameWithOwner,isPrivate,defaultBranchRef,url
```

Expected for a new publication: GitHub reports the repository is not found. If
it exists, inspect its default branch and refs with read-only `gh repo view` and
`gh api repos/Sh1njuuovo/recagent-eval/branches`; stop if it contains unrelated
content. Never force-push.

- [ ] **Step 3: Create and switch to local `main`**

Run:

```bash
git switch -c main
git status --short
git rev-parse HEAD
```

Expected: a clean local `main` at the verified publication commit. Keep
`feat/recagent-eval` as a recoverable local branch.

### Task 4: Create, push, and configure the GitHub repository

**Files:**
- Modify: local Git remote configuration
- External target: public GitHub repository metadata and `main` ref

- [ ] **Step 1: Create the empty public repository**

Run:

```bash
gh repo create Sh1njuuovo/recagent-eval \
  --public \
  --description 'Evaluation-first conversational movie recommendation agent with structured LLM planning, deterministic retrieval, and frozen-test gates.'
```

Expected: GitHub returns the new repository URL. If repository creation fails,
stop without changing remote configuration or deleting anything.

- [ ] **Step 2: Add the HTTPS remote and push without force**

Run:

```bash
git remote add origin https://github.com/Sh1njuuovo/recagent-eval.git
git push -u origin main
```

Expected: a normal first push succeeds and local `main` tracks `origin/main`.
No `--force` or `--force-with-lease` is permitted.

- [ ] **Step 3: Apply topics and default branch**

Run:

```bash
gh repo edit Sh1njuuovo/recagent-eval \
  --default-branch main \
  --add-topic recommender-system \
  --add-topic llm-agent \
  --add-topic movielens \
  --add-topic deepseek \
  --add-topic information-retrieval \
  --add-topic evaluation \
  --add-topic python
```

Expected: metadata update succeeds. Do not enable Discussions, Pages, Actions,
webhooks, collaborators, or repository secrets.

### Task 5: Verify the public repository end to end

**Files:**
- Inspect: local refs, remote refs, and public GitHub metadata

- [ ] **Step 1: Compare local and remote commits**

Run:

```bash
git fetch origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
```

Expected: local HEAD equals `origin/main`, the worktree is clean, and local
`main` tracks the remote branch.

- [ ] **Step 2: Verify public metadata and key files**

Run:

```bash
gh repo view Sh1njuuovo/recagent-eval \
  --json nameWithOwner,isPrivate,defaultBranchRef,description,repositoryTopics,url
gh api repos/Sh1njuuovo/recagent-eval/readme --jq '.html_url'
gh api repos/Sh1njuuovo/recagent-eval/contents/LICENSE --jq '.html_url'
gh api repos/Sh1njuuovo/recagent-eval/contents/NOTICE --jq '.html_url'
gh api repos/Sh1njuuovo/recagent-eval/contents/reports/experiments/offline-ranker-selection.md --jq '.html_url'
```

Expected: `isPrivate=false`, default branch `main`, the exact description and
seven topics are present, and every key file resolves publicly.

- [ ] **Step 3: Confirm no forbidden public paths**

Run:

```bash
gh api repos/Sh1njuuovo/recagent-eval/git/trees/main?recursive=1 \
  --jq '.tree[].path' | \
  rg -n '(^|/)data/raw/|(^|/)artifacts/runs/|(^|/)\.env$|episodes\.jsonl$' || true
```

Expected: no forbidden path is printed. `.env.example` is allowed.

- [ ] **Step 4: Record the final publication status**

Report:

- public repository URL;
- published commit SHA;
- default branch and visibility;
- local test/lint/coverage evidence;
- confirmation that raw data, secrets, and private episode logs are absent;
- any remaining optional work, such as Qwen smoke testing or adding CI in a
  separately approved iteration.

If the remote was created but a later step fails, preserve both local history
and the remote repository, report the exact failed step, and resume from that
step. Do not delete the repository automatically.

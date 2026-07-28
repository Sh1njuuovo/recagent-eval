# Constraint-Aware Retrieval and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fixed cases constraint-consistent, make configured retrieval routes mandatory, improve deterministic semantic candidate recall, and report stage-level retrieval evidence before rerunning DeepSeek.

**Architecture:** Keep the LLM responsible for preference extraction and typed planning, while `AgentConfig` owns the minimum executable retrieval policy. Add deterministic content-profile TF-IDF queries and candidate IDs to tool traces, then compute label eligibility, route-level candidate recall, and pipeline compliance in the evaluation layer. Select retrieval depth and profile cap only on validation users before freezing the new formal test configuration.

**Tech Stack:** Python 3.11, Pydantic v2, NumPy, Typer, PyYAML, pytest, MovieLens-1M, DeepSeek OpenAI-compatible API

---

## File Structure

- Modify `src/recagent_eval/cases.py`: construct constraint-consistent multi-turn cases and expose case relevance preflight validation.
- Modify `src/recagent_eval/models.py`: retain ordered candidate movie IDs in tool traces.
- Modify `src/recagent_eval/agent.py`: enforce configured retrieval policy, build capped content profiles, and trace candidate IDs.
- Modify `src/recagent_eval/config.py`: load required retrieval tools and semantic history cap.
- Modify `src/recagent_eval/runner.py`: preflight cases, preserve final-turn traces, compute diagnostics, and record the execution policy.
- Modify `src/recagent_eval/evaluation.py`: define candidate diagnostics and aggregate stage-level metrics.
- Modify `src/recagent_eval/tuning.py`: evaluate retrieval-depth/profile-cap combinations using validation targets.
- Modify `src/recagent_eval/cli.py`: expose retrieval selection and frozen-config generation.
- Modify `configs/baseline.yaml`, `configs/structured_memory.yaml`, and `configs/full.yaml`: declare each variant's retrieval policy.
- Create `tests/test_retrieval_selection.py`: cover validation-only selection and deterministic tie-breaking.
- Modify focused existing test modules beside each production change.
- Regenerate `cases/fixed_cases.json`, `cases/multi_turn_cases.json`, `cases/stability_cases.json`, and `cases/qwen_smoke_cases.json` through existing CLI commands.
- Create `reports/experiments/deepseek-constraint-aware.md`: preserve old results and report the new frozen evaluation separately.

### Task 1: Enforce relevance-label eligibility in fixed cases

**Files:**
- Modify: `src/recagent_eval/cases.py`
- Modify: `tests/test_cases.py`

- [ ] **Step 1: Write failing tests for conflict-free negative genres and preflight errors**

Add imports and tests to `tests/test_cases.py`:

```python
import pytest

from recagent_eval.cases import (
    EvaluationCase,
    generate_cases,
    select_stratified_cases,
    validate_case_relevance,
)


def test_multi_turn_negative_genre_does_not_conflict_with_target() -> None:
    movies = {
        1: Movie(1, "History", ("Comedy",), 1998),
        2: Movie(2, "Validation", ("Drama",), 1999),
        3: Movie(3, "Target", ("Comedy", "Romance"), 2000),
        4: Movie(4, "Catalog Action", ("Action",), 2001),
    }
    ratings = [
        Rating(1, 1, 5, 1),
        Rating(1, 2, 5, 2),
        Rating(1, 3, 5, 3),
    ]
    split = chronological_split(ratings)

    case = generate_cases(
        movies,
        split,
        ratings,
        single_turn_count=0,
        multi_turn_count=1,
        seed=7,
    )[0]

    excluded = case.expected_preferences.excluded_genres
    assert excluded
    assert not (excluded & set(movies[3].genres))
    validate_case_relevance(case, movies)


def test_case_preflight_reports_conflicting_relevance_target() -> None:
    movies = {9: Movie(9, "Target", ("Action",), 2000)}
    case = EvaluationCase(
        case_id="bad-case",
        user_id=1,
        turns=("Avoid Action.",),
        relevant_movie_ids={9},
        initial_state=PreferenceState(excluded_genres={"Action"}),
    )

    with pytest.raises(
        ValueError,
        match=r"bad-case.*movie 9.*excluded genre Action",
    ):
        validate_case_relevance(case, movies)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_cases.py -v
```

Expected: collection fails because `validate_case_relevance` is not defined.

- [ ] **Step 3: Implement case validation and target-aware negative-genre selection**

Add to `src/recagent_eval/cases.py`:

```python
def validate_case_relevance(
    case: EvaluationCase,
    movies: dict[int, Movie],
) -> None:
    state = case.expected_preferences or case.initial_state
    blocked_ids = (
        state.liked_movie_ids
        | state.disliked_movie_ids
        | state.excluded_movie_ids
    )
    for movie_id in sorted(case.relevant_movie_ids):
        movie = movies.get(movie_id)
        if movie is None:
            raise ValueError(
                f"{case.case_id}: relevant movie {movie_id} is missing from metadata"
            )
        reasons: list[str] = []
        if movie_id in blocked_ids:
            reasons.append("blocked movie ID")
        genres = set(movie.genres)
        excluded = sorted(state.excluded_genres & genres)
        if excluded:
            reasons.append(f"excluded genre {', '.join(excluded)}")
        if state.required_genres and not state.required_genres.issubset(genres):
            reasons.append("missing required genre")
        if state.year_min is not None and (
            movie.year is None or movie.year < state.year_min
        ):
            reasons.append("below minimum year")
        if state.year_max is not None and (
            movie.year is None or movie.year > state.year_max
        ):
            reasons.append("above maximum year")
        if reasons:
            raise ValueError(
                f"{case.case_id}: relevant movie {movie_id} violates "
                + "; ".join(reasons)
            )


def validate_cases_relevance(
    cases: list[EvaluationCase],
    movies: dict[int, Movie],
) -> None:
    for case in cases:
        validate_case_relevance(case, movies)


def _different_genre(
    liked_genre: str,
    movies: dict[int, Movie],
    *,
    forbidden_genres: set[str],
) -> str | None:
    genres = sorted({genre for movie in movies.values() for genre in movie.genres})
    return next(
        (
            genre
            for genre in genres
            if genre != liked_genre and genre not in forbidden_genres
        ),
        None,
    )
```

In `generate_cases`, scan shuffled users until `multi_turn_count` users have a
non-`None` negative genre for their test target. Store `(user_id, disliked_genre)`
pairs, build the multi-turn cases from those pairs, and call
`validate_cases_relevance(cases, movies)` before returning.

Update the deterministic-mix fixture's movie catalog with an unrated
`Movie(13, "Action Catalog", ("Action",), 2000)` so the synthetic dataset has a
valid negative genre that is absent from every held-out Drama target.

- [ ] **Step 4: Run case tests**

Run:

```bash
.venv/bin/pytest tests/test_cases.py -v
```

Expected: all case tests pass, including deterministic mix and conflict checks.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/cases.py tests/test_cases.py
git commit -m "fix: make evaluation labels constraint-consistent"
```

### Task 2: Make retrieval policy configuration-owned

**Files:**
- Modify: `src/recagent_eval/agent.py`
- Modify: `src/recagent_eval/config.py`
- Modify: `src/recagent_eval/runner.py`
- Modify: `configs/baseline.yaml`
- Modify: `configs/structured_memory.yaml`
- Modify: `configs/full.yaml`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for required retrieval routes**

Add to `tests/test_agent.py`:

```python
def itemcf_only_response() -> LLMResponse:
    return LLMResponse(
        structured={
            "preference_patch": {},
            "steps": [
                {"tool": "hard_filter", "args": {}},
                {"tool": "itemcf_retrieve", "args": {"top_k": 100}},
                {"tool": "rerank", "args": {"top_k": 2}},
                {"tool": "explain", "args": {}},
            ],
        }
    )


def test_full_profile_repairs_plan_missing_semantic_route() -> None:
    provider = CapturingProvider([itemcf_only_response(), valid_response()])
    agent = RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((0.5, 0.3, 0.2)),
        provider=provider,
        config=AgentConfig(
            required_retrieval_tools=(
                "itemcf_retrieve",
                "semantic_retrieve",
            )
        ),
    )

    result = agent.recommend("recommend", PreferenceState(liked_movie_ids={1}))

    assert result.llm_calls == 2
    assert result.plan_valid is True
    assert "semantic_retrieve" in provider.messages[0][0]["content"]


def test_full_profile_fallback_retains_both_required_routes() -> None:
    provider = SequenceProvider([itemcf_only_response(), itemcf_only_response()])
    agent = RecommendationAgent(
        movies=MOVIES,
        itemcf=ItemCFRetriever.fit(RATINGS),
        semantic=TfidfSemanticRetriever.fit(MOVIES),
        ranker=HybridRanker((0.5, 0.3, 0.2)),
        provider=provider,
        config=AgentConfig(
            required_retrieval_tools=(
                "itemcf_retrieve",
                "semantic_retrieve",
            )
        ),
    )

    result = agent.recommend("recommend", PreferenceState(liked_movie_ids={1}))

    assert result.fallback_used is True
    assert [step.tool for step in result.plan.steps].count("semantic_retrieve") == 1
```

Add a config assertion to `tests/test_cli.py` that loads a YAML with:

```yaml
name: hybrid
required_retrieval_tools: [itemcf_retrieve, semantic_retrieve]
semantic_profile_history_cap: 20
```

and asserts both fields are present in `show-config` JSON.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_agent.py tests/test_cli.py -v
```

Expected: failures mention unknown `required_retrieval_tools` and missing config
fields.

- [ ] **Step 3: Add policy fields and profile-aware plan validation**

In `src/recagent_eval/agent.py`, define:

```python
@dataclass(frozen=True)
class AgentConfig:
    retrieval_top_k: int = 100
    provider_timeout_seconds: float = 30
    enable_memory: bool = True
    enable_semantic_retrieval: bool = True
    structured_planning: bool = True
    required_retrieval_tools: tuple[ToolName, ...] = ("itemcf_retrieve",)
    semantic_profile_history_cap: int = 20


def _plan_has_required_retrieval(
    plan: ToolPlan,
    required: tuple[ToolName, ...],
) -> bool:
    names = {step.tool for step in plan.steps}
    return set(required).issubset(names)
```

Pass `required_retrieval_tools` into `_parse_planning_response`; return `None`
when the parsed plan omits a required tool. Build `_plan_safety_instructions`
from the configured tuple and pass it into both planning and repair messages.
Change `_fallback_plan` to append exactly the configured retrieval tools between
`hard_filter` and `rerank`.

Add matching fields to `ExperimentConfig`, pass them into `AgentConfig`, load
them in `config.py`, and save them in `run_manifest.json`.

- [ ] **Step 4: Declare policies in all three YAML files**

Use:

```yaml
# configs/baseline.yaml and configs/structured_memory.yaml
required_retrieval_tools: [itemcf_retrieve]
semantic_profile_history_cap: 20
```

and:

```yaml
# configs/full.yaml
required_retrieval_tools: [itemcf_retrieve, semantic_retrieve]
semantic_profile_history_cap: 20
```

- [ ] **Step 5: Run agent, config, and runner tests**

Run:

```bash
.venv/bin/pytest tests/test_agent.py tests/test_cli.py tests/test_runner.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/recagent_eval/agent.py src/recagent_eval/config.py \
  src/recagent_eval/runner.py configs tests/test_agent.py tests/test_cli.py
git commit -m "feat: enforce configured retrieval policy"
```

### Task 3: Build deterministic content profiles and trace candidates

**Files:**
- Modify: `src/recagent_eval/models.py`
- Modify: `src/recagent_eval/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests for capped profiles and candidate traces**

Add to `tests/test_agent.py`:

```python
from recagent_eval.agent import build_semantic_profile


def test_semantic_profile_is_ordered_capped_and_omits_negative_tokens() -> None:
    state = PreferenceState(
        liked_movie_ids={3, 1, 2},
        liked_genres={"Sci-Fi"},
        disliked_genres={"Horror"},
        excluded_genres={"Action"},
    )

    profile = build_semantic_profile(
        "recommend something",
        state,
        MOVIES,
        history_cap=2,
    )

    assert "Space Quest" in profile
    assert "Galaxy War" in profile
    assert "Quiet Drama" not in profile
    assert "Sci-Fi" in profile
    assert "Horror" not in profile


def test_retrieval_traces_save_ordered_candidate_ids() -> None:
    result = make_agent(SequenceProvider([valid_response()])).recommend(
        "science fiction",
        PreferenceState(liked_movie_ids={1}),
    )

    traces = {trace.tool: trace for trace in result.traces}
    assert traces["itemcf_retrieve"].candidate_movie_ids == [2]
    assert traces["semantic_retrieve"].candidate_movie_ids
    assert traces["rerank"].candidate_movie_ids == [2]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_agent.py -v
```

Expected: import failure for `build_semantic_profile` and missing
`candidate_movie_ids`.

- [ ] **Step 3: Add trace storage and content-profile construction**

Add to `ToolTrace` in `src/recagent_eval/models.py`:

```python
candidate_movie_ids: list[int] = Field(default_factory=list)
```

Add to `src/recagent_eval/agent.py`:

```python
def build_semantic_profile(
    message: str,
    state: PreferenceState,
    movies: dict[int, Movie],
    *,
    history_cap: int,
) -> str:
    parts = [message]
    parts.extend(sorted(state.liked_genres))
    for movie_id in sorted(state.liked_movie_ids)[:history_cap]:
        movie = movies.get(movie_id)
        if movie is not None:
            parts.append(movie.text)
    return " ".join(parts)
```

Use this function in `semantic_retrieve`. When appending traces, save ordered
IDs from ItemCF tuples, semantic tuples, and `RecommendedMovie` results.
Hard-filter and lookup traces keep an empty candidate-ID list to avoid writing
the full catalog into every episode.

- [ ] **Step 4: Run agent and model tests**

Run:

```bash
.venv/bin/pytest tests/test_agent.py tests/test_models.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/models.py src/recagent_eval/agent.py tests/test_agent.py
git commit -m "feat: add content-profile semantic retrieval traces"
```

### Task 4: Add stage-level diagnostics and fail-fast preflight

**Files:**
- Modify: `src/recagent_eval/evaluation.py`
- Modify: `src/recagent_eval/runner.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing diagnostics tests**

Add to `tests/test_evaluation.py`:

```python
from recagent_eval.evaluation import (
    build_candidate_diagnostics,
    pipeline_compliant,
)


def test_candidate_diagnostic_distinguishes_retrieval_and_ranking_miss() -> None:
    movies = {
        7: Movie(7, "Target", ("Drama",), 2000),
        8: Movie(8, "Other", ("Drama",), 2001),
    }
    traces = [
        ToolTrace(
            tool="itemcf_retrieve",
            candidate_movie_ids=[8, 7],
        ),
        ToolTrace(
            tool="semantic_retrieve",
            candidate_movie_ids=[8],
        ),
        ToolTrace(tool="rerank", candidate_movie_ids=[8]),
    ]

    diagnostic = build_candidate_diagnostics(
        {7},
        movies,
        PreferenceState(),
        traces,
    )[0]

    assert diagnostic["eligible"] is True
    assert diagnostic["itemcf_rank"] == 2
    assert diagnostic["semantic_rank"] is None
    assert diagnostic["union_member"] is True
    assert diagnostic["final_rank"] is None


def test_pipeline_compliance_checks_required_order() -> None:
    traces = [
        ToolTrace(tool="hard_filter"),
        ToolTrace(tool="itemcf_retrieve"),
        ToolTrace(tool="semantic_retrieve"),
        ToolTrace(tool="rerank"),
    ]

    assert pipeline_compliant(
        traces,
        ("itemcf_retrieve", "semantic_retrieve"),
    )
    assert not pipeline_compliant(
        traces[:-2] + [ToolTrace(tool="rerank")],
        ("itemcf_retrieve", "semantic_retrieve"),
    )
```

Add to `tests/test_runner.py` a provider whose `chat` raises
`AssertionError("provider must not be called")`; pass it an inconsistent case
and assert `run_experiment` raises the preflight `ValueError`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py tests/test_runner.py -v
```

Expected: diagnostics imports fail and the runner still calls the provider.

- [ ] **Step 3: Implement diagnostics helpers**

Add to `src/recagent_eval/evaluation.py`:

```python
def _rank(movie_id: int, ids: list[int]) -> int | None:
    try:
        return ids.index(movie_id) + 1
    except ValueError:
        return None


def build_candidate_diagnostics(
    relevant_movie_ids: set[int],
    movies: dict[int, Movie],
    state: PreferenceState,
    traces: list[ToolTrace],
) -> list[dict[str, Any]]:
    by_tool = {trace.tool: trace for trace in traces}
    itemcf_ids = by_tool.get(
        "itemcf_retrieve",
        ToolTrace(tool="itemcf_retrieve"),
    ).candidate_movie_ids
    semantic_ids = by_tool.get(
        "semantic_retrieve",
        ToolTrace(tool="semantic_retrieve"),
    ).candidate_movie_ids
    ranked_ids = by_tool.get(
        "rerank",
        ToolTrace(tool="rerank"),
    ).candidate_movie_ids
    allowed = {movie.movie_id for movie in hard_filter(movies.values(), state)}
    return [
        {
            "movie_id": movie_id,
            "eligible": movie_id in allowed,
            "itemcf_rank": _rank(movie_id, itemcf_ids),
            "semantic_rank": _rank(movie_id, semantic_ids),
            "union_member": movie_id in set(itemcf_ids) | set(semantic_ids),
            "final_rank": _rank(movie_id, ranked_ids),
        }
        for movie_id in sorted(relevant_movie_ids)
    ]


def pipeline_compliant(
    traces: list[ToolTrace],
    required_tools: tuple[ToolName, ...],
) -> bool:
    names = [trace.tool for trace in traces if trace.success]
    if "hard_filter" not in names or "rerank" not in names:
        return False
    positions = [names.index(tool) for tool in required_tools if tool in names]
    return (
        len(positions) == len(required_tools)
        and names.index("hard_filter") < min(positions)
        and max(positions) < names.index("rerank")
    )
```

Import `hard_filter`, `ToolName`, and `ToolTrace`. Extend aggregate metrics to
read `candidate_diagnostics` and `pipeline_compliant` from record metadata and
report:

```python
"relevance_label_eligibility_rate": _mean(
    [float(record.metadata["label_eligible"]) for record in records]
),
"final_state_target_eligibility_rate": _mean(final_state_eligibility),
"itemcf_candidate_recall": _mean(itemcf_candidate_hits),
"semantic_candidate_recall": _mean(semantic_candidate_hits),
"union_candidate_recall": _mean(union_candidate_hits),
"pipeline_compliance_rate": _mean(pipeline_compliance),
```

- [ ] **Step 4: Wire preflight and final-turn diagnostics into the runner**

Call `validate_cases_relevance(cases, movies)` before fitting retrievers or
calling the provider. Preserve `final_turn_traces = turn_results[-1].traces`,
build diagnostics from those traces, and place both diagnostics and compliance
in `EvaluationRecord.metadata` and the serialized episode. Set
`metadata["label_eligible"] = True` only after the whole fixed matrix passes
preflight. Add the required retrieval tools to the manifest.

- [ ] **Step 5: Run evaluation and runner tests**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py tests/test_runner.py -v
```

Expected: all selected tests pass and runner fixtures report pipeline compliance.

- [ ] **Step 6: Commit**

```bash
git add src/recagent_eval/evaluation.py src/recagent_eval/runner.py \
  tests/test_evaluation.py tests/test_runner.py
git commit -m "feat: report candidate-stage evaluation diagnostics"
```

### Task 5: Select retrieval parameters on validation data

**Files:**
- Modify: `src/recagent_eval/tuning.py`
- Modify: `src/recagent_eval/cli.py`
- Create: `tests/test_retrieval_selection.py`

- [ ] **Step 1: Write a failing deterministic selection test**

Create `tests/test_retrieval_selection.py`:

```python
from recagent_eval.tuning import select_retrieval_parameters


def test_selection_uses_validation_targets_and_deterministic_ties(monkeypatch) -> None:
    rows = [
        {
            "retrieval_top_k": 100,
            "semantic_profile_history_cap": 10,
            "ndcg_at_10": 0.2,
            "union_candidate_recall": 0.5,
        },
        {
            "retrieval_top_k": 200,
            "semantic_profile_history_cap": 20,
            "ndcg_at_10": 0.2,
            "union_candidate_recall": 0.6,
        },
    ]
    monkeypatch.setattr(
        "recagent_eval.tuning.build_retrieval_ablation",
        lambda *args, **kwargs: rows,
    )

    selection = select_retrieval_parameters(object(), object())

    assert selection["retrieval_top_k"] == 200
    assert selection["semantic_profile_history_cap"] == 20
    assert selection["selection_metric"] == "validation_ndcg_at_10"
```

- [ ] **Step 2: Run the new test and verify failure**

Run:

```bash
.venv/bin/pytest tests/test_retrieval_selection.py -v
```

Expected: import failure for `select_retrieval_parameters`.

- [ ] **Step 3: Implement validation ablation and deterministic selection**

In `src/recagent_eval/tuning.py`, add
`build_retrieval_ablation(movies, split, depths=(100, 200, 500),
history_caps=(10, 20, 50), max_users=500)`. For each combination:

1. fit ItemCF and TF-IDF only on `split.train`;
2. build each user's state from positive training history;
3. call `build_semantic_profile("", state, movies, history_cap=cap)`;
4. retrieve both routes at `depth`;
5. rank with the current frozen `(0.7, 0.3, 0.0)` weights;
6. measure elapsed retrieval/ranking time for reporting;
7. aggregate validation target candidate recall and NDCG@10.

Return rows with these exact keys:

```python
{
    "retrieval_top_k": depth,
    "semantic_profile_history_cap": cap,
    "itemcf_candidate_recall": itemcf_hits / users,
    "semantic_candidate_recall": semantic_hits / users,
    "union_candidate_recall": union_hits / users,
    "ndcg_at_10": sum(ndcgs) / users,
    "latency_ms_per_user": elapsed_ms / users,
    "users": users,
}
```

Implement deterministic selection:

```python
def select_retrieval_parameters(
    movies: dict[int, Movie],
    split: DatasetSplit,
) -> dict[str, float | int | str]:
    rows = build_retrieval_ablation(movies, split)
    best = max(
        rows,
        key=lambda row: (
            row["ndcg_at_10"],
            row["union_candidate_recall"],
            -row["retrieval_top_k"],
            -row["semantic_profile_history_cap"],
        ),
    )
    return {
        **best,
        "selection_metric": "validation_ndcg_at_10",
    }
```

- [ ] **Step 4: Add a CLI command that writes both evidence and a frozen config**

Add `select-retrieval` to `src/recagent_eval/cli.py` with options for data
directory, evidence output, and config output. Construct the generated config
from the selection with:

```python
frozen_config = {
    "name": "structured-memory-hybrid-constraint-aware",
    "seed": 42,
    "retrieval_top_k": int(selection["retrieval_top_k"]),
    "semantic_profile_history_cap": int(
        selection["semantic_profile_history_cap"]
    ),
    "enable_memory": True,
    "enable_semantic_retrieval": True,
    "structured_planning": True,
    "required_retrieval_tools": [
        "itemcf_retrieve",
        "semantic_retrieve",
    ],
    "weights": [0.7, 0.3, 0.0],
}
```

Write it with `yaml.safe_dump(frozen_config, sort_keys=False)`.

Also extend `tune_on_validation` and `build_validation_examples` with
`retrieval_top_k` and `semantic_profile_history_cap` keyword arguments. Use
`build_semantic_profile` and the selected candidate depth when constructing
weight-tuning examples.

Extend the existing `tune` CLI with `--config` and `--config-output`. Load the
frozen retrieval parameters, tune weights with those parameters, replace the
config's `weights` list, and save the updated YAML to `--config-output`. Add a
CLI test asserting the saved weights contain three floats summing to `1.0`.

- [ ] **Step 5: Run tuning and CLI tests**

Run:

```bash
.venv/bin/pytest tests/test_tuning.py tests/test_retrieval_selection.py \
  tests/test_cli.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/recagent_eval/tuning.py src/recagent_eval/cli.py \
  tests/test_retrieval_selection.py tests/test_tuning.py tests/test_cli.py
git commit -m "feat: select retrieval parameters on validation data"
```

### Task 6: Regenerate cases, freeze retrieval, and retune weights

**Files:**
- Modify: `cases/fixed_cases.json`
- Modify: `cases/multi_turn_cases.json`
- Modify: `cases/stability_cases.json`
- Modify: `cases/qwen_smoke_cases.json`
- Create: `artifacts/retrieval_ablation.json`
- Create: `configs/full_constraint_aware.yaml`
- Modify: `configs/full_constraint_aware.yaml` after validation-only weight tuning

- [ ] **Step 1: Regenerate the fixed 40+10 case matrix**

Run:

```bash
.venv/bin/recagent-eval prepare-cases \
  --data-dir data/raw/ml-1m \
  --output cases/fixed_cases.json \
  --single-turn-count 40 \
  --multi-turn-count 10 \
  --seed 42
```

Expected: `Wrote 50 fixed cases to cases/fixed_cases.json`.

- [ ] **Step 2: Regenerate derived subsets from the new fixed matrix**

Run:

```bash
.venv/bin/recagent-eval subset-cases \
  --source cases/fixed_cases.json \
  --output cases/stability_cases.json \
  --single-turn-count 16 \
  --multi-turn-count 4
.venv/bin/recagent-eval subset-cases \
  --source cases/fixed_cases.json \
  --output cases/qwen_smoke_cases.json \
  --single-turn-count 8 \
  --multi-turn-count 2
.venv/bin/recagent-eval subset-cases \
  --source cases/fixed_cases.json \
  --output cases/multi_turn_cases.json \
  --single-turn-count 0 \
  --multi-turn-count 10
```

Expected: outputs contain 20, 10, and 10 cases respectively.

- [ ] **Step 3: Run validation-only retrieval selection**

Run:

```bash
.venv/bin/recagent-eval select-retrieval \
  --data-dir data/raw/ml-1m \
  --evidence-output artifacts/retrieval_ablation.json \
  --config-output configs/full_constraint_aware.yaml
```

Expected: the command prints the selected depth and cap; the JSON contains all
nine depth/cap combinations and the YAML contains the selected integers.

- [ ] **Step 4: Retune hybrid weights using the frozen retrieval parameters**

Extend the existing `tune` command in Task 5 to accept
`--config configs/full_constraint_aware.yaml`, then run:

```bash
.venv/bin/recagent-eval tune \
  --data-dir data/raw/ml-1m \
  --config configs/full_constraint_aware.yaml \
  --config-output configs/full_constraint_aware.yaml \
  --output artifacts/tuned_weights_constraint_aware.json \
  --step 0.1
```

Expected: output reports three weights summing to `1.0`, and the command writes
those same validation-selected weights into `configs/full_constraint_aware.yaml`.

- [ ] **Step 5: Run an offline end-to-end preflight twice**

Run:

```bash
.venv/bin/recagent-eval evaluate \
  --config configs/full_constraint_aware.yaml \
  --cases cases/fixed_cases.json \
  --provider rule-based \
  --output artifacts/runs/constraint-aware-offline-a
.venv/bin/recagent-eval evaluate \
  --config configs/full_constraint_aware.yaml \
  --cases cases/fixed_cases.json \
  --provider rule-based \
  --output artifacts/runs/constraint-aware-offline-b
```

Expected: eligibility is `1.0`, pipeline compliance is `1.0`, and both runs have
identical recommendation IDs after ignoring timestamps and latency.

- [ ] **Step 6: Commit frozen cases and validation evidence**

```bash
git add cases artifacts/retrieval_ablation.json \
  artifacts/tuned_weights_constraint_aware.json \
  configs/full_constraint_aware.yaml
git commit -m "data: freeze constraint-aware evaluation matrix"
```

### Task 7: Verify the implementation and run the new formal evaluation

**Files:**
- Modify: `README.md`
- Create: `reports/experiments/deepseek-constraint-aware.md`

- [ ] **Step 1: Run formatting, tests, and coverage**

Run:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
.venv/bin/pytest --cov=recagent_eval --cov-report=term-missing -q
```

Expected: Ruff has no findings, all tests pass, and coverage does not fall below
the existing 87% baseline.

- [ ] **Step 2: Run the three DeepSeek variants into new directories**

With `DEEPSEEK_API_KEY` present only in the environment, run:

```bash
.venv/bin/recagent-eval evaluate \
  --config configs/baseline.yaml \
  --cases cases/fixed_cases.json \
  --provider deepseek \
  --output artifacts/runs/baseline-deepseek-constraint-aware
.venv/bin/recagent-eval evaluate \
  --config configs/structured_memory.yaml \
  --cases cases/fixed_cases.json \
  --provider deepseek \
  --output artifacts/runs/structured-deepseek-constraint-aware
.venv/bin/recagent-eval evaluate \
  --config configs/full_constraint_aware.yaml \
  --cases cases/fixed_cases.json \
  --provider deepseek \
  --output artifacts/runs/full-deepseek-constraint-aware
```

Expected: each directory contains `episodes.jsonl`, `metrics.json`, and
`run_manifest.json`; no file contains the API key.

- [ ] **Step 3: Run the stratified stability subset**

Run:

```bash
.venv/bin/recagent-eval evaluate \
  --config configs/full_constraint_aware.yaml \
  --cases cases/stability_cases.json \
  --provider deepseek \
  --output artifacts/runs/full-deepseek-constraint-aware-stability
```

Expected: 20 episodes use the same frozen retrieval configuration and report a
separate case fingerprint.

- [ ] **Step 4: Write the formal report without overwriting historical results**

Create `reports/experiments/deepseek-constraint-aware.md` with:

- the new case fingerprint and frozen retrieval configuration;
- the Top-100/200/500 validation ablation table from
  `artifacts/retrieval_ablation.json`;
- baseline, structured-memory, and full-hybrid Recall@10, NDCG@10, HitRate@10;
- label eligibility, pipeline compliance, plan validity, tool success, hard
  constraint satisfaction, excluded-item violations;
- p50/p95 latency, calls, tokens, fallback, and failure rates;
- stability-subset results;
- a comparison to the archived diagnostic report with an explicit warning that
  the case fingerprints differ;
- an honest conclusion if full-hybrid NDCG@10 does not exceed ItemCF.

- [ ] **Step 5: Update README reproduction commands and evidence links**

Add the `select-retrieval`, constraint-aware tuning, offline preflight, and new
formal evaluation commands to `README.md`. Link both the archived
`deepseek-formal.md` and new `deepseek-constraint-aware.md` reports.

- [ ] **Step 6: Verify secrets and repository state**

Run:

```bash
rg -n "DEEPSEEK_API_KEY=|sk-[A-Za-z0-9_-]{16,}" . \
  --glob '!artifacts/runs/**' \
  --glob '!.git/**'
git diff --check
git status --short
```

Expected: secret scan has no matches, diff check is clean, and status lists only
the intended README/report changes.

- [ ] **Step 7: Commit documentation and final evidence**

```bash
git add README.md reports/experiments/deepseek-constraint-aware.md
git commit -m "docs: report constraint-aware DeepSeek evaluation"
```

- [ ] **Step 8: Run final verification from the committed tree**

Run:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
git status --short
```

Expected: lint and tests pass, and the worktree is clean.

# Offline Ranker Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic RRF and percentile-calibrated fusion, select them using validation NDCG@10, and prevent frozen-case evaluation unless a new ranker strictly beats same-depth ItemCF.

**Architecture:** Keep retrieval and Agent planning unchanged. Extend the existing `HybridRanker` with explicit strategy dispatch, build raw-score validation examples once, evaluate every ranker on those identical inputs, and persist a fingerprinted selection-evidence document. A separate offline case evaluator validates that evidence before reading the frozen cases; existing YAML files without a nested `ranker` block retain the current min-max behavior.

**Tech Stack:** Python 3.11, dataclasses, Pydantic models already in the project, Typer, PyYAML, NumPy-free ranking helpers, pytest, Ruff.

---

## File Map

- Modify `src/recagent_eval/ranking.py`: ranker kinds, score validation, RRF,
  percentile calibration, and strategy dispatch.
- Modify `src/recagent_eval/config.py`: nested ranker YAML validation with
  backward compatibility.
- Modify `src/recagent_eval/runner.py`: carry ranker settings into the Agent and
  run manifest.
- Create `src/recagent_eval/ranker_selection.py`: build identical validation
  examples, compute ablations, select a winner, fingerprint evidence, enforce
  the frozen-test gate, and evaluate frozen cases without an LLM.
- Modify `src/recagent_eval/cli.py`: add `select-ranker` and
  `evaluate-ranker` commands.
- Modify `tests/test_ranking.py`: unit tests for score fusion and failure cases.
- Create `tests/test_config.py`: nested/legacy configuration tests.
- Create `tests/test_ranker_selection.py`: selector, fingerprint, gate, and
  offline frozen-case tests.
- Modify `tests/test_cli.py`: command artifact and refusal-path tests.
- Create `configs/full_ranker_selected.yaml`: selected configuration only when
  the validation gate opens; otherwise write a checked-in rejected-candidate
  record instead.
- Create `artifacts/ranker_ablation.json`: complete real-data validation
  evidence.
- Create `reports/experiments/offline-ranker-selection.md`: result table and
  honest conclusion.
- Modify `README.md`: reproduction commands and current ranking status.

### Task 1: Add ranker configuration without breaking formal runs

**Files:**
- Modify: `src/recagent_eval/ranking.py`
- Modify: `src/recagent_eval/runner.py`
- Modify: `src/recagent_eval/config.py`
- Create: `tests/test_config.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing legacy and nested configuration tests**

Add:

```python
from pathlib import Path

import pytest

from recagent_eval.config import load_experiment_config


def test_legacy_weights_keep_minmax_linear_behavior(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text("name: legacy\nweights: [0.7, 0.3, 0.0]\n")

    config = load_experiment_config(path)

    assert config.ranker_kind == "minmax_linear"
    assert config.weights == (0.7, 0.3, 0.0)
    assert config.rrf_k == 60


def test_nested_rrf_config_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "rrf.yaml"
    path.write_text("name: rrf\nranker:\n  kind: rrf\n  rrf_k: 30\n")

    config = load_experiment_config(path)

    assert config.ranker_kind == "rrf"
    assert config.rrf_k == 30


@pytest.mark.parametrize(
    "ranker_yaml, message",
    [
        ("kind: unknown", "ranker.kind"),
        ("kind: rrf\n  rrf_k: 0", "rrf_k"),
        ("kind: percentile_linear\n  weights: [0.2, 0.2]", "sum to 1"),
    ],
)
def test_invalid_nested_ranker_is_rejected(
    tmp_path: Path,
    ranker_yaml: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(f"name: bad\nranker:\n  {ranker_yaml}\n")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)
```

Extend the runner manifest test to assert:

```python
assert manifest["ranker"] == {
    "kind": "minmax_linear",
    "rrf_k": 60,
    "weights": [0.5, 0.3, 0.2],
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_config.py tests/test_runner.py -q
```

Expected: failures because `ExperimentConfig` has no `ranker_kind` or `rrf_k`
and the manifest has no `ranker` object.

- [ ] **Step 3: Add validated ranker fields**

Define the shared type in `ranking.py`:

```python
RankerKind = Literal["itemcf", "minmax_linear", "rrf", "percentile_linear"]
```

Import it into `runner.py`, then extend `ExperimentConfig`:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    ranker_kind: RankerKind = "minmax_linear"
    rrf_k: int = 60
    retrieval_top_k: int = 100
    enable_memory: bool = True
    enable_semantic_retrieval: bool = True
    structured_planning: bool = True
    required_retrieval_tools: tuple[ToolName, ...] = ("itemcf_retrieve",)
    semantic_profile_history_cap: int = 20
    seed: int = 42
```

In `load_experiment_config`, parse `payload.get("ranker") or {}`. Validate:

```python
ranker_kind = str(ranker_payload.get("kind", "minmax_linear"))
if ranker_kind not in {"itemcf", "minmax_linear", "rrf", "percentile_linear"}:
    raise ValueError("ranker.kind must be itemcf, minmax_linear, rrf, or percentile_linear")
rrf_k = int(ranker_payload.get("rrf_k", 60))
if rrf_k <= 0:
    raise ValueError("ranker.rrf_k must be positive")
if "weights" in ranker_payload:
    route_weights = tuple(float(value) for value in ranker_payload["weights"])
    if len(route_weights) != 2 or not math.isclose(sum(route_weights), 1.0, abs_tol=1e-8):
        raise ValueError("ranker.weights must contain two values that sum to 1")
    weights = (route_weights[0], route_weights[1], 0.0)
```

Pass `ranker_kind` and `rrf_k` into `ExperimentConfig`, and add the nested
ranker object to `run_manifest.json`.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_config.py tests/test_runner.py -q
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest -q
```

Expected: all tests pass; the existing formal YAML files load as
`minmax_linear`.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/ranking.py src/recagent_eval/config.py \
  src/recagent_eval/runner.py tests/test_config.py tests/test_runner.py
git commit -m "feat: add backward-compatible ranker configuration"
```

### Task 2: Implement deterministic rank fusion

**Files:**
- Modify: `src/recagent_eval/ranking.py`
- Modify: `src/recagent_eval/runner.py`
- Modify: `tests/test_ranking.py`

- [ ] **Step 1: Write failing RRF and percentile tests**

Add tests:

```python
import math

import pytest

from recagent_eval.ranking import (
    HybridRanker,
    percentile_scores,
    reciprocal_rank_scores,
)


def test_rrf_sums_route_rank_contributions() -> None:
    itemcf = reciprocal_rank_scores({1: 9.0, 2: 8.0}, k=10)
    semantic = reciprocal_rank_scores({2: 0.9, 3: 0.8}, k=10)

    assert itemcf == {1: 1 / 11, 2: 1 / 12}
    assert semantic == {2: 1 / 11, 3: 1 / 12}


def test_percentiles_handle_empty_singleton_and_ties() -> None:
    assert percentile_scores({}) == {}
    assert percentile_scores({7: 2.0}) == {7: 1.0}
    assert percentile_scores({1: 5.0, 2: 5.0, 3: 1.0}) == {
        1: 1.0,
        2: 1.0,
        3: 0.0,
    }


def test_rrf_ranker_promotes_cross_route_support() -> None:
    ranked = HybridRanker(kind="rrf", rrf_k=10).rank(
        MOVIES,
        itemcf_scores={1: 10.0, 2: 9.0, 3: 8.0},
        semantic_scores={3: 1.0},
        state=PreferenceState(),
        top_k=3,
    )

    assert ranked[0].movie_id == 3


def test_itemcf_ranker_does_not_promote_semantic_only_candidates() -> None:
    ranked = HybridRanker(kind="itemcf").rank(
        MOVIES,
        itemcf_scores={2: 1.0},
        semantic_scores={1: 10.0},
        state=PreferenceState(),
        top_k=3,
    )

    assert [item.movie_id for item in ranked] == [2]


def test_ranker_rejects_non_finite_route_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        HybridRanker(kind="rrf").rank(
            MOVIES,
            itemcf_scores={1: math.nan},
            semantic_scores={},
            state=PreferenceState(),
        )
```

- [ ] **Step 2: Run ranking tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranking.py -q
```

Expected: import/signature failures for the new helpers and `kind`.

- [ ] **Step 3: Implement score transforms and dispatch**

Add:

```python
def _ordered_ids(scores: dict[int, float]) -> list[int]:
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("ranker scores must be finite")
    return sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))


def reciprocal_rank_scores(
    scores: dict[int, float],
    *,
    k: int,
) -> dict[int, float]:
    if k <= 0:
        raise ValueError("rrf k must be positive")
    return {
        movie_id: 1.0 / (k + rank)
        for rank, movie_id in enumerate(_ordered_ids(scores), start=1)
    }


def percentile_scores(scores: dict[int, float]) -> dict[int, float]:
    ordered = _ordered_ids(scores)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    first_index_by_score: dict[float, int] = {}
    result: dict[int, float] = {}
    for index, movie_id in enumerate(ordered):
        score = scores[movie_id]
        first_index_by_score.setdefault(score, index)
        result[movie_id] = 1.0 - first_index_by_score[score] / (len(ordered) - 1)
    return result
```

Extend `HybridRanker`:

```python
@dataclass(frozen=True)
class HybridRanker:
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    kind: RankerKind = "minmax_linear"
    rrf_k: int = 60
```

Inside `rank`, validate both raw maps and choose route contributions:

```python
if self.kind == "itemcf":
    candidate_ids = set(itemcf_scores)
    cf_contrib = dict(itemcf_scores)
    semantic_contrib = {}
elif self.kind == "rrf":
    candidate_ids = set(itemcf_scores) | set(semantic_scores)
    cf_contrib = reciprocal_rank_scores(itemcf_scores, k=self.rrf_k)
    semantic_contrib = reciprocal_rank_scores(semantic_scores, k=self.rrf_k)
elif self.kind == "percentile_linear":
    candidate_ids = set(itemcf_scores) | set(semantic_scores)
    cf_contrib = percentile_scores(itemcf_scores)
    semantic_contrib = percentile_scores(semantic_scores)
else:
    candidate_ids = set(itemcf_scores) | set(semantic_scores)
    cf_contrib = normalize_scores(itemcf_scores)
    semantic_contrib = normalize_scores(semantic_scores)
```

For RRF, set `final = cf_contrib.get(id, 0) +
semantic_contrib.get(id, 0)`. For other kinds, retain weighted scoring. Keep
movie ID as the deterministic final tie-break.

Construct the runner ranker with:

```python
HybridRanker(
    config.weights,
    kind=config.ranker_kind,
    rrf_k=config.rrf_k,
)
```

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranking.py tests/test_agent.py tests/test_runner.py -q
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest -q
```

Expected: all checks pass and legacy rank order remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/ranking.py src/recagent_eval/runner.py \
  tests/test_ranking.py
git commit -m "feat: add deterministic RRF and percentile fusion"
```

### Task 3: Build validation ablation and strict selection gate

**Files:**
- Create: `src/recagent_eval/ranker_selection.py`
- Create: `tests/test_ranker_selection.py`

- [ ] **Step 1: Write failing selector and gate tests**

Create tests using hand-built rows:

```python
import pytest

from recagent_eval.ranker_selection import (
    RankerSelectionEvidence,
    select_ranker,
    validate_test_gate,
)


def _row(kind: str, ndcg: float, recall: float = 0.1, **parameters):
    return {
        "kind": kind,
        "parameters": parameters,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "hit_rate_at_10": recall,
        "users": 10,
    }


def test_tie_with_itemcf_never_unlocks_test() -> None:
    evidence = select_ranker(
        [
            _row("itemcf", 0.2),
            _row("rrf", 0.2, rrf_k=30),
        ],
        dataset_fingerprint="abc",
        retrieval_top_k=500,
        history_cap=50,
        max_users=10,
    )

    assert evidence.selected["kind"] == "itemcf"
    assert evidence.test_unlocked is False
    assert evidence.margin == 0.0


def test_strict_improvement_unlocks_exact_selected_ranker() -> None:
    evidence = select_ranker(
        [
            _row("itemcf", 0.2),
            _row("rrf", 0.21, rrf_k=30),
            _row("percentile_linear", 0.205, weights=[0.8, 0.2]),
        ],
        dataset_fingerprint="abc",
        retrieval_top_k=500,
        history_cap=50,
        max_users=10,
    )

    assert evidence.selected["kind"] == "rrf"
    assert evidence.test_unlocked is True
    assert evidence.margin == pytest.approx(0.01)


def test_gate_reports_all_evidence_mismatches() -> None:
    evidence = RankerSelectionEvidence.model_validate({
        "rows": [_row("itemcf", 0.2), _row("rrf", 0.21, rrf_k=30)],
        "selected": _row("rrf", 0.21, rrf_k=30),
        "itemcf_ndcg_at_10": 0.2,
        "selected_ndcg_at_10": 0.21,
        "margin": 0.01,
        "test_unlocked": True,
        "dataset_fingerprint": "abc",
        "retrieval_top_k": 500,
        "semantic_profile_history_cap": 50,
        "max_users": 10,
    })

    with pytest.raises(ValueError, match="dataset_fingerprint.*retrieval_top_k"):
        validate_test_gate(
            evidence,
            dataset_fingerprint="different",
            retrieval_top_k=200,
            semantic_profile_history_cap=50,
            ranker_kind="rrf",
            ranker_parameters={"rrf_k": 30},
        )
```

- [ ] **Step 2: Run selector tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranker_selection.py -q
```

Expected: import failure because `ranker_selection.py` does not exist.

- [ ] **Step 3: Implement evidence schema, selection, and gate**

Use a Pydantic evidence model:

```python
class RankerSelectionEvidence(BaseModel):
    rows: list[dict[str, object]]
    selected: dict[str, object]
    itemcf_ndcg_at_10: float
    selected_ndcg_at_10: float
    margin: float
    test_unlocked: bool
    dataset_fingerprint: str
    retrieval_top_k: int
    semantic_profile_history_cap: int
    max_users: int
```

Implement a fixed method priority:

```python
METHOD_PRIORITY = {
    "itemcf": 0,
    "rrf": 1,
    "percentile_linear": 2,
}
```

Retain the min-max row in `rows`, but exclude it from the selectable set:

```python
eligible = [
    row
    for row in rows
    if str(row["kind"]) in {"itemcf", "rrf", "percentile_linear"}
]
best = max(
    eligible,
    key=lambda row: (
        float(row["ndcg_at_10"]),
        float(row["recall_at_10"]),
        -METHOD_PRIORITY[str(row["kind"])],
        json.dumps(row["parameters"], sort_keys=True),
    ),
)
margin = float(best["ndcg_at_10"]) - itemcf_ndcg
unlocked = (
    str(best["kind"]) in {"rrf", "percentile_linear"}
    and margin > 1e-12
)
```

`validate_test_gate` first rejects `test_unlocked=False`, then compares dataset
fingerprint, depth, history cap, selected kind, and selected parameters. Collect
all differing field names and values into one `ValueError`.

- [ ] **Step 4: Write a failing real-example ablation test**

Add a tiny two-user MovieLens-shaped fixture and assert that
`build_ranker_ablation` returns exactly:

```python
assert [row["kind"] for row in rows] == [
    "itemcf",
    "minmax_linear",
    "rrf",
    "rrf",
    "rrf",
    "rrf",
    *["percentile_linear"] * 11,
]
assert all(row["users"] == 1 for row in rows)
assert all(0.0 <= row["ndcg_at_10"] <= 1.0 for row in rows)
assert fingerprint == second_fingerprint
```

Call with `rrf_ks=(10, 30, 60, 100)`, `weight_step=0.1`, `max_users=1`,
`retrieval_top_k=2`, and `history_cap=1`.

- [ ] **Step 5: Run the new test and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranker_selection.py -q
```

Expected: failure because the ablation builder/fingerprint functions are
missing.

- [ ] **Step 6: Implement shared raw validation examples and ablation**

Create an internal frozen dataclass:

```python
@dataclass(frozen=True)
class RankerExample:
    itemcf_scores: dict[int, float]
    semantic_scores: dict[int, float]
    relevant_movie_ids: set[int]
```

Build each example from `split.train` history and
`split.validation_targets`, using `ItemCFRetriever`,
`TfidfSemanticRetriever`, `build_semantic_profile`, Top-K, and history cap.
Do not normalize scores in the example builder.

Evaluate:

- one ItemCF row;
- one existing min-max row with weights `(0.7, 0.3, 0.0)`;
- four RRF rows;
- eleven percentile rows `(1.0, 0.0)` through `(0.0, 1.0)`.

For each row, aggregate `recall_at_10`, `ndcg_at_10`, `hit_rate_at_10`,
ItemCF/semantic/union candidate recall, users, and measured
`latency_ms_per_user`. Selection must not use latency.

Fingerprint a canonical JSON object containing sorted movies, sorted training
ratings, sorted validation targets, `max_users`, Top-K, and history cap:

```python
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
return hashlib.sha256(canonical.encode()).hexdigest()
```

- [ ] **Step 7: Run focused and full verification**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranker_selection.py -q
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest -q
```

Expected: all checks pass.

- [ ] **Step 8: Commit**

```bash
git add src/recagent_eval/ranker_selection.py tests/test_ranker_selection.py
git commit -m "feat: add validation-only ranker selection gate"
```

### Task 4: Add the ranker-selection CLI

**Files:**
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write a failing CLI artifact test**

Monkeypatch the dataset loader and ablation builder, invoke:

```python
result = CliRunner().invoke(
    app,
    [
        "select-ranker",
        "--config",
        str(source_config),
        "--evidence-output",
        str(evidence_path),
        "--config-output",
        str(selected_config),
    ],
)
```

Assert:

```python
assert result.exit_code == 0, result.output
payload = json.loads(evidence_path.read_text())
assert payload["test_unlocked"] is True
selected = yaml.safe_load(selected_config.read_text())
assert selected["ranker"] == {"kind": "rrf", "rrf_k": 30}
assert selected["retrieval_top_k"] == 500
assert "Frozen test unlocked" in result.output
```

Add a second test where ItemCF wins and assert the evidence is written,
`config_output` does not exist, and output contains
`Frozen test remains locked`.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_cli.py -q
```

Expected: Typer reports no `select-ranker` command.

- [ ] **Step 3: Implement `select-ranker`**

Add a command with defaults:

```python
@app.command("select-ranker")
def select_ranker_command(
    config_path: Annotated[Path, typer.Option("--config")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    evidence_output: Annotated[Path, typer.Option()] = Path(
        "artifacts/ranker_ablation.json"
    ),
    config_output: Annotated[Path, typer.Option()] = Path(
        "configs/full_ranker_selected.yaml"
    ),
    max_users: int = 500,
) -> None:
```

Require the source config to have semantic retrieval enabled and both retrieval
tools required. Use its frozen depth and history cap. Build rows and
fingerprint, call `select_ranker`, and always write the evidence JSON.

Only when unlocked, copy the source YAML and replace/add:

```python
payload["ranker"] = selected_ranker_yaml(evidence.selected)
```

Do not overwrite the legacy top-level weights. They remain provenance for the
old min-max control but are ignored by RRF.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_cli.py -q
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest -q
```

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/recagent_eval/cli.py tests/test_cli.py
git commit -m "feat: add offline ranker selection command"
```

### Task 5: Enforce the gate for frozen-case evaluation

**Files:**
- Modify: `src/recagent_eval/ranker_selection.py`
- Modify: `src/recagent_eval/cli.py`
- Modify: `tests/test_ranker_selection.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing offline case-evaluation tests**

Build one single-turn and one multi-turn `EvaluationCase`. For the multi-turn
case, make `expected_preferences` exclude a genre that is absent from the
target. Assert that `evaluate_frozen_cases` uses
`case.expected_preferences or case.initial_state`, returns deterministic
Recall/NDCG/HitRate and candidate-stage metrics, and never constructs an
`LLMProvider`.

Add a CLI refusal test:

```python
result = CliRunner().invoke(
    app,
    [
        "evaluate-ranker",
        "--config",
        str(config_path),
        "--evidence",
        str(locked_evidence),
        "--cases",
        str(cases_path),
        "--output",
        str(output_path),
    ],
)

assert result.exit_code != 0
assert "frozen test is locked" in result.output.lower()
assert not output_path.exists()
```

Add an unlocked test that monkeypatches dataset loading and the offline
evaluator, then asserts `metrics.json` contains the selected kind and evidence
fingerprint.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranker_selection.py tests/test_cli.py -q
```

Expected: failures because `evaluate_frozen_cases` and `evaluate-ranker` do not
exist.

- [ ] **Step 3: Implement deterministic frozen-case evaluation**

For each case:

1. choose `state = case.expected_preferences or case.initial_state`;
2. hard-filter the movie catalog with that state;
3. use `state.liked_movie_ids` as ItemCF history;
4. build the semantic profile with the frozen history cap;
5. retrieve both routes at the frozen depth;
6. rank with the selected ranker;
7. aggregate top-10 and route-candidate metrics.

Return a JSON-serializable dictionary with:

```python
{
    "cases": len(cases),
    "ranker_kind": ranker.kind,
    "recall_at_10": ...,
    "ndcg_at_10": ...,
    "hit_rate_at_10": ...,
    "itemcf_candidate_recall": ...,
    "semantic_candidate_recall": ...,
    "union_candidate_recall": ...,
}
```

No prompt parsing, Provider, Agent, or LLM call is allowed in this function.

- [ ] **Step 4: Implement `evaluate-ranker` gate checks**

Add:

```python
@app.command("evaluate-ranker")
def evaluate_ranker(
    config_path: Annotated[Path, typer.Option("--config")],
    evidence_path: Annotated[Path, typer.Option("--evidence")],
    cases_path: Annotated[Path, typer.Option("--cases")],
    data_dir: Annotated[Path, typer.Option()] = Path("data/raw/ml-1m"),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/runs/offline-ranker-test/metrics.json"
    ),
) -> None:
```

Load evidence and config, recompute the validation-data fingerprint using the
recorded `max_users`, validate every gate field, then load fixed cases and call
the offline evaluator. Add `selection_evidence_fingerprint`, selected margin,
case fingerprint, and configuration fields to the output.

Create the output directory only after all gate checks pass.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest \
  tests/test_ranker_selection.py tests/test_cli.py -q
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest -q
```

Expected: all checks pass, and a locked gate leaves no test output.

- [ ] **Step 6: Commit**

```bash
git add src/recagent_eval/ranker_selection.py src/recagent_eval/cli.py \
  tests/test_ranker_selection.py tests/test_cli.py
git commit -m "feat: gate frozen-case ranker evaluation"
```

### Task 6: Run the real ablation and publish the evidence

**Files:**
- Create: `artifacts/ranker_ablation.json`
- Create only if unlocked: `configs/full_ranker_selected.yaml`
- Create: `reports/experiments/offline-ranker-selection.md`
- Modify: `README.md`

- [ ] **Step 1: Verify the pre-experiment repository**

Run:

```bash
git status --short
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest --cov=recagent_eval \
  --cov-report=term-missing
```

Expected: only planned artifact/report changes are present; all tests and lint
pass.

- [ ] **Step 2: Run validation-only ranker selection**

Run:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run recagent-eval \
  select-ranker \
  --config configs/full_constraint_aware.yaml \
  --data-dir data/raw/ml-1m \
  --evidence-output artifacts/ranker_ablation.json \
  --config-output configs/full_ranker_selected.yaml \
  --max-users 500
```

Expected: the command writes all 17 validation rows and explicitly prints
either `Frozen test unlocked` or `Frozen test remains locked`.

- [ ] **Step 3: Respect the gate**

If locked, verify:

```bash
test ! -e configs/full_ranker_selected.yaml
```

and do not invoke `evaluate-ranker`.

If unlocked, run exactly the selected configuration:

```bash
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run recagent-eval \
  evaluate-ranker \
  --config configs/full_ranker_selected.yaml \
  --evidence artifacts/ranker_ablation.json \
  --cases cases/fixed_cases.json \
  --data-dir data/raw/ml-1m \
  --output artifacts/runs/offline-ranker-test/metrics.json
```

Expected: gate validation succeeds and writes one frozen 50-case metric file.
Do not tune or select using this output.

- [ ] **Step 4: Write the result report**

Create a Markdown table containing every method/parameter validation row, the
ItemCF score, selected score, margin, and gate decision. If the gate opened,
add the single frozen-case row. If it remained locked, state that no test metric
was generated.

The conclusion must use one of these exact evidence shapes:

```text
RRF/percentile fusion improved validation NDCG@10 by <margin>; the frozen test
was evaluated once and <did/did not> improve.
```

or:

```text
Neither RRF nor percentile fusion strictly exceeded ItemCF on validation;
the frozen test remained locked and no DeepSeek rerun was justified.
```

- [ ] **Step 5: Update README reproduction and status**

Add:

```bash
uv run recagent-eval select-ranker \
  --config configs/full_constraint_aware.yaml \
  --evidence-output artifacts/ranker_ablation.json \
  --config-output configs/full_ranker_selected.yaml
```

Link the report and evidence. Do not change the checked-in DeepSeek metrics.
Mention `evaluate-ranker` only when the real evidence unlocks it.

- [ ] **Step 6: Run final verification**

Run:

```bash
rg -n "TBD|TODO|0\\.08|0\\.0418" \
  README.md reports/experiments/offline-ranker-selection.md \
  artifacts/ranker_ablation.json || true
git diff --check
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/recagent-eval-uv-cache uv run pytest --cov=recagent_eval \
  --cov-report=term-missing
git status --short
```

Expected: no stale metrics/placeholders, no whitespace errors, lint passes, all
tests pass, and only intended files are changed.

- [ ] **Step 7: Commit**

If locked:

```bash
git add artifacts/ranker_ablation.json \
  reports/experiments/offline-ranker-selection.md README.md
git commit -m "docs: report validation-locked ranker ablation"
```

If unlocked:

```bash
git add artifacts/ranker_ablation.json configs/full_ranker_selected.yaml \
  artifacts/runs/offline-ranker-test/metrics.json \
  reports/experiments/offline-ranker-selection.md README.md
git commit -m "feat: publish validation-selected offline ranker"
```

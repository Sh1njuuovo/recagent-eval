# Constraint-Aware Retrieval and Evaluation Design

Date: 2026-07-28
Status: approved direction, implementation pending

## Context

The first DeepSeek evaluation established that the agent layer can produce and
execute legal multi-turn plans, but it also exposed three confounded problems:

1. Four of ten multi-turn relevance targets violate the generated hard genre
   exclusion, so a constraint-correct recommender cannot receive credit.
2. Only one of the ten targets enters the allowed ItemCF Top-100, while none
   enters the TF-IDF Top-100.
3. All ten final DeepSeek plans omit `semantic_retrieve`. The experiment named
   "full hybrid" therefore does not consistently execute the configured hybrid
   pipeline.

On 500 validation users, ItemCF and TF-IDF union candidate coverage is 39.8% at
Top-100, 55.0% at Top-200, and 77.8% at Top-500. Ranking changes alone cannot
recover targets that are absent from the candidate union.

This design separates evaluation validity, execution policy, candidate recall,
and final ranking so improvements can be attributed correctly.

## Goals

- Guarantee that every relevance label is eligible under its case's hard
  constraints.
- Guarantee that each experimental variant executes the retrieval components
  declared by its configuration.
- Improve candidate coverage using validation-only decisions.
- Preserve deterministic non-LLM recommendation behavior under a fixed dataset,
  split, configuration, and seed.
- Keep old DeepSeek results as diagnostic evidence rather than overwriting them.
- Produce stage-level evidence suitable for a recommendation/LLM internship
  project: eligibility, candidate recall, ranking quality, plan reliability,
  latency, and cost.

## Non-goals

- Training a neural ranking model.
- Adding a second dataset or multi-agent architecture.
- Replacing DeepSeek as the formal planning provider.
- Claiming an improvement before the frozen test evaluation is complete.
- Silently selecting cases or hyperparameters based on test performance.

## Design

### 1. Constraint-consistent cases

Multi-turn case generation will select a negative genre that differs from the
liked genre and is absent from the held-out target movie. A user is eligible for
a generated multi-turn case only when such a genre exists.

A case validation function will derive the hard preference state represented by
the fixed case and check every relevant movie against:

- liked, disliked, and excluded movie IDs;
- required and excluded genres;
- minimum and maximum year.

Case preparation will fail with a case ID and exact violation reason if any
relevance target is ineligible. Evaluation will repeat this preflight check so a
manually edited case file cannot bypass the invariant.

Generated fixed cases receive a new fingerprint. Existing case files, manifests,
and reports remain unchanged as historical artifacts. New experiments must use a
new output directory and explicitly report the new fingerprint.

### 2. Configuration-owned execution policy

The LLM remains responsible for preference extraction and a typed tool plan, but
the experiment configuration defines the minimum retrieval policy:

| Variant | Required retrieval tools |
| --- | --- |
| Unstructured baseline | `itemcf_retrieve` |
| Structured memory | `itemcf_retrieve` |
| Full hybrid | `itemcf_retrieve`, `semantic_retrieve` |

`AgentConfig` will carry the required retrieval tools. Planning prompts will name
the required tools for the active variant. Parsed plans will undergo a
profile-aware validation after Pydantic schema validation:

- missing a required retrieval tool makes the plan invalid;
- the existing single repair attempt is allowed;
- a second failure uses the deterministic fallback plan for that profile.

The executor will not silently inject a missing tool into a plan declared valid.
This keeps plan-validity metrics honest. The full fallback includes both
retrievers, while ItemCF-only variants do not execute semantic retrieval.

Tool traces and run manifests will record the configured required tools. A new
pipeline-compliance metric reports the fraction of episodes whose final
recommendation turn executed every required retrieval tool in the required
order.

### 3. Content-profile semantic retrieval

The current semantic query mostly contains a generic final request plus genre
tokens. The revised query will build a deterministic user content profile from:

- the current user message;
- titles and genres of positively liked history movies;
- explicitly liked genres;
- negative preferences in the saved diagnostic context, but not as positive
  TF-IDF query tokens.

To avoid very long and popularity-biased profiles, history movies are ordered by
movie ID and capped by a configuration value. The default cap will be selected
on validation data. Hard exclusions remain the responsibility of `hard_filter`;
negative preferences remain available to hard filtering and the explicit
preference score. They are intentionally omitted from the unsigned TF-IDF query:
including text such as "not Action" would still increase the positive weight of
the token `Action`.

The TF-IDF index stays local and deterministic. The first optimization does not
introduce an external embedding model, GPU dependency, or model download. This
preserves the project's one-command CPU reproducibility.

### 4. Candidate-depth selection

Candidate depth becomes an explicit experiment parameter. The validation study
will compare Top-100, Top-200, and Top-500 for:

- ItemCF candidate recall;
- semantic candidate recall;
- union candidate recall;
- NDCG@10 after hybrid ranking;
- retrieval and total latency.

Selection uses validation NDCG@10, with union candidate recall as the first
tie-breaker and lower retrieval depth/profile cap as a deterministic latency
proxy. Measured latency is reported but does not control selection because
runtime noise would make an exact-metric tie irreproducible. The chosen depth
and profile-history cap are frozen before the test run.

Top-100 remains in the ablation table so the original design and the optimized
design are directly comparable. The project will not claim that increasing
candidate depth itself is a ranking innovation.

### 5. Ranking and weight tuning

The ranker keeps the interpretable three-part score:

```text
final = w_cf * normalized_itemcf
      + w_semantic * normalized_semantic
      + w_preference * explicit_preference_affinity
```

Weights continue to use a step-0.1 simplex grid search on validation users. The
candidate depth, semantic profile construction, and weights are tuned in a
fixed, documented order:

1. compare candidate depth and semantic profile cap using candidate coverage and
   fixed current weights;
2. freeze those retrieval parameters;
3. retune hybrid weights;
4. freeze the complete configuration for test evaluation.

No test target participates in retrieval-parameter or weight selection.

### 6. Stage-level diagnostics

Each evaluated episode will persist a compact candidate diagnostic for every
relevant target:

- eligible under final hard constraints;
- ItemCF rank or absent;
- semantic rank or absent;
- union membership;
- final hybrid rank or absent;
- executed retrieval tools.

Aggregate metrics will add:

- relevance-label eligibility rate;
- target eligibility under the agent's final preference state;
- ItemCF candidate Recall@K;
- semantic candidate Recall@K;
- union candidate Recall@K;
- pipeline compliance rate.

These metrics distinguish an invalid label, retrieval miss, ranking miss, and
planning-policy failure without logging private prompts or API credentials.

## Data Flow

```text
fixed case
  -> relevance eligibility preflight
  -> DeepSeek preference patch + typed plan
  -> profile-aware plan validation / one repair / deterministic fallback
  -> hard filter
  -> required ItemCF and/or semantic retrieval at frozen depth
  -> candidate union
  -> frozen hybrid reranking
  -> recommendations + tool traces + candidate diagnostics
  -> aggregate quality, reliability, constraint, latency, and cost metrics
```

## Error Handling

- Invalid case relevance: fail before any paid LLM call.
- Missing required retrieval tool: one repair, then profile-specific fallback.
- Empty allowed catalog: report a hard-constraint empty-candidate error; never
  violate excluded movies or genres.
- One empty retrieval route: continue with the other configured route and mark
  the empty route in diagnostics.
- Both retrieval routes empty: preserve the existing deterministic recovery
  behavior only if it respects all hard constraints; otherwise return no
  recommendations with an explicit error.
- Missing target metadata: fail case preflight with the target movie ID.

## Test Strategy

### Unit tests

- Negative genre selection never conflicts with the held-out target.
- Case validation reports movie-ID, genre, and year violations.
- Full-profile plan validation rejects an ItemCF-only plan.
- ItemCF-only profiles accept the same plan.
- Repair and fallback retain the configured retrieval policy.
- Content-profile construction is capped, ordered, and deterministic.
- Candidate diagnostics distinguish filtered, retrieval-missed, and
  ranking-missed targets.
- Pipeline compliance uses only the final recommendation turn.

### Integration tests

- A synthetic full-hybrid episode executes both retrieval tools and saves their
  traces.
- Evaluation aborts before provider invocation for an inconsistent case.
- A MovieLens sample run produces eligible cases, stage diagnostics, and
  deterministic recommendation IDs.

### Regression tests

- Hard exclusions always have zero violations.
- Existing API timeout, invalid JSON, unknown tool, and empty-candidate tests
  continue to pass.
- The unstructured and structured-memory variants remain ItemCF-only.
- Existing historical reports and run directories are not modified.

## Experiment and Acceptance Plan

1. Regenerate the 40 single-turn and 10 multi-turn fixed cases with the same
   seed and record the new fingerprint.
2. Run an offline rule-based preflight and retrieval-depth ablation.
3. Freeze semantic profile cap, candidate depth, and validation-selected weights.
4. Run the three DeepSeek variants on the same fixed cases at temperature zero.
5. Re-run the stratified 20-case stability subset.
6. Compare the new results with both the ItemCF baseline and the archived
   diagnostic run.

Acceptance criteria:

- relevance-label eligibility rate is 100%;
- full-hybrid pipeline compliance rate is at least 95%;
- plan legal rate remains at least 95%;
- tool success rate remains at least 98%;
- hard-constraint satisfaction is at least 95%;
- excluded movie violation rate remains zero;
- optimized union candidate recall exceeds Top-100 union candidate recall on
  validation;
- full-hybrid test NDCG@10 is reported honestly against ItemCF, whether it wins
  or loses;
- fixed-input non-LLM recommendations remain deterministically reproducible.

## Reporting

The experiment report will clearly separate:

- the original DeepSeek run that exposed invalid labels and policy drift;
- the offline retrieval ablation used for design selection;
- the new frozen-case formal comparison;
- any failed hypothesis, including semantic retrieval that does not improve
  final NDCG.

This separation turns the failure analysis into project evidence without
retroactively rewriting results.

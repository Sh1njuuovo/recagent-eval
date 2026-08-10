# RecAgent-Eval v2 Offline Ranking Design

## 1. Status and authority

This document specifies the validation-only v2 recommendation experiment for
RecAgent-Eval. It was approved section by section on 2026-08-10. The project
methodology in `docs/project-methodology.md` remains the higher-level authority.

The design preserves these boundaries:

- the LLM interprets preferences and plans tools, but does not rank items;
- hard constraints run before retrieval and ranking;
- recommendation quality is diagnosed at retrieval and Top-10 ranking stages;
- model selection uses validation data only;
- negative results remain first-class artifacts;
- every public claim must be traceable to code, configuration, fingerprints,
  tests, and experiment output.

No frozen test, DeepSeek call, or RTX 4090 run is authorized by this design.

## 2. Audited v1 baseline

The design starts from the following reverified repository state:

- Ruff passes.
- The test suite has 79 passing tests and 90% line coverage.
- The deterministic offline smoke command passes without a private key.
- MovieLens-1M contains 3,883 movies and 1,000,209 ratings.
- Per-user chronological splitting produces 988,139 training interactions and
  6,035 users with both validation and test targets.
- The checked-in fixed-case and ranker-dataset fingerprints match the current
  ranker ablation evidence.
- The validation-only ItemCF row has NDCG@10 `0.033388484023395205` on 500
  users.
- The TF-IDF route raises union candidate recall but RRF and genuine percentile
  fusion do not pass the ItemCF selection gate.
- `test_unlocked` is false, so the frozen test remains locked.

The existing TF-IDF route uses title and genre tokens. It is not a learned
sentence embedding, and v2 must continue to describe it accurately.

## 3. Problem and falsifiable hypotheses

The observed bottleneck is ranking calibration: the second retrieval route adds
held-out targets to the candidate union, but the deterministic fusion does not
reliably place them in the Top 10.

The v2 experiment tests two factors separately and together:

1. **Embedding hypothesis:** a real sentence embedding route retrieves useful
   targets that title/genre TF-IDF misses.
2. **Ranking hypothesis:** route scores, ranks, membership, popularity, history
   compatibility, and metadata features let a regularized ranker distinguish
   useful semantic candidates from noise.
3. **Combined hypothesis:** embedding retrieval plus learned calibration
   improves validation NDCG@10 over same-depth ItemCF without regressing Recall,
   constraints, reproducibility, or the resource budget.

The hypotheses fail if their corresponding metrics do not improve under the
pre-registered protocol. Candidate-recall improvement alone is not evidence of
Top-10 improvement.

## 4. Compared systems

All routes use Top-500 retrieval and the same eligible users, ordered histories,
hard-filter policy, metric code, and fold assignments.

| ID | Candidate routes | Final ranking | Role |
| --- | --- | --- | --- |
| ItemCF | ItemCF | raw ItemCF | same-depth gate baseline |
| v1 control | ItemCF + TF-IDF | fixed 0.7/0.3 min-max | historical deterministic control |
| A | ItemCF + BGE | fixed 0.7/0.3 min-max | embedding-only ablation |
| B | ItemCF + TF-IDF | pairwise linear ranker | reranker-only ablation |
| C | ItemCF + BGE | pairwise linear ranker | pre-registered primary system |

The v1 control is contextual evidence and cannot unlock another test run. Only
system C is eligible for the v2 frozen-test gate. Comparisons `C - B` and
`C - A` diagnose the embedding and learned-ranking contributions respectively.
If B improves but C fails, the frozen test remains locked; a reranker-only route
would require a new pre-registration cycle.

## 5. Embedding model and resource choice

The primary model is `BAAI/bge-small-en-v1.5`:

- 33.4 million parameters;
- 384-dimensional embeddings;
- maximum sequence length 512;
- approximately 133 MB for the safetensors weights;
- MIT license.

The model repository revision is resolved and pinned during the explicitly
authorized preparation step. Formal validation never uses a floating `main`
revision. The resolved revision, weight hash, license, library versions, pooling
configuration, normalization setting, and device are saved in the cache and run
manifests.

`sentence-transformers/all-MiniLM-L6-v2` is the documented CPU fallback option
because it is smaller, but it is not searched as part of this experiment.
`intfloat/e5-small-v2` is not selected because its query/passage prefix protocol
adds an unnecessary experimental degree of freedom for the symmetric
history-to-item similarity used here.

The formal search uses brute-force cosine similarity. With only 3,883 items,
FAISS or a vector database would add operational complexity without solving the
observed bottleneck.

Primary references:

- BGE model card: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- BGE weights and size: <https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/model.safetensors>
- MiniLM model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- E5 model card: <https://huggingface.co/intfloat/e5-small-v2>
- Sentence Transformers inference guidance:
  <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html>

## 6. Data and leakage boundaries

### 6.1 Outer MovieLens split

MovieLens interactions remain split chronologically per user:

- all interactions except the last two positive interactions form
  `split.train`;
- the penultimate positive item is the validation target;
- the latest positive item is the frozen test target.

Only `split.train` may fit ItemCF, popularity, user histories, profiles, item
compatibility features, scalers, and training examples. Validation targets may
be labels inside the appropriate nested-CV training folds and evaluation labels
inside held-out outer folds. Test targets may not enter any validation command,
feature, cache, hyperparameter, early-stopping decision, or report.

Movie metadata for every catalog item may be embedded because it is treated as
catalog information available before recommendation. This transductive catalog
assumption is recorded in the manifest and does not permit use of future user
interactions.

### 6.2 Ordered histories

Each user history contains only positive `split.train` rows with rating at least
4. Rows are ordered by `(timestamp, movie_id)`, and the most recent 50 are used.
This replaces the v1 behavior that sorted a set of liked movie IDs and therefore
did not implement a true recency cap.

### 6.3 Nested user-level validation

The formal experiment uses all 6,035 eligible validation users. A 100-user run
is permitted only as a schema and boundary smoke test and cannot contribute to
selection evidence.

Formal validation uses:

- outer folds: 5;
- repeated seeds: `42`, `2025`, and `3407`;
- total outer fold/seed cells: 15;
- inner folds: 3, using only the current outer-training users;
- logistic regularization grid: `C in {0.01, 0.1, 1.0}`.

User IDs are indivisible groups. Each seed deterministically assigns each user
to one outer fold, and each outer-training set receives a deterministic inner
assignment. Fold maps are generated once, saved, fingerprinted, and shared by
all systems. Inner selection maximizes mean NDCG@10, then Recall@10, then chooses
the smallest `C` to prefer stronger regularization.

Each user receives one held-out prediction per seed. Primary aggregate and
bootstrap calculations first average the three paired per-user outcomes, so a
user remains the statistical unit rather than being counted three times.

## 7. Item text, embedding cache, and user profiles

The versioned item-text template is:

```text
Title: {title_without_year}. Genres: {sorted_genres}. Release year: {year_or_unknown}.
```

Genres are sorted lexicographically. Unknown years use the literal `unknown`.
Whitespace, punctuation, title-year removal, and normalization rules are part
of the text-schema fingerprint.

Item embeddings are batch encoded, converted to float32, L2 normalized, ordered
by movie ID, and stored once. The user embedding profile is the L2-normalized
arithmetic mean of the most recent 50 positive-history item embeddings. The
primary experiment does not tune recency decay, rating weights, profile blend
weights, or embedding dimensions.

The TF-IDF route in B continues to use the existing title/genre retriever with
the same legal ordered history cap. It is not relabeled as an embedding model.

## 8. Candidate feature schema

Both learned systems use the same ordered, versioned feature names:

1. `itemcf_score_log1p`
2. `itemcf_reciprocal_rank`
3. `semantic_score`
4. `semantic_reciprocal_rank`
5. `in_itemcf`
6. `in_semantic`
7. `log1p_train_popularity`
8. `history_genre_jaccard`
9. `history_year_abs_gap`
10. `history_year_missing`
11. `max_history_semantic_similarity`
12. `mean_history_semantic_similarity`
13. `explicit_liked_genre_overlap`
14. `explicit_required_genres_match`
15. `explicit_year_range_match`
16. `explicit_preference_present`

Definitions are fixed as follows:

- reciprocal rank is `1 / rank`, with rank starting at 1;
- absent route score and reciprocal rank are zero, while route-membership fields
  distinguish absence from a genuine zero;
- popularity counts positive interactions in the legal training rows only;
- history genre Jaccard compares the candidate genres with the user's three most
  frequent training-history genres, using lexical order to break count ties;
- history year gap is the absolute difference from the mean known history year;
  the gap is zero and `history_year_missing` is one if either side is unknown;
- history semantic similarities compare a candidate with the individual legal
  history items under the active semantic representation;
- explicit fields are derived only from an externally supplied legal
  `PreferenceState`, never inferred from the held-out target.

MovieLens validation contains no authentic natural-language preference labels.
Therefore the explicit-preference fields are neutral in primary CV and must not
be claimed as learned effects. Their feature contract is implemented and tested
with synthetic states so a later conversational training dataset can activate
them without changing the schema. History-derived genre and year features are
described as history-derived, not explicit.

All finite-value checks run before model fitting. NaN or infinity reports the
user, movie, feature, and fold and fails the run. No undeclared imputation is
allowed.

## 9. Pairwise linear ranker

The ranker uses scikit-learn `StandardScaler` followed by L2-regularized
`LogisticRegression` with `fit_intercept=False`. The locked dependency version
is recorded by `uv.lock`; the project declares a bounded optional v2 dependency
rather than adding XGBoost, LightGBM, pandas, or FAISS.

For each training user whose validation target is present in the active union:

1. take the target as the positive candidate;
2. take the highest 20 non-target ItemCF candidates;
3. add the highest 20 non-target semantic candidates not already selected;
4. fill to at most 50 negatives from the remaining union, ordered by SHA-256 of
   `(seed, user_id, movie_id)`;
5. create both `positive - negative` with label 1 and
   `negative - positive` with label 0;
6. assign pair-row weights so the total weight contributed by each user is 1.

Users with a retrieval miss create no training pair but remain in every
evaluation denominator. Counts of eligible users, retrieval misses, generated
pairs, and negative sources are saved per fold.

The scaler is fit only on candidate features from the current training fold.
Pair differences are formed after that fold-local transformation. At inference,
the learned linear decision score ranks every candidate in the complete union;
evaluation never uses only the sampled negatives.

Global coefficients and per-candidate feature contributions are serialized for
interpretability. Tree rankers remain out of scope because they expand the
dependency and hyperparameter surface before the linear hypothesis is tested.

Primary ranker reference:

- scikit-learn LogisticRegression:
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- scikit-learn GroupKFold:
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>
- scikit-learn StandardScaler:
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>

## 10. Metrics, bootstrap, and subgroups

Every system reports:

- Recall@10;
- NDCG@10;
- HitRate@10;
- ItemCF candidate recall;
- semantic candidate recall;
- union candidate recall;
- excluded-seen-item violation rate;
- applicable hard-constraint satisfaction rate;
- latency and resource measurements.

Recall@10 and HitRate@10 are expected to be equal when each user has one target,
but both remain explicit for API and reporting consistency.

Paired uncertainty uses 10,000 user-level bootstrap resamples with seed
`20260810`. The report stores the full procedure, the 95% percentile interval,
absolute NDCG delta, relative lift, and paired win/tie/loss user proportions.
The bootstrap resamples users, not candidate rows or folds.

The following diagnostic subgroups are computed from training information and
do not participate in model selection:

- positive-history length quartiles;
- validation-target training popularity: unseen, nonzero 0-50th percentile,
  50-90th percentile, and 90-100th percentile;
- target route membership: ItemCF-only, semantic-only, both, or neither;
- training-history genre-diversity quartiles.

Every subgroup stores its user count and the full metric/delta set. Groups with
fewer than 100 users are labeled exploratory and cannot support a public
advantage claim.

## 11. Primary promotion gate

Only system C may pass the gate, and every condition is conjunctive:

1. repeated-CV OOF mean NDCG@10 is strictly greater than same-depth ItemCF;
2. the paired user-bootstrap 95% interval for `delta NDCG@10` has lower bound
   strictly greater than zero;
3. the aggregate NDCG delta is positive for each of the three seeds;
4. at least 10 of 15 outer fold/seed cells have positive NDCG delta;
5. Recall@10 and HitRate@10 point estimates are not lower than ItemCF;
6. union candidate recall is not lower than ItemCF candidate recall;
7. excluded-seen-item violations are zero and applicable hard constraints remain
   fully satisfied;
8. all data, fold, model, schema, metric, candidate-policy, and artifact
   fingerprints match the registered run;
9. the resource budget passes;
10. every expected row, fold, seed, subgroup, and failure record is present.

A positive mean with a confidence interval crossing zero is a failure. A
statistical pass with a resource, constraint, completeness, or fingerprint
failure is also a failure.

## 12. Resource budget

The CPU path is the required resource baseline:

- primary model weights: approximately 133 MB in the user model cache, never
  committed to Git;
- item embedding cache: expected around 6 MB;
- serialized final ranker: less than 5 MB;
- total run cache and intermediate artifacts: no more than 1 GB;
- peak resident memory: no more than 4 GB;
- item embedding preparation: no more than 30 minutes;
- complete 3-seed by 5-fold nested CV: no more than 4 hours;
- cache-hit single-user recommendation p95: no more than 100 ms, excluding first
  model load.

Resource overrun locks the frozen test. Engineering optimization may improve
batching or cache reuse, but may not remove users, folds, seeds, failed results,
or metrics to meet the budget.

## 13. Artifacts and fingerprints

An embedding cache has this contract:

```text
artifacts/v2/cache/<embedding-fingerprint>/
|-- item_embeddings.npz
`-- manifest.json
```

A validation run has this contract:

```text
artifacts/v2/validation/<run-id>/
|-- manifest.json
|-- feature_schema.json
|-- fold_assignments.json
|-- fold_metrics.jsonl
|-- user_metrics.jsonl
|-- ablations.json
|-- bootstrap.json
|-- subgroups.json
|-- resource_usage.json
|-- models/
|   `-- <seed>-<fold>.json
|-- report.md
`-- promotion_manifest.json  # present only after every gate passes
```

The manifest records at least:

- raw MovieLens file hashes and split fingerprint;
- eligible user and ordered-history fingerprints;
- model ID, resolved revision, weight hash, license, and embedding settings;
- item-text and feature-schema hashes;
- candidate depth, history cap, and route kind;
- fold maps, seeds, inner selections, and negative-sampling policy;
- estimator, scaler, and dependency versions;
- config hash, Git commit, Python, platform, dtype, and device;
- degradation, error, completeness, and gate status.

All machine-readable outputs use stable field names and deterministic ordering
where ordering is meaningful. Complete per-user outputs stay in the ignored run
directory; an aggregate JSON and Markdown report may be committed. Experiment
commands never delete or overwrite a previous run, including failed runs.

The final model, configuration, feature schema, dependency lock, and every
upstream fingerprint are copied into the promotion manifest. The promotion
manifest is generated only after outputs are durably written and the gate is
recomputed from those outputs.

## 14. Frozen-test isolation

`configs/frozen_test_lock.yaml` stores the already established canonical fixed
case fingerprint but no test metrics. Validation and promotion code neither
imports a frozen-case loader nor accepts a case path.

The protected frozen command executes in this order:

1. load and independently recompute the promotion evidence;
2. reject immediately if the promotion gate is not open;
3. validate model, feature, configuration, dependency, and data fingerprints;
4. check the canonical local frozen-test consumption marker;
5. only then load the frozen lock and fixed cases;
6. verify the case fingerprint;
7. mark the frozen evaluation as started;
8. run one offline evaluation and atomically mark it completed.

An existing started or completed consumption marker blocks another run. A crash
after the started marker does not authorize an automatic retry; recovery
requires explicit user review. This is a methodological guard rather than an
attempt to hide the checked-in cases from a repository reader.

The frozen command does not invoke DeepSeek. Any later DeepSeek matrix requires
separate user approval. Qwen/vLLM smoke remains a separately labeled experiment.

## 15. Failure behavior

- Missing model weights produce an actionable error that states model purpose,
  expected size, license, and preparation command.
- Cache, revision, text-schema, feature-schema, or model mismatch is fatal.
- Fold overlap, outer-label use during inner selection, or fold-local scaler
  leakage is fatal.
- A fold without trainable positive pairs is saved as a failed fold and fails
  the complete gate.
- Formal BGE failure never silently substitutes TF-IDF.
- NaN or infinity is fatal and records the exact context.
- Locked, partial, or inconsistent evidence cannot create a promotion manifest.
- Existing output paths are never overwritten.

Interpretation rules are also pre-registered:

- A raises candidate recall but C does not raise NDCG: retrieval expanded but
  ranking calibration still failed.
- B improves while C fails: the current embedding candidate noise exceeds its
  benefit; the primary gate remains locked.
- C improves in mean but its interval crosses zero: evidence is insufficient.
- all learned routes fail: retain the full ablation and stop this route.

No failure state authorizes a frozen test or a paid LLM matrix.

## 16. Test strategy and TDD order

Implementation follows test-driven development. Tests are written before each
corresponding behavior.

Required tests cover:

- ordered histories exclude validation and test targets;
- item text, model metadata, and embedding-cache fingerprints are stable;
- a fake embedder satisfies the contract so CI never downloads a real model;
- popularity, profiles, and scalers read only the legal current training fold;
- feature ordering, missing-route encoding, and finite-value validation;
- pair orientation, per-user weights, hard-negative order, and reproducibility;
- B and C share a schema but cannot share a mismatched route cache;
- outer and inner fold maps are deterministic and user-disjoint;
- inner selection cannot inspect outer labels;
- user-level bootstrap is reproducible;
- every individual gate failure keeps the test locked;
- a locked gate never calls the frozen-case loader;
- a started or completed consumption marker blocks a repeated frozen run;
- a small fixture completes A/B/C evaluation and writes every required JSON and
  Markdown artifact;
- all 79 existing tests remain passing.

CI uses fake embeddings and tiny fixtures. Real BGE preparation, 100-user smoke,
and full 6,035-user validation are explicit local experiment steps.

## 17. Expected file changes

New source files:

- `src/recagent_eval/embedding.py`
- `src/recagent_eval/candidate_features.py`
- `src/recagent_eval/learned_ranking.py`
- `src/recagent_eval/v2_selection.py`

New configuration and tests:

- `configs/v2_offline.yaml`
- `configs/frozen_test_lock.yaml`
- focused tests for each new module and the protected CLI flow.

Small modifications:

- `src/recagent_eval/data.py`: expose ordered legal user histories;
- `src/recagent_eval/config.py`: parse and validate v2 configuration;
- `src/recagent_eval/cli.py`: add `prepare-embeddings`, `validate-v2`, and the
  protected `evaluate-v2-frozen` command;
- `pyproject.toml` and `uv.lock`: add bounded optional v2 dependencies for
  Sentence Transformers/PyTorch and scikit-learn;
- `.gitignore`: ignore model/cache and full per-user run outputs while retaining
  selected aggregate evidence;
- `README.md`: before results, document only pending status and reproduction;
  after validation, publish the complete honest outcome table.

Generated evidence after validation:

- `reports/experiments/v2-validation.json`
- `reports/experiments/v2-validation.md`

The existing v1 ranker selection code, reports, and negative results are not
rewritten or removed.

## 18. Out of scope

This design does not include:

- a large neural ranking model or cross-encoder;
- embedding fine-tuning;
- a second recommendation dataset;
- multiple Agents;
- Demo redesign;
- DeepSeek API calls;
- remote RTX 4090 usage;
- Qwen/vLLM execution;
- frozen-test execution before an independently verified promotion manifest.

## 19. Acceptance criteria for implementation readiness

The design is ready for implementation planning when:

- the user approves this written spec;
- the spec contains no unresolved placeholder or hidden model-selection choice;
- `writing-plans` produces a per-file, per-test TDD implementation plan;
- dependency installation and model download remain separate, explicitly
  described actions with size, license, and cache behavior;
- implementation begins in an isolated branch or worktree without disturbing
  the current untracked handoff documents.

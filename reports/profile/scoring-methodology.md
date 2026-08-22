# Candidate ranking methodology and evidence

## Snapshot identity

- Generated at: 2026-08-20 (Asia/Shanghai).
- Taste: disabled. No `taste.json` or `--taste` argument was used.
- Normalization denominator: 104 points.
- Python: 3.13.11; the generator uses only the Python standard library.
- Candidate scoring implementation: repository-local
  `scripts/candidate_score.py`.
- Generator SHA-256:
  `edd9ccabdcb73770df92273b405ecffb28e4f737fe0637e31e0bc83cfba4c536`.

The required `--as-of 2026-08-20` argument freezes activity scoring. The result
does not depend on the machine's current date or on a user-local skill install.

## Inputs and outputs

| File | Role | SHA-256 |
| --- | --- | --- |
| `reports/profile/jd.txt` | active proxy JD | `2cceeba154e3d164312352b95245403afcc98e4acf038ac2eae5e27e8391d220` |
| `reports/profile/candidates.json` | explicit candidate facts and scores | `268c6c3706e08419e1e94ce80e33c5c8696e7b10244ab563d4e360a52fdbfd23` |
| `reports/ranking/candidate_score.json` | generated machine-readable ranking | `037b8aa403f807a87d65ef0117ca97d6c254e2872a2584fb75b87698566f7442` |
| `reports/ranking/candidate_score.md` | generated human-readable ranking | `04d8193109f2a33e35ecfa7c08dcb78520d6f4a8b8e4a0bcc855a6c47b482d6e` |

## Exact generation command

Run from any clean checkout's repository root:

```bash
.venv/bin/python scripts/candidate_score.py \
  --jd reports/profile/jd.txt \
  --candidates reports/profile/candidates.json \
  --out reports/ranking \
  --as-of 2026-08-20
```

The repo-local command has no `--taste` option and writes final JSON and
Markdown directly.

## Deterministic postprocessing

None. `scripts/candidate_score.py` omits Taste/preference fields, uses correct
singular/plural risk grammar, and renders every scored risk. A generated file
must not be edited after the command.

## Scoring fields

The generator reads explicit candidate fields rather than inferring scores from
free text:

| Component | Rule |
| --- | --- |
| JD match | `jd_match_score`, clamped to 0–30 |
| License | `license_score`, clamped to 0–4 |
| Runnable | `runnable_score`, clamped to -5–20 |
| Resource fit | `resource_fit_score`, clamped to 0–10 |
| Activity | 10 within 180 days, 7 within 365, 4 within 730, 1 otherwise, 0 if missing/unparseable |
| Stars | 10 at ≥1,000; 8 at ≥200; 6 at ≥50; 3 above zero; otherwise 0 |
| Modification space | 5 per `mod_ideas` entry, capped at 20 |
| Risk penalty | -3 per `risk_notes` entry, capped at -20 |

`raw_score` is the sum of those components. The no-taste score is
`round(clamp(raw_score, 0, 104) * 100 / 104, 2)`.

## Source evidence and score rationales

Community counts and last-commit dates are dated shortlist observations, not
live GitHub queries. They are retained so the snapshot can be reproduced; a
future ranking refresh must recheck the official repository pages and create a
new dated snapshot.

### RecAI/InteRecAgent

- Official source: <https://github.com/microsoft/RecAI/tree/main/InteRecAgent>.
- Checked-in audit: `reports/audit/audit.json`, which records the upstream
  README, `app.py`, `run.sh`, retrieval/query/ranking modules, evaluation files,
  legacy dependency surface, and prepared-resource signals.
- JD 30/30: the audited README and module tree directly cover LLM Agent,
  query, retrieval, ranking, recommendation, and evaluation.
- License 4/4: the audited shortlist recorded MIT for the official project.
- Runnable 17/20: README plus application/run entry points are present; legacy
  dependencies and external prepared resources prevent a full-run claim.
- Resource fit 9/10: the chosen project is a bounded independent refactor path;
  upstream prepared resources still require validation.
- Activity 7/10 and stars 10/10: the dated snapshot records 2026-01-27 and
  1,200 stars.
- Modification 20/20: four concrete changes are recorded.
- Risk -6: two separately scored risks are present.
- Arithmetic: `30 + 4 + 17 + 9 + 7 + 10 + 20 - 6 = 91`.

### RecBole

- Official source: <https://github.com/RUCAIBox/RecBole>.
- Official documentation: <https://recbole.io/docs/>.
- JD 24/30: it directly covers recommendation, ranking, datasets, and
  evaluation, while native LLM-Agent integration is absent from the audited
  shortlist.
- License 3/4: the shortlist recorded MIT with an academic-use notice, so it
  does not receive the unqualified-license maximum.
- Runnable 18/20: official documentation provides a mature baseline path; this
  task did not repeat execution, so the score stays below the maximum.
- Resource fit 10/10: the audited shortlist identified a bounded
  CPU/single-GPU baseline path.
- Activity 4/10 and stars 10/10: the dated snapshot records 2025-02-23 and
  4,500 stars.
- Modification 20/20: four concrete integration/evaluation changes are
  recorded.
- Risk -3: weak native Agent integration is the single scored risk.
- Arithmetic: `24 + 3 + 18 + 10 + 4 + 10 + 20 - 3 = 86`.

### OpenOneRec

- Official source: <https://github.com/Kuaishou-OneRec/OpenOneRec>.
- JD 25/30: generative recommendation, LLM, and ranking are direct matches;
  the audited shortlist did not establish the same bounded interactive-Agent
  path as InteRecAgent.
- License 4/4: the official repository currently declares Apache-2.0 in
  <https://github.com/Kuaishou-OneRec/OpenOneRec/blob/main/LICENSE>, an
  unambiguous OSI-approved license scored consistently with RecAI's MIT license.
- Runnable 4/20: the recorded path centers on distributed pretraining and a
  100M-interaction dataset, outside the two-week bounded full-run path.
- Resource fit 1/10: the same distributed/data requirements conflict with the
  approved local CPU plus single-4090 scope.
- Activity 10/10 and stars 8/10: the dated snapshot records 2026-05-18 and 885
  stars.
- Modification 10/20: two bounded extraction/smoke ideas are recorded.
- Risk -9: distributed pretraining, the 100M-interaction dataset, and incomplete
  documentation are three separately scored risks.
- Arithmetic: `25 + 4 + 4 + 1 + 10 + 8 + 10 - 9 = 53`.

## Reproduction acceptance

A reproduction is accepted only if:

- input and repository-local generator hashes match this file;
- the generator command is run with `--as-of 2026-08-20`;
- no postprocessing is applied;
- output hashes match this file;
- each JSON candidate has `max_raw_score: 104` and no Taste fields or
  `user_preference`; and
- the Markdown risk cell lists every `risk_notes` entry for each candidate.

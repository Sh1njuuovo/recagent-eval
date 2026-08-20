# Candidate ranking methodology and evidence

## Snapshot identity

- Generated at: 2026-08-20 (Asia/Shanghai).
- Taste: disabled. No `taste.json` or `--taste` argument was used.
- Normalization denominator: 104 points.
- Python: 3.13.11; jq: 1.7.1-apple.
- Candidate scoring implementation: the bundled `shushu-internship-tool`
  module at
  `/Users/shinjuu/.codex/skills/shushu-internship-tool/scripts/shushu_internship_tool/candidate_score.py`.
- `candidate_score.py` SHA-256:
  `230237218e3becd2b449196dd696f139fd37ee41174102a77daeec6be43802bc`.
- Imported rendering/common helper SHA-256:
  `89476ad8504cabdc726e92ce107d37a2ca065053e089a492f7b72fae4ca4eaaf`.

The scoring CLI derives activity points from the system date. Reproducing this
exact snapshot requires the 2026-08-20 date. A run on a later date is a new
snapshot and may move a candidate into a different activity bucket.

## Inputs and outputs

| File | Role | SHA-256 |
| --- | --- | --- |
| `reports/profile/jd.txt` | active proxy JD | `2cceeba154e3d164312352b95245403afcc98e4acf038ac2eae5e27e8391d220` |
| `reports/profile/candidates.json` | explicit candidate facts and scores | `42ec95dbe8eb06ad62953ce97279780c158710e9595198c653e2b9757cea9d86` |
| `reports/ranking/candidate_score.json` | postprocessed machine-readable ranking | `c17dc0595426ccf5bedceeda3528e9cc6a081ca6546b2bfd1b67eccbde348a57` |
| `reports/ranking/candidate_score.md` | postprocessed human-readable ranking | `8656bb2cd54afe720abb1fe8e4111d9941da715f1ad095f56ae39285ced58d14` |

## Exact generation command

Run from the repository root on the snapshot date:

```bash
PYTHONPATH=/Users/shinjuu/.codex/skills/shushu-internship-tool/scripts \
  .venv/bin/python -m shushu_internship_tool.candidate_score \
  --jd reports/profile/jd.txt \
  --candidates reports/profile/candidates.json \
  --out reports/ranking
```

The command intentionally has no `--taste` argument.

## Deterministic postprocessing

The external generator currently emits a zero-valued
`score_breakdown.user_preference` even on the no-taste path, writes the phrase
`1 risk notes`, and limits Markdown risk display to three entries. The active
contract forbids the inactive preference field and requires every scored risk
to remain visible. Apply these deterministic steps after generation:

```bash
jq '(.candidates[]) |= ((.score_breakdown |= del(.user_preference)) | del(.taste_matches, .taste_mismatches, .user_preference_notes) | (.score_reasons |= map(if . == "1 risk notes" then "1 risk note" else . end)))' \
  reports/ranking/candidate_score.json \
  > /private/tmp/candidate-score-postprocessed.json
mv /private/tmp/candidate-score-postprocessed.json \
  reports/ranking/candidate_score.json
```

Then expand the unique OpenOneRec Markdown risk cell:

```bash
perl -0pi -e 's/license unclear; distributed pretraining; 100M interaction dataset \|/license unclear; distributed pretraining; 100M interaction dataset; documentation incomplete |/' \
  reports/ranking/candidate_score.md
```

The source text must occur exactly once before applying the command. This
changes display completeness only; all numerical scores continue to come from
`candidate_score`.

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
- License 0/4: no license was declared during the recorded audit. This is an
  audit-time risk statement, not a claim about future repository state.
- Runnable 4/20: the recorded path centers on distributed pretraining and a
  100M-interaction dataset, outside the two-week bounded full-run path.
- Resource fit 1/10: the same distributed/data requirements conflict with the
  approved local CPU plus single-4090 scope.
- Activity 10/10 and stars 8/10: the dated snapshot records 2026-05-18 and 885
  stars.
- Modification 10/20: two bounded extraction/smoke ideas are recorded.
- Risk -12: license uncertainty, distributed pretraining, the 100M-interaction
  dataset, and incomplete documentation are four separately scored risks.
- Arithmetic: `25 + 0 + 4 + 1 + 10 + 8 + 10 - 12 = 46`.

## Reproduction acceptance

A reproduction is accepted only if:

- input and external-script hashes match this file;
- the generator command is run without `--taste` on the snapshot date;
- deterministic postprocessing is applied exactly once;
- output hashes match this file;
- each JSON candidate has `max_raw_score: 104` and no Taste fields or
  `user_preference`; and
- the Markdown risk cell lists every `risk_notes` entry for each candidate.

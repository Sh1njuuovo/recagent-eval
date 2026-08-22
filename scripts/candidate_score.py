from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

MAX_RAW_SCORE = 104
TASTE_FIELDS = {
    "taste_tags",
    "avoid_tags",
    "project_taste_notes",
    "taste_matches",
    "taste_mismatches",
    "user_preference_notes",
}


def _clamp(value: Any, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(float(value)))))


def _activity_score(raw: Any, as_of: date) -> tuple[int, str]:
    try:
        commit_date = datetime.strptime(str(raw), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0, "last_commit missing or unparsable"
    days = max((as_of - commit_date).days, 0)
    if days <= 180:
        return 10, "active within 180 days"
    if days <= 365:
        return 7, "active within 1 year"
    if days <= 730:
        return 4, "active within 2 years"
    return 1, "inactive for more than 2 years"


def _stars_score(raw: Any) -> tuple[int, str]:
    stars = int(raw or 0)
    if stars >= 1000:
        return 10, "strong community signal"
    if stars >= 200:
        return 8, "good community signal"
    if stars >= 50:
        return 6, "some community signal"
    if stars > 0:
        return 3, "small community signal"
    return 0, "no star signal"


def _score_candidate(candidate: dict[str, Any], as_of: date) -> dict[str, Any]:
    clean = {key: value for key, value in candidate.items() if key not in TASTE_FIELDS}
    jd_score = _clamp(clean["jd_match_score"], 0, 30)
    license_score = _clamp(clean["license_score"], 0, 4)
    runnable_score = _clamp(clean["runnable_score"], -5, 20)
    resource_score = _clamp(clean["resource_fit_score"], 0, 10)
    activity_score, activity_reason = _activity_score(clean.get("last_commit"), as_of)
    stars_score, stars_reason = _stars_score(clean.get("stars"))
    mod_ideas = list(clean.get("mod_ideas") or [])
    risks = list(clean.get("risk_notes") or [])
    modification_score = min(20, len(mod_ideas) * 5)
    risk_penalty = min(20, len(risks) * 3)
    raw_score = (
        jd_score
        + license_score
        + runnable_score
        + resource_score
        + activity_score
        + stars_score
        + modification_score
        - risk_penalty
    )
    risk_label = "risk note" if len(risks) == 1 else "risk notes"
    return {
        **clean,
        "score": round(max(0, min(raw_score, MAX_RAW_SCORE)) * 100 / MAX_RAW_SCORE, 2),
        "raw_score": raw_score,
        "max_raw_score": MAX_RAW_SCORE,
        "score_breakdown": {
            "jd_match": jd_score,
            "license": license_score,
            "runnable": runnable_score,
            "resource_fit": resource_score,
            "activity": activity_score,
            "stars": stars_score,
            "modification_space": modification_score,
            "risk_penalty": -risk_penalty,
        },
        "matched_keywords": list(clean.get("matched_jd_terms") or []),
        "score_reasons": [
            "JD match score from jd_match_score",
            "license score from license_score",
            "runnable score from runnable_score",
            "resource fit score from resource_fit_score",
            activity_reason,
            stars_reason,
            f"{len(mod_ideas)} modification ideas",
            f"{len(risks)} {risk_label}",
        ],
    }


def _markdown(ranked: list[dict[str, Any]], jd_path: str) -> str:
    best = ranked[0]
    backup = ranked[1]
    lines = [
        "# 候选项目排序",
        "",
        f"- JD 来源：`{jd_path}`",
        f"- 主项目推荐：`{best['name']}`，score={best['score']:.2f}",
        f"- 备选项目：`{backup['name']}`，score={backup['score']:.2f}",
        "- 分数说明：先计算 raw_score，再按 max_raw_score 归一化到 0-100。",
        "",
        "| Rank | Name | Score | Raw | Max Raw | License | Stars | Last Commit "
        "| Runnable | Resources | Matched | Risks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(ranked, start=1):
        row = [
            str(rank),
            str(item["name"]),
            f"{item['score']:.2f}",
            str(item["raw_score"]),
            str(item["max_raw_score"]),
            str(item.get("license", "")),
            str(item.get("stars", "")),
            str(item.get("last_commit", "")),
            str(item.get("runnable", "")),
            str(item.get("resources", "")),
            ", ".join(item["matched_keywords"][:6]),
            "; ".join(item.get("risk_notes") or []) or "无",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- 这个脚本只根据显式字段打分；语义判断、JD 命中度和最终选择仍需 AI 助手/人工审阅。",
            "- 不可运行、资源要求过高、风险说明过多的项目，除非非常贴 JD，否则不建议作为主项目。",
            "- 推荐项目应尽快进入最小路径摸底、简历 4-5 行版本和面试 Q&A，而不是卡在完美复现。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank internship project candidates without Taste weighting."
    )
    parser.add_argument("--jd", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--as-of", required=True, help="Snapshot date in YYYY-MM-DD format.")
    args = parser.parse_args()

    jd_path = Path(args.jd)
    jd_path.read_text()
    payload = json.loads(Path(args.candidates).read_text())
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    ranked = sorted(
        (_score_candidate(dict(candidate), as_of) for candidate in payload["candidates"]),
        key=lambda item: (-item["score"], item["name"].lower()),
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "candidate_score.json"
    markdown_path = output / "candidate_score.md"
    json_path.write_text(json.dumps({"candidates": ranked}, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(_markdown(ranked, str(jd_path)))
    print(f"candidate_score_json: {json_path}")
    print(f"candidate_score_md: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 候选项目排序

- JD 来源：`reports/profile/jd.txt`
- 主项目推荐：`RecAI/InteRecAgent`，score=87.50
- 备选项目：`RecBole`，score=82.69
- 分数说明：先计算 raw_score，再按 max_raw_score 归一化到 0-100。

| Rank | Name | Score | Raw | Max Raw | License | Stars | Last Commit | Runnable | Resources | Matched | Risks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | RecAI/InteRecAgent | 87.50 | 91 | 104 | MIT | 1200 | 2026-01-27 | README, app.py and run.sh were found in the checked-in audit; external prepared resources remain required | local refactor/smoke path is plausible; upstream dependencies and prepared resources require validation | LLM Agent, query, retrieval, ranking, recommendation, evaluation | legacy dependencies; external prepared resources |
| 2 | RecBole | 82.69 | 86 | 104 | MIT with academic-use notice | 4500 | 2025-02-23 | historical shortlist audit rated the recommendation baseline and evaluation path highly; no run was repeated in this task | historical shortlist audit marked the baseline as CPU/single-GPU friendly | recommendation, ranking, datasets, evaluation | weak native agent integration |
| 3 | OpenOneRec | 44.23 | 46 | 104 | not declared during audit | 885 | 2026-05-18 | historical shortlist audit found no bounded two-week full-run path | historical shortlist audit identified distributed pretraining and a 100M-interaction dataset | generative recommendation, LLM, ranking | license unclear; distributed pretraining; 100M interaction dataset; documentation incomplete |

## 使用说明

- 这个脚本只根据显式字段打分；语义判断、JD 命中度和最终选择仍需 AI 助手/人工审阅。
- 不可运行、资源要求过高、风险说明过多的项目，除非非常贴 JD，否则不建议作为主项目。
- 推荐项目应尽快进入最小路径摸底、简历 4-5 行版本和面试 Q&A，而不是卡在完美复现。

# 候选项目排序

- JD 来源：`reports/profile/jd.txt`
- 主项目推荐：`RecAI / InteRecAgent`，score=88.60
- 备选项目：`RecBole`，score=82.46
- 分数说明：先计算 raw_score，再按 max_raw_score 归一化到 0-100。
- 用户偏好：已启用 Taste Fit 小权重；它只用于近分排序辅助，不覆盖 JD 匹配、可运行性和风险。

| Rank | Name | Score | Raw | Max Raw | License | Stars | Last Commit | Runnable | Resources | Matched | Taste Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | RecAI / InteRecAgent | 88.60 | 101 | 114 | MIT | 1200 | 2026-01-27 |  |  | LLM Agent, query, retrieval, ranking, recommendation, evaluation | 10/10; match: independent-project, interview-friendly, llm-agent, remote-4090 | legacy dependencies; external prepared resources |
| 2 | RecBole | 82.46 | 94 | 114 | MIT with academic-use notice | 4500 | 2025-02-23 |  |  | recommendation, ranking, datasets, evaluation | 8/10; match: interview-friendly, remote-4090, retrieval-ranking, search-recommendation | weak native agent integration |
| 3 | OpenOneRec | 42.11 | 48 | 114 | not declared during audit | 885 | 2026-05-18 |  |  | generative recommendation, LLM, ranking | 2/10; match: llm-agent, retrieval-ranking, search-recommendation; avoid: full-pretraining, multi-gpu | license unclear; distributed pretraining; 100M interaction dataset |

## 使用说明

- 这个脚本只根据显式字段打分；语义判断、JD 命中度和最终选择仍需 AI 助手/人工审阅。
- 不可运行、资源要求过高、风险说明过多的项目，除非非常贴 JD，否则不建议作为主项目。
- 推荐项目应尽快进入最小路径摸底、简历 4-5 行版本和面试 Q&A，而不是卡在完美复现。

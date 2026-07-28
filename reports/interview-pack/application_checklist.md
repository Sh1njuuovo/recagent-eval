# 投递检查表

| 项目项 | 证据 | 状态 |
| --- | --- | --- |
| JD 命中标题 | RecAgent-Eval：可评测的对话式电影推荐 Agent | 完成 |
| 用户画像/运行深度 | `resume_star.md` Profile Header | 完成 |
| 项目来源与许可证 | `NOTICE`、`LICENSE`、上游审计 | 完成 |
| 本地最小路径 | `uv run recagent-eval smoke` | 完成 |
| MovieLens 全链路 | 50 案例三组 rule-based 对照 | 完成 |
| 核心代码讲解 | `docs/core-code-walkthrough.md` | 完成 |
| 自动化测试 | pytest 35 tests | 完成 |
| 指标口径与负结果 | `reports/experiments/offline-rule-based.md` | 完成 |
| 简历 4–5 行 | `resume_star.md` | 完成 |
| 面试追问 | `interview_qa.md` | 完成 |
| 10 分钟展示 | `docs/demo-script.md` | 完成 |
| DeepSeek 正式 50 案例 | `reports/experiments/deepseek-formal.md` | 完成 |
| Qwen/vLLM 4090 冒烟 | `scripts/run_remote_qwen.sh` | 待远程主机 |
| 远程硬件/成本记录 | `docs/remote-4090.md` | 待实跑填写 |

投递时可使用已完成的本地与 DeepSeek 指标，但必须说明 plan validity 未达到
95% 目标；Qwen 数字仍需等远程主机实跑。

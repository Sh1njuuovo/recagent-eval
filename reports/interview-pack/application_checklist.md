# 投递检查表

| 项目项 | 证据 | 状态 |
| --- | --- | --- |
| JD 命中标题 | RecAgent-Eval：可评测的对话式电影推荐 Agent | 完成 |
| 用户画像/运行深度 | `resume_star.md` Profile Header | 完成 |
| 项目来源与许可证 | `NOTICE`、`LICENSE`、上游审计 | 完成 |
| 本地最小路径 | `uv run recagent-eval smoke` | 完成 |
| MovieLens 全链路 | 50 案例三组 rule-based 对照 | 完成 |
| 核心代码讲解 | `docs/core-code-walkthrough.md` | 完成 |
| 自动化测试 | pytest 229 tests，90% line coverage | 完成 |
| 指标口径与负结果 | `reports/experiments/deepseek-constraint-aware.md` | 完成 |
| 简历 4–5 行 | `resume_star.md` | 完成 |
| 面试追问 | `interview_qa.md` | 完成 |
| 10 分钟展示 | `docs/demo-script.md` | 完成 |
| DeepSeek 正式 50 案例 | `reports/experiments/deepseek-constraint-aware.md` | 完成 |
| dense 缓存与 LambdaMART 证据 | `reports/experiments/v2-dense-lambdamart-500user.md` | 完成 |
| Confirmation-B 正式认证 | Recall@10 0.118、NDCG@10 0.0555 | 完成 |
| 本地 Demo 截图（rule-based） | `reports/demo/v2-demo-lambdamart-rule-based.png` | 完成 |
| Qwen/vLLM 4090 冒烟 | `scripts/run_remote_qwen.sh` | 待远程主机 |
| 远程硬件/成本记录 | `docs/remote-4090.md` | 待实跑填写 |

投递时保留早期 LambdaMART 负结果作为诊断过程。主要算法数字来自全新
1000-user Confirmation-B：Recall@10 0.118、NDCG@10 0.0555，ItemCF 为
0.064/0.0323，bootstrap CI 下界大于 0，约束 100%。Confirmation-A 标为开发
证据。The frozen test remains unconsumed. Qwen/4090 remains pending.

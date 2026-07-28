# PPT 生成提示词

请生成一份 8 页中文技术面试 PPT 大纲，项目名为“RecAgent-Eval：可评测的对话式电影推荐 Agent”，目标岗位是搜推 × LLM 算法实习。每页给出标题、3–5 个要点、建议图表和 30–45 秒讲稿。

必须包含：

1. 问题：直接 LLM 推荐的幻觉、约束违规与不可复现。
2. 来源：RecAI/InteRecAgent 的工具化思想；明确本项目为独立实现。
3. 架构：Conversation → PreferenceState/ToolPlan → Hard Filter → ItemCF + TF-IDF → Hybrid Rerank → Traces/Metrics。
4. 数据：MovieLens-1M，用户时间切分，验证调权、测试冻结。
5. 改造：DeepSeek/vLLM Provider、一次修复、确定性回退、50 案例、35 个测试、稳定 manifest。
6. 结果表：baseline `Recall/NDCG=0.06/0.0486`；full `0.08/0.0418`；约束与工具成功率 100%。
7. 失败分析：覆盖提升但头部排序下降；零相似度候选和 set fingerprint 两个 debug 案例。
8. 下一步：DeepSeek 正式矩阵、Qwen 4090 冒烟、分数校准/轻量 LTR。

不要把 rule-based provider 的结果表述成 LLM 效果，不要声称 NDCG 提升，不要编造 Qwen 吞吐或显存数字。

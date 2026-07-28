# PPT 生成提示词

请生成一份 8 页中文技术面试 PPT 大纲，项目名为“RecAgent-Eval：可评测的对话式电影推荐 Agent”，目标岗位是搜推 × LLM 算法实习。每页给出标题、3–5 个要点、建议图表和 30–45 秒讲稿。

必须包含：

1. 问题：直接 LLM 推荐的幻觉、约束违规与不可复现。
2. 来源：RecAI/InteRecAgent 的工具化思想；明确本项目为独立实现。
3. 架构：Conversation → PreferenceState/ToolPlan → Hard Filter → ItemCF + TF-IDF → Hybrid Rerank → Traces/Metrics。
4. 数据：MovieLens-1M，用户时间切分，验证调权、测试冻结。
5. 改造：DeepSeek/vLLM Provider、一次修复、确定性回退、50 案例、54 个测试、稳定 manifest 与 case fingerprint。
6. 结果表：unstructured baseline `Recall/NDCG=0.06/0.0486`；structured ItemCF `0.06/0.0360`、候选覆盖 78%；full hybrid `0.04/0.0149`、候选覆盖 88%；正式结构化方案的计划、工具、pipeline 和硬约束指标均为 100%。
7. 失败分析：双路召回相对 ItemCF 将候选覆盖提高 10 个百分点，但未经校准的线性融合损害头部排序；同时讲标签预检、检索路由约束和冻结参数防覆盖三个评测完整性修复。
8. 下一步：在冻结测试集不变的前提下，先用验证集比较分数校准、RRF 和轻量 LTR；离线 NDCG 改善后再重跑 DeepSeek，Qwen 4090 只做兼容性冒烟。

不要把 rule-based provider 的结果表述成 LLM 效果，不要声称 Recall 或 NDCG 提升，不要把无记忆流行度基线在 50 案例上的优势泛化，也不要编造 Qwen 吞吐或显存数字。

# PPT 生成提示词

请生成一份 8–9 页中文技术面试 PPT 大纲，项目名为“RecAgent-Eval：可评测的对话式电影推荐 Agent（dense 召回 + LambdaMART）”，目标岗位是搜推 × LLM 算法实习。每页给出标题、3–5 个要点、建议图表和 30–45 秒讲稿。

必须包含：

1. 问题：直接 LLM 推荐的幻觉、约束违规与不可复现。
2. 来源：RecAI/InteRecAgent 的工具化思想；明确本项目为独立实现。
3. 架构：Conversation → PreferenceState/ToolPlan → Hard Filter → ItemCF + Dense（`all-MiniLM-L6-v2`）→ 十特征 LambdaMART → Traces/Metrics。
4. 数据与契约：MovieLens-1M，用户时间切分，整用户 GroupKFold 三分 CV，模型/证据/指纹 bundle 单次消费、fail-closed。
5. 真实调试：faulthandler + lldb 定位三份 `libomp.dylib` 共存导致的段错误，`n_jobs=1`/`OMP_NUM_THREADS=1` 修复并补子进程回归测试；229 个测试、90% 覆盖。
6. v2 结果表：500 用户约束满足率 100%；LambdaMART NDCG@10 `0.0327` vs ItemCF `0.0334`，bootstrap CI `[−0.0146, 0.0129]` 跨零；并集候选召回 77.6%、dense 仅 28.8%。frozen 门保持锁定。
7. 失败分析：瓶颈是候选召回而非最终排序；保留完整负结果；DeepSeek 历史矩阵中双路召回把候选覆盖从 78% 提到 88% 但 top-10 未改善，同一结论互相印证。
8. 下一步：先提高 dense 候选召回与分数校准，离线 NDCG 改善后再考虑 frozen 重跑；Qwen 4090 只做兼容性冒烟，不虚构吞吐/显存数字。

不要把 rule-based provider 的结果表述成 LLM 效果，不要声称 Recall 或 NDCG 提升，不要把无记忆流行度基线在 50 案例上的优势泛化，也不要编造 Qwen 吞吐或显存数字；所有指标必须与 `artifacts/experiments/v2-500/validation.json` 一致。

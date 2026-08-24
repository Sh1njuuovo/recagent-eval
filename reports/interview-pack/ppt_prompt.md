# PPT 生成提示词

请生成一份 8–9 页中文技术面试 PPT 大纲，项目名为“RecAgent-Eval：可评测的对话式电影推荐 Agent（dense 召回 + LambdaMART）”，目标岗位是搜推 × LLM 算法实习。每页给出标题、3–5 个要点、建议图表和 30–45 秒讲稿。

必须包含：

1. 问题：直接 LLM 推荐的幻觉、约束违规与不可复现。
2. 来源：RecAI/InteRecAgent 的工具化思想；明确本项目为独立实现。
3. 架构：Conversation → PreferenceState/ToolPlan → Hard Filter → ItemCF + Dense（`all-MiniLM-L6-v2`）→ 十特征 LambdaMART → Traces/Metrics。
4. 数据与契约：MovieLens-1M，用户时间切分，整用户 GroupKFold 三分 CV，模型/证据/指纹 bundle 单次消费、fail-closed。
5. 真实调试：faulthandler + lldb 定位三份 `libomp.dylib` 共存导致的段错误，`n_jobs=1`/`OMP_NUM_THREADS=1` 修复并补子进程回归测试；229 个测试、90% 覆盖。
6. 证据演进：保留 500-user LambdaMART NDCG@10 `0.0327` vs ItemCF `0.0334`
   的负结果；ALS latent recall@500 `0.838`、union recall `0.928`；全新
   Confirmation-B 上 current_v2b Recall@10 `0.118`、NDCG@10 `0.0555`
   vs ItemCF `0.064`/`0.0323`，bootstrap CI 下界大于 0，约束 100%。
7. 失败分析：瓶颈是候选召回而非最终排序；保留完整负结果；DeepSeek 历史矩阵中双路召回把候选覆盖从 78% 提到 88% 但 top-10 未改善，同一结论互相印证。
8. 证据纪律：Confirmation-A 降级为开发证据，Confirmation-B 唯一认证；P0 后
   再申请一次 frozen 授权，失败也不调参或重跑；Qwen 4090 仍 pending。

不要把 rule-based provider 的结果表述成 LLM 效果，不要把项目内 LightGCN
写成 canonical 实现，不要编造 Qwen 吞吐或显存数字。主要算法数字来自
Confirmation-B JSON/evidence bundle。The frozen test remains unconsumed.

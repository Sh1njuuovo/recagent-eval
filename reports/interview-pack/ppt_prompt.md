# PPT 生成提示词

请生成一份 8–9 页中文技术面试 PPT 大纲，项目名为“RecAgent-Eval：可评测的对话式电影推荐 Agent（dense 召回 + LambdaMART）”，目标岗位是搜推 × LLM 算法实习。每页给出标题、3–5 个要点、建议图表和 30–45 秒讲稿。

必须包含：

1. 问题：直接 LLM 推荐的幻觉、约束违规与不可复现。
2. 来源：RecAI/InteRecAgent 的工具化思想；明确本项目为独立实现。
3. 架构：Conversation → PreferenceState/ToolPlan → Hard Filter → ItemCF + Dense（`all-MiniLM-L6-v2`）→ 十特征 LambdaMART → Traces/Metrics。
4. 数据与契约：MovieLens-1M，用户时间切分，整用户 GroupKFold 三分 CV，模型/证据/指纹 bundle 单次消费、fail-closed。
5. 真实调试：faulthandler + lldb 定位三份 `libomp.dylib` 共存导致的段错误，`n_jobs=1`/`OMP_NUM_THREADS=1` 修复并补子进程回归测试；438 个测试、90.04% 覆盖。
6. 证据演进：保留 500-user LambdaMART NDCG@10 `0.0327` vs ItemCF `0.0334`
   的负结果；ALS latent recall@500 `0.838`、union recall `0.928`；全新
   Confirmation-B 上 current_v2b Recall@10 `0.118`、NDCG@10 `0.0555`
   vs ItemCF `0.064`/`0.0323`，bootstrap CI 下界大于 0，约束 100%。
7. 失败分析：保留完整负结果；50-case promotion 的 union 覆盖为 47/50、Top-10
   命中为 4/50，说明主要剩余瓶颈是头部排序深度。
8. 证据纪律：Confirmation-A 降级为开发证据，Confirmation-B 是算法比较主证据；
   锁定 current_v2b 后的一次性 50-case promotion 为泛化补充，Recall@10 `0.08`、
   NDCG@10 `0.03964`。同一 suite 曾用于历史 DeepSeek 实验，本次没有匹配
   ItemCF/ALS 对照，因此不声明纯净 holdout 或 baseline 显著胜出；identity 已永久
   消费，后续不调参、不重跑，v3 预注册新 holdout。Qwen 4090 仍 pending。

不要把 rule-based provider 的结果表述成 LLM 效果，不要把项目内 LightGCN
写成 canonical 实现，不要编造 Qwen 吞吐或显存数字。主要算法数字来自
Confirmation-B JSON/evidence bundle。50-case promotion 与历史 DeepSeek 必须分层展示。

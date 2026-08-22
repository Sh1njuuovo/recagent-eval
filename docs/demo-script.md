# 十分钟演示脚本（本地可跑版本）

当前版本以**无需 API key** 的本地 Demo 为主线：rule-based provider +
dense 语义召回 + v2 LambdaMART 排序模型。DeepSeek 结果作为历史对照表，
Qwen/vLLM 只说明待远程主机。

演示命令（先在 `.venv` 中 `uv sync --extra demo --extra ml`）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/recagent-eval demo \
  --data-dir /Users/shinjuu/intern/recagent-eval/data/raw/ml-1m \
  --provider rule-based \
  --semantic-config configs/v2_dense_validation.yaml \
  --ranker-config configs/v2_demo_lambdamart.yaml
```

在输入框输入：`Recommend a science fiction movie from the 1990s.` 后点击
Recommend。截图见
[reports/demo/v2-demo-lambdamart-rule-based.png](../reports/demo/v2-demo-lambdamart-rule-based.png)；
runtime 面板会明确标注 provider 为 rule-based，不能表述成 LLM 效果。

1. **0:00–1:00 — 问题。** 对话推荐既需要 LLM 理解开放语言，也需要确定性的候选
   边界、硬约束和离线评测；直接让 LLM 生成电影不可控、不可复现。
2. **1:00–2:00 — 项目边界。** 参考 RecAI/InteRecAgent 的工具化思想，本项目独立
   实现（见 `NOTICE`）；LLM 只做偏好理解与工具计划，过滤/召回/排序/指标全部
   确定性化。
3. **2:00–4:00 — 架构（v2）。** 从用户表达依次讲 `PreferenceState`、schema
   校验后的 `ToolPlan`、硬过滤、ItemCF + Dense（`all-MiniLM-L6-v2`）双路召回、
   十特征 LambdaMART 重排、特征贡献解释与工具轨迹。Demo 右侧面板展示
   PreferenceState、ToolPlan、Tool trace、ScoreBreakdown 与 runtime。
4. **4:00–5:15 — 可靠性。** 测试注入非法 provider 输出，展示一次修复、失败则
   确定性回退；正式矩阵中计划、工具、pipeline 和硬约束指标 100%，排除项违规 0。
5. **5:15–6:30 — 防泄漏评测。** 按用户时间切分，验证目标与测试目标都不进入
   ItemCF 训练；LambdaMART 用整用户 GroupKFold 三分 CV 选参、验证证据可回放，
   frozen 评测单次消费、fail-closed。
6. **6:30–8:15 — v2 本地结果。** 展示 500-user 表：约束 100%，但 LambdaMART
   NDCG@10（0.0327）未超过 ItemCF（0.0334），bootstrap CI 跨零，frozen 门保持
   锁定；候选并集召回 77.6%、dense 召回仅 28.8%。这是**如实保留的负结果**。
7. **8:15–9:30 — 真实调试案例。** 讲 LightGBM 段错误：torch/LightGBM/scikit-learn
   各加载一份 `libomp.dylib`，多线程训练在 `__kmp_suspend_initialize_thread`
   空指针崩溃；用 faulthandler/lldb 拿到原生栈后，通过 `n_jobs=1` 与
   `OMP_NUM_THREADS=1` 修复，并新增两个子进程回归测试（先写测试、看它崩、
   再修代码）。
8. **9:30–10:00 — 下一步。** 保持测试矩阵冻结；先在验证集提高候选召回
   （尤其是 dense 通道 28.8%）与分数校准，离线 NDCG 改善后再考虑 frozen 重跑。
   DeepSeek 历史结果见
   [deepseek-constraint-aware](../reports/experiments/deepseek-constraint-aware.md)；
   Qwen/vLLM 待 4090 空闲后仅做兼容性冒烟，不虚构吞吐/显存数字。

## 必须遵守的口径

- rule-based Demo 截图只能标注为本地链路验证，不能表述成 LLM 推荐效果。
- 简历数字一律从 `artifacts/experiments/v2-500/validation.json` 等已核验 JSON
  复制，不临时估算。
- 当前安全主结论：搭建了防泄漏的 Agent/检索/排序评测系统，完成可复现的 dense
  召回与 LambdaMART 证据契约，并定位到剩余的真实训练/运行瓶颈。

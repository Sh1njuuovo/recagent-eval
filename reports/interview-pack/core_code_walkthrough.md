# 核心代码讲解稿

1. **入口**：`recagent_eval.cli` 的 `evaluate` 读取 YAML、MovieLens 与固定案例，再调用 `run_experiment`。
2. **配置**：三个 YAML 只改变 structured planning、memory、semantic retrieval 和 weights，避免为每个 baseline 写分叉代码。
3. **输入输出**：输入为 user message、`PreferenceState`、电影表和训练评分；输出为 `RecommendationResult`，包含推荐项、分数分解、计划、轨迹、token、延迟和错误。
4. **Agent**：`agent.py` 完成 provider 调用、schema 校验、一次修复、fallback 与工具顺序执行。
5. **搜推**：`retrieval.py` 做硬过滤、ItemCF、TF-IDF；`ranking.py` 做归一化、混合排序与验证调权。
6. **评测**：`evaluation.py` 计算 Recall/NDCG/HitRate、计划合法率、工具成功率、约束与延迟；`runner.py` 落盘 episode、metrics 和 manifest。
7. **测试**：provider 用真实 HTTP request 形状配合 `httpx.MockTransport`；其余模块尽量用真实小数据而非 mock。
8. **失败案例**：重点讲零相似度候选和跨进程 hash 不稳定两个由测试发现的问题。

完整版本见 `docs/core-code-walkthrough.md`。

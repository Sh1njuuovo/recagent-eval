# 面试拷问 Q&A

## 1. 为什么这个项目匹配搜推 × LLM？

LLM 只负责把开放语言转换为 `PreferencePatch` 和 `ToolPlan`；过滤、召回、排序、指标仍由确定性搜推模块完成。因此既能讲 Agent 的工具调用和记忆，也能讲 ItemCF、向量召回、NDCG、消融和数据泄漏。

## 2. 原项目和你的工作边界是什么？

RecAI/InteRecAgent 提供 Query、Filter、Retrieval、Ranking 和 plan-first 的概念启发。我的仓库没有拷贝其源码；我重新实现 Python 3.11 接口、MovieLens 切分、Provider、结构化状态、双路召回、重排、评测、测试和远程脚本，并在 `NOTICE` 保留归属。

## 3. 一次请求怎么走？

输入是用户文本与当前 `PreferenceState`。Provider 输出 JSON；Pydantic 校验工具白名单与顺序；硬过滤先剔除不允许项目；ItemCF 和 TF-IDF 分别返回 Top-100；混合重排输出 Top-10、分数分解、理由、工具轨迹和系统指标。

## 4. 为什么不让 LLM 直接推荐电影？

直接生成会出现幻觉、库存外物品、违反排除条件以及难以复现的问题。LLM 更适合处理自然语言和规划，候选边界与排序交给可审计模块，能保证 excluded item 违规率可测且当前为 0。

## 5. 非法 JSON 或接口失败怎么办？

Provider 对 408/409/429/5xx 和网络错误指数退避。Agent 对非法 schema 只修复一次；仍失败就使用固定安全计划，保留错误码和 `fallback_used`，但不让一个 episode 中断整批评测。API Key 只在私有属性和环境变量中出现。

## 6. 如何避免数据泄漏？

对每个用户按时间排序；最后一个正反馈作为测试目标，倒数第二个作为验证目标，两者都不进入 ItemCF 训练。混合权重只用验证目标搜索，选定后冻结。所有 variant 使用同一规范化 case fingerprint。

## 7. 为什么权重最后是 0.7/0.3/0？

步长 0.1、最多 500 个验证用户的搜索选择了 0.7 ItemCF、0.3 TF-IDF、0.0 手工偏好分。偏好仍参与硬过滤、记忆和解释，但手工 affinity 没提高验证 NDCG，所以没有为了故事完整强行给正权重。

## 8. 为什么 Recall 提升但 NDCG 下降？

语义通道扩大了覆盖，使一个额外测试目标进入 Top-10，所以 Recall/HitRate 从 0.06 到 0.08；但该目标位置偏后，整体折损增益不足，NDCG 为 0.0418，低于 0.0486 基线。下一步应做源分数校准或轻量 learning-to-rank。

## 9. 你实际 debug 过什么？

第一，集成测试发现语义召回会把零相似度电影塞入候选，修复为只接收正相似度。第二，三次 manifest 的 case hash 不一致，定位到 `set` 序列化受 hash seed 影响，改成递归排序后再 SHA-256。

## 10. 本地、DeepSeek 和 Qwen 的结果如何区分？

rule-based 表格只验证离线链路；DeepSeek 正式矩阵完整组最初计划合法率
86%、fallback 14%，且失败全部集中在多轮 episode。复现发现最终轮漏掉
`hard_filter`，强化首轮与 repair 约束后，10 个多轮案例复测达到 100%
合法率、0% fallback。Qwen/vLLM 仍只用于 10–20 条兼容性测试。

## 11. 如何估算线上延迟和成本？

manifest 和 episode 同时记录 LLM 延迟、工具延迟、调用次数和 token。线上会为 Provider 设置超时/重试预算，对 ItemCF/TF-IDF 做常驻索引与批量计算，对热门 query 缓存候选，并把确定性回退作为降级路径。

## 12. 最大限制是什么？

MovieLens 文本只有标题和类型，TF-IDF 语义较弱；固定案例只覆盖离线用户历史；当前没有训练排序器。v1 故意不引入多 Agent 或大规模训练，优先保证证据链清楚且一周可完成。

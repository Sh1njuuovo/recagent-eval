# 面试拷问 Q&A

## 1. 为什么这个项目匹配搜推 × LLM？

LLM 只负责把开放语言转换为 `PreferencePatch` 和 `ToolPlan`；过滤、召回、排序、指标仍由确定性搜推模块完成。因此既能讲 Agent 的工具调用和记忆，也能讲 ItemCF、向量召回、NDCG、消融和数据泄漏。

## 2. 原项目和你的工作边界是什么？

RecAI/InteRecAgent 提供 Query、Filter、Retrieval、Ranking 和 plan-first 的概念启发。我的仓库没有拷贝其源码；我重新实现 Python 3.11 接口、MovieLens 切分、Provider、结构化状态、双路召回、重排、评测、测试和远程脚本，并在 `NOTICE` 保留归属。

## 3. 一次请求怎么走？

输入是用户文本与当前 `PreferenceState`。Provider 输出 JSON；Pydantic 校验工具白名单、顺序和当前实验要求的检索路由；硬过滤先剔除不允许项目；ItemCF 和 TF-IDF 按验证集冻结的 Top-500 深度召回；混合重排输出 Top-10、分数分解、理由、工具轨迹和系统指标。

## 4. 为什么不让 LLM 直接推荐电影？

直接生成会出现幻觉、库存外物品、违反排除条件以及难以复现的问题。LLM 更适合处理自然语言和规划，候选边界与排序交给可审计模块，能保证 excluded item 违规率可测且当前为 0。

## 5. 非法 JSON 或接口失败怎么办？

Provider 对 408/409/429/5xx 和网络错误指数退避。Agent 对非法 schema 只修复一次；仍失败就使用固定安全计划，保留错误码和 `fallback_used`，但不让一个 episode 中断整批评测。API Key 只在私有属性和环境变量中出现。

## 6. 如何避免数据泄漏？

对每个用户按时间排序；最后一个正反馈作为测试目标，倒数第二个作为验证目标，两者都不进入 ItemCF 训练。混合权重只用验证目标搜索，选定后冻结。所有 variant 使用同一规范化 case fingerprint。

## 7. 为什么权重最后是 0.7/0.3/0？

步长 0.1、最多 500 个验证用户的搜索选择了 0.7 ItemCF、0.3 TF-IDF、0.0 手工偏好分。偏好仍参与硬过滤、记忆和解释，但手工 affinity 没提高验证 NDCG，所以没有为了故事完整强行给正权重。

## 8. 为什么候选覆盖提升，但 Recall 和 NDCG 反而下降？

和同样使用记忆、同样冻结 Top-500 深度的 ItemCF 方案相比，语义通道把测试目标的候选并集覆盖率从 0.78 提高到 0.88；但固定线性融合没有把新增相关项排进前十，反而扰动了 ItemCF 的头部顺序。因此 Recall@10 从 0.06 降到 0.04，NDCG@10 从 0.0360 降到 0.0149。这说明问题不是“没召回”，而是异构分数不可直接相加；下一步应先在验证集做分数校准、RRF 或轻量 learning-to-rank。

## 9. 你实际 debug 过什么？

第一，集成测试发现语义召回会把零相似度电影塞入候选，修复为只接收正相似度。第二，三次 manifest 的 case hash 不一致，定位到 `set` 序列化受 hash seed 影响，改成递归排序后再 SHA-256。

## 10. 本地、DeepSeek 和 Qwen 的结果如何区分？

rule-based 表格只验证离线链路，不能表述成 LLM 效果。当前可对外使用的是同一 case fingerprint 下的 DeepSeek 正式矩阵：结构化 ItemCF 和完整方案的计划合法率、pipeline 合规率、工具成功率、硬约束满足率均为 100%，fallback 为 0。此前旧矩阵暴露过标签与负类型约束冲突、完整方案漏语义路由、计划 `top_k` 覆盖冻结预算等评测污染；旧、新 fingerprint 不同，因此不合并结果。Qwen/vLLM 仍只计划做 10–20 条兼容性冒烟，服务器未空闲前不报告吞吐和显存。

## 11. 如何估算线上延迟和成本？

manifest 和 episode 同时记录 LLM 延迟、工具延迟、调用次数和 token。线上会为 Provider 设置超时/重试预算，对 ItemCF/TF-IDF 做常驻索引与批量计算，对热门 query 缓存候选，并把确定性回退作为降级路径。

## 12. 最大限制是什么？

MovieLens 文本只有标题和类型，TF-IDF 语义较弱；50 个固定案例的 top-10 指标方差较大；无记忆基线会退化为流行度召回，在这组小样本上得到更高 NDCG，不能泛化为“不需要记忆”。当前也没有训练排序器。v1 故意不引入多 Agent 或大规模训练，优先保证证据链清楚且一周可完成。

## 13. v2 为什么把语义通道换成 dense？缓存怎么做安全？

MovieLens 只有标题和类型文本，TF-IDF 是词袋近似；`all-MiniLM-L6-v2` 提供更稳
定的语义表征。缓存不是普通二进制文件：保存 schema 版本、模型名/修订 SHA、
数据集指纹、维度、归一化校验和与库版本，加载时逐一校验，防止模型或数据被静默
替换后污染评测。

## 14. LambdaMART 怎么保证不漏标签、不偷看？

三层隔离：时间切分（验证/测试目标都不进 ItemCF 训练）；整用户 GroupKFold 三分
CV（同一用户不会同时出现在训练与验证 fold）；证据契约（bundle 记录训练行、
历史、fold map、候选策略、配置、案例与验证回放行的指纹，frozen 评测把验证行
重算并逐字节比对）。bundle 是单次消费、fail-closed，不能绕过门禁重放。

## 15. 你实际 debug 过原生段错误吗？怎么做的？

`train-ranker` 首次实跑直接 exit 139。我用 faulthandler 复现，再用 lldb 拿到
原生栈，发现多个线程死在 `__kmp_suspend_initialize_thread`；`image list` 显示
进程里同时有三份 `libomp.dylib`（torch、LightGBM、scikit-learn）。根因是
LightGBM 的 OpenMP worker 线程 barrier 调用落入 torch 的 libomp，空指针崩溃。
修复是 `n_jobs=1`、Booster 预测 `num_threads=1`、加载前 `OMP_NUM_THREADS=1`，
并先写子进程回归测试看它崩，再让测试变绿。

## 16. 500 用户验证没超过 ItemCF，为什么还写进简历？

因为负结果是有信息量的证据：约束满足率 100% 证明评测契约工作；NDCG 未提升且
bootstrap 95% CI 跨零，说明差异不显著；并集候选召回 77.6%、dense 仅 28.8%，
把瓶颈定位在“候选进没进 Top-N”，而不是最终排序器。隐藏负结果反而会在追问中
失去可信度；报告里同时给出复现命令和全部指纹。

后续实验没有覆盖这项负结果：它通过 ALS latent route 和新特征形成新的 v2b
路线。在此前未参与选择的 1000-user Confirmation-B 上，Recall@10 0.118、
NDCG@10 0.0555，超过 ItemCF 0.064/0.0323，paired bootstrap CI 下界大于 0。
Confirmation-A 因读数后修复只作为开发证据。

## 17. 固定单线程是不是在绕开问题？

不是。根因是三个 OpenMP 运行时共存，任何 LightGBM 并行区都可能崩；单线程让
LightGBM 不进入并行区，是确定性修复，不是把段错误吞掉或关掉门禁。回归测试在
torch 已加载的子进程里做真实训练/预测/加载，换环境或依赖升级后会重新暴露问题。
项目还保留了完整证据契约，门禁、bootstrap、证据回放和 frozen 标记一个都没动。

## 18. 为什么 frozen 只能跑一次？

反复查看 frozen 结果会把测试集变成调参集。P0 先锁定 commit、数据、cohort、
config、model 和 evidence fingerprint，再请求一次明确授权；失败或崩溃也不会
调参或重跑。The frozen test remains unconsumed. Qwen/4090 remains pending.

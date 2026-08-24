# STAR 简历项目

## Profile Header

- 目标岗位：搜索/推荐与 LLM Agent 算法实习
- 用户水平：有 PyTorch、Agent/实验工程项目经验
- 技术栈：Python、Pydantic、NumPy、scikit-learn、LightGBM、
  sentence-transformers、pytest
- 时间预算：一周
- 资源条件：本地 Mac + 远程 RTX 4090（Qwen 待跑）
- 当前状态：本地全链路、DeepSeek 50 案例矩阵、dense 召回与 LambdaMART
  证据契约均已完成；Qwen/4090 冒烟待服务器空闲

## 4–5 行简历版本

**RecAgent-Eval｜可评测的对话式电影推荐 Agent（dense 召回 + LambdaMART）**

- 针对传统推荐难以处理自然语言约束、LLM 推荐又缺乏可重复评测的问题，独立实现
  MovieLens-1M 对话推荐系统：LLM 只负责偏好理解与工具计划，过滤、召回、排序、
  指标全部由确定性模块完成。
- 实现时间防泄漏的用户切分、ItemCF 与 `all-MiniLM-L6-v2` dense 双路召回、
  硬约束过滤及验证集调参；新增整用户 GroupKFold 三分 CV 的十特征 LambdaMART
  排序器，模型/证据/指纹 bundle 单次消费、fail-closed。
- 用 faulthandler + lldb 定位并修复真实原生崩溃：torch/LightGBM/scikit-learn
  三份 `libomp.dylib` 共存导致 LightGBM 多线程训练段错误，通过固定单线程并补
  两个子进程回归测试解决；395 个自动化测试，90.02% 行覆盖率。
- 从早期 500 用户 LambdaMART 负结果出发，通过 ALS latent route 与交叉/时序
  特征改善候选深度；在全新 1000-user Confirmation-B 上将 Recall@10 从
  ItemCF 0.064 提升至 0.118、NDCG@10 从 0.0323 提升至 0.0555，paired
  bootstrap 95% CI 下界大于 0，约束满足率 100%。
- DeepSeek 历史矩阵中，双路召回相对同深度 ItemCF 将候选覆盖率从 78% 提升到
  88%；计划合法率、工具成功率、硬约束满足率均 100%，同时如实保留
  Recall@10/NDCG@10 未提升的负排序结果。

## STAR 口述版

- S：对话推荐里，LLM 很会理解意图，但直接生成电影不可控、不可复现；传统推荐
  又处理不了自然语言约束。
- T：一周内做出既有 Agent 感、又能用搜推指标、消融和可复现证据证明的项目，
  并把“能训练”升级为“评测契约防泄漏”。
- A：把 LLM 输出收敛为结构化偏好与工具计划；推荐侧独立执行过滤、双路召回、
  排序和指标；实现 dense 缓存指纹校验、整用户分组 CV 的 LambdaMART、验证证据
  回放与单次 frozen 消费；遇到原生段错误时不绕过，先写复现测试再修。
- R：保留早期负结果，并在此前未参与选择的 Confirmation-B 上得到 Recall@10
  0.118、NDCG@10 0.0555；相对 ItemCF 0.064/0.0323 的提升通过 2,000 次
  用户级 paired bootstrap 显著性验证。
- Reflection：Confirmation-A 因读数后 baseline 修复降级为开发证据；主 claim
  只使用 Confirmation-B。frozen test 尚未消费，Qwen/4090 仍 pending。

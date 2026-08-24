# RecAgent-Eval interview pack

- [STAR 简历项目](resume_star.md)
- [面试拷问 Q&A](interview_qa.md)
- [核心代码讲解](core_code_walkthrough.md)
- [PPT 生成提示词](ppt_prompt.md)
- [投递检查表](application_checklist.md)
- [10 分钟演示脚本](../../docs/demo-script.md)

当前可安全对外表述：早期 500-user dense/LambdaMART 负结果完整保留；ALS latent
route 将 latent recall@500 提至 0.838、union recall 提至 0.928。在全新
1000-user Confirmation-B 上，current_v2b Recall@10 0.118、NDCG@10 0.0555，
超过 ItemCF 0.064/0.0323 且 paired-bootstrap CI 下界大于 0，约束 100%。
Confirmation-A 只作开发证据。锁定 current_v2b 后的一次性 50-case final
promotion evaluation 得到
Recall@10 0.08、NDCG@10 0.03964。该 suite 曾用于历史 DeepSeek 系统实验，且
本次没有匹配 ItemCF/ALS 对照，因此只作泛化补充，不支持 holdout 纯净性或
baseline 显著性 claim。该 identity 已永久消费，50 cases 不再用于调参；
Qwen/4090 remains pending.

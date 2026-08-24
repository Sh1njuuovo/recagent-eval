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
Confirmation-A 只作开发证据。The frozen test remains unconsumed. Qwen/4090
remains pending.

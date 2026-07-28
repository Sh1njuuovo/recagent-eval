# RecAI-InteRecAgent 项目摸底报告

## 基本信息

| 字段 | 值 |
| --- | --- |
| repo_path | /private/tmp/recagent-audit.JQMsWk/RecAI/InteRecAgent |
| generated_at | 2026-07-28T04:56:23.822592+00:00 |
| file_count_scanned | 69 |
| approx_total_bytes | 3455653 |

## 语言和文件类型

| 语言 | 文件数 |
| --- | --- |
| Python | 44 |
| Other | 17 |
| Notebook | 5 |
| Shell | 3 |

## 依赖和环境线索

- `requirements.txt`

## README

- `README.md`
- `demonstration/README.md`
- `tests/README.md`

## 核心链路线索

| 类别 | 命中文件数 | 代表路径 |
| --- | --- | --- |
| data_pipeline | 5 | preprocess/ItemCF.ipynb<br>preprocess/movies.ipynb<br>preprocess/prepare_amazon.ipynb<br>preprocess/prepare_steam.ipynb<br>preprocess/preprocess_redial.ipynb |
| database_state | 3 | llm4crs/ranking/reco_model_tool.py<br>llm4crs/retrieval/sql_tool.py<br>llm4crs/utils/sql.py |
| evaluation | 10 | eval/eval_simulator.sh<br>eval/eval_single_turn.sh<br>eval/one_turn_eval.py<br>eval/one_turn_eval_ranking.py<br>eval/user_simulator.py<br>llm4crs/retrieval/__init__.py<br>llm4crs/retrieval/itemcf_tool.py<br>llm4crs/retrieval/sql_tool.py |
| frontend_mobile | 1 | llm4crs/buffer/store.py |
| inference_demo | 18 | app.py<br>demonstration/README.md<br>demonstration/filter.py<br>demonstration/filtered/filtered_2023-07-12-14_06_31.jsonl<br>demonstration/fixed/case1.jsonl<br>demonstration/fixed/case2.jsonl<br>demonstration/fixed/case3.jsonl<br>demonstration/generator.py |
| model | 1 | llm4crs/ranking/reco_model_tool.py |
| testing_quality | 2 | tests/README.md<br>tests/single_turn_cases.md |

## Notebook / Docker / Test 线索

### Notebooks
- `preprocess/ItemCF.ipynb`
- `preprocess/movies.ipynb`
- `preprocess/prepare_amazon.ipynb`
- `preprocess/prepare_steam.ipynb`
- `preprocess/preprocess_redial.ipynb`

### Docker
- 无

### Tests
- `tests/README.md`
- `tests/single_turn_cases.md`

## 潜在数据/状态/模型/资源路径

- `assets/framework.pdf`
- `assets/framework.png`

## 目录树摘要

```text
InteRecAgent/
  assets/
  demonstration/
  eval/
  llm4crs/
  preprocess/
  tests/
  .gitignore
  README.md
  app.py
  requirements.txt
  run.sh
    framework.pdf
    framework.png
    filtered/
    fixed/
    seed/
    README.md
    filter.py
    generator.py
    seed_demos.jsonl
    seed_demos_placeholder.jsonl
    tagger.py
      filtered_2023-07-12-14_06_31.jsonl
      case1.jsonl
      case2.jsonl
      case3.jsonl
      beauty.jsonl
      games.jsonl
      movies.jsonl
    eval_simulator.sh
    eval_single_turn.sh
    one_turn_eval.py
    one_turn_eval_ranking.py
    user_simulator.py
    buffer/
    corups/
    critic/
    demo/
    mapper/
    memory/
    prompt/
    query/
    ranking/
    retrieval/
    utils/
    __init__.py
    agent.py
    agent_plan_first.py
    agent_plan_first_openai.py
    environ_variables.py
      __init__.py
      base.py
      clear.py
      store.py
      __init__.py
      base.py
      __init__.py
      base.py
      __init__.py
      base.py
      __init__.py
      map_tool.py
      __init__.py
      memory.py
      __init__.py
      critic.py
      system.py
      tool.py
      __init__.py
      query_tool.py
      __init__.py
      reco_model_tool.py
      __init__.py
      itemcf_tool.py
      sql_tool.py
      __init__.py
      exceptions.py
      open_ai.py
      prompt.py
      sql.py
      text_sim.py
      util.py
    ItemCF.ipynb
    movies.ipynb
    prepare_amazon.ipynb
    prepare_steam.ipynb
    preprocess_redial.ipynb
    README.md
    single_turn_cases.md
```

## 下一步人工确认

- 找到最小可运行命令：API、页面、CLI、worker、测试、训练或 demo 至少一个。
- 确认依赖、环境变量、数据库/数据文件、端口和外部服务。
- 确认 baseline/demo 是否能在本地、Docker、云服务器或 GPU 环境上跑通。
- 确认自己要做的面试亮点：改造点、demo、测试、报告或实验计划。

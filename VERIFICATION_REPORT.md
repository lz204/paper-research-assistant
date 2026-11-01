# 多语种文献翻译与论文研读助手 — 验证报告

> **更新日期**：2026-11-01（整理文档口径与表述；数据为 2026-09-07 / 2026-10-19 的实际运行记录）
>
> **范围**：三大痛点闭环、可复现测试矩阵、覆盖率门控、真实论文端到端演示、Agent 化改造与真实评测
>
> **口径**：① mock 可复现（无需 key）/ ② 真实 API 单次实测（需 key + 网络）/ ③ 历史实验†（仅见项目文档附录 B，不可复现）
>
> 本文所有结论均来自实际运行输出，非代码走读推断。

---

## 1. 结论摘要

| 验证项 | 结论 |
| --- | --- |
| 三大痛点（公式上下文 / 长文遗忘 / 综述横跳） | 全部闭环（mock 场景验证，`verify_pain_points.py`） |
| 可复现测试（pytest + verify + E2E mock + Agent mock） | 全部通过（mock 隔离，无费用） |
| 核心模块覆盖率 | context_manager 95.0% / fusion_engine 94.9% / citation_verifier 86.8%（门槛 85%） |
| 真实论文端到端（②） | 通过：Attention Is All You Need，15 页 / 107 chunks |
| 批量真实论文（②） | 通过：28 篇 arXiv / 917 页 / 7,726 chunks，0 失败 |
| Agent 真实评测（②，小样本） | 6/6 任务编排冒烟通过；样本小，不代表端到端文档问答质量 |
| 未验证边界 | Docker 未实机构建、无部署、无真实用户；pgvector / PG 成本 / Redis 为接口预留 |

## 2. 验证环境

- 操作系统：Windows；Python 3.13.0（实测 3.13.0；`requirements.txt` 声明 >= 3.11）
- 关键依赖（实测版本，见 `requirements.lock`）：fastapi 0.110.0 / httpx 0.27.2 / starlette 0.36.3 / pydantic 2.10.4 / PyMuPDF 1.25.5 / pdfplumber 0.11.9 / rank_bm25 0.2.2
- DashScope：`.env` 配置真实 key；真实调用仅用于 ② 档演示，测试默认 mock 隔离
- 可选依赖：spaCy（含模型）、structlog、langfuse 未安装，分别降级为纯正则 / 标准 logging / mock Langfuse

## 3. 三大痛点验证（全 mock，`python scripts/verify_pain_points.py`）

| 痛点 | 机制 | 验证结果 |
| --- | --- | --- |
| 公式上下文断裂 | FormulaBinder 四级规则绑定（段落序号 / 位置映射 / 文本锚点 / 兜底） | 模拟 VLM 位置描述 + 公式 LaTeX，成功绑定到含 attention 上下文段落，`need_review=false` |
| 万字论文「中间遗忘」 | ContextManager KV 记忆池（入库抽取实体 → 问答按意图注入） | 40 节长文，第 6 节中段实体 `ImageNet-1K` / `mAP=0.66` / `F1=0.61` 均写入 KV 池并被注入 |
| 写综述反复横跳 | ReviewPipeline 跨多文档 RAG + MMR + 引用溯源 | 3 篇文档一次生成 3 章节、3 条跨文档参考文献，正文带可溯源引用 |

> 验证过程还暴露并修复了 entity_extractor 两个抽取缺口（见第 6 节第 10 项）。

## 4. 可复现测试矩阵（①，mock 隔离，无 key 无费用）

| 测试 | 命令 | 结果 |
| --- | --- | --- |
| 语法编译 | `python -m compileall -q app tests` | 0 错误 |
| pytest 回归 | `python -m pytest`（pyproject testpaths=tests/unit） | 全绿；86 项（74 基线 + 12 记忆池 / MMR / NLI 接线用例） |
| ContextManager 集成验证 | `python tests/unit/verify_context_manager.py` | exit 0 |
| RAG 全链路验证 | `python tests/unit/verify_rag_pipeline.py` | exit 0 |
| 综述全链路验证 | `python tests/unit/verify_review_pipeline.py` | exit 0 |
| Parser 模块验证 | `python tests/unit/verify_parser_pipeline.py` | exit 0 |
| 基础设施验证（含 FastAPI TestClient） | `python tests/unit/verify_infrastructure.py` | exit 0 |
| 补齐模块验证（翻译 / GB-T 7714 / 结构拆解） | `python tests/unit/verify_completion_modules.py --mock` | 28/28 检查通过 |
| Agent 链路验证 | `python tests/unit/verify_agent_pipeline.py` | 66/66 断言通过 |
| Agent 评测（mock） | `python tests/eval/run_agent_eval.py --mode mock` | 6/6 任务；success 100% / tool selection 100% / avg 2.17 步 / ¥0 |
| MCP 协议冒烟 | `python scripts/agent_mcp_smoke.py` | initialize（2024-11-05）+ tools/list 4 工具通过 |
| E2E mock（全链路） | `python tests/e2e/test_e2e_mock.py` | exit 0 |
| API mock 全链路 | `create_app(mock_mode=True)` + TestClient | 主要端点 HTTP 200；`/ask memory_used=true, memory_items>0（实测 2~4 条）`；`/review citation_stats.total=2, dedup_stats 有值, verified=true, verification_skipped=false`；`/agent framework=langgraph`；`/agent/tools` 4 工具 |

覆盖率门控（合并 verify 脚本 + pytest 口径，`make coverage && make coverage-check`）：

```
[coverage-check] app/rag/context_manager.py:     95.0%
[coverage-check] app/parser/fusion_engine.py:    94.9%
[coverage-check] app/review/citation_verifier.py: 86.8%
✓ 核心模块覆盖率均 >= 85%
```

## 5. 真实 API 单次实测（②，需 `DASHSCOPE_API_KEY` + 网络）

### 5.1 真实论文端到端（`python scripts/demo_real_paper.py`）

- 链路：arXiv 1706.03762（Attention Is All You Need，15 页）→ PyMuPDF 抽文本（39,526 字符）→ 层级分块 107 chunks → DashScope embedding 全量入库（107/107）→ HybridRetriever（向量 + BM25 + RRF）→ Qwen-Plus 生成
- 提问与回答：WMT 2014 用了什么数据集与指标 → 英德约 450 万句对，BLEU 英德 28.4 / 英法 41.0；训练方式 → Adam（β1=0.9, β2=0.98, ε=1e-9），warmup 4000 步线性增后衰减
- 单问成本约 ¥0.003–0.004（Qwen-Plus 真实调用）

### 5.2 用户上传 28 篇真实 arXiv 批量验证（`python scripts/demo_uploaded_papers.py`）

- 28/28 篇解析成功（0 失败），合计 917 页 / 2,737,781 字符 / 7,726 chunks；单篇 107~627 chunks
- 前 4 篇实体入库 KV 池 417 个实体（metric 类为主，符合技术报告数字密集特征）
- 说明：`*.pdf` 因版权不入 git（`.gitignore`），按关键词重跑即可复现同链路

### 5.3 长文中间检索（②，真实 DeepSeek-V3，53 页 / 464 chunks）

| 问题（内容位于全文中部） | 命中 chunk | 实测回答（正确） |
| --- | --- | --- |
| MLA 相比标准 MHA 如何压缩 KV 缓存？ | 第 62 号（全文中部） | 低秩联合压缩键值 + 共享投影分发各头，仅缓存低维 c_KV 与 RoPE 增强 k_R |
| auxiliary-loss-free 负载均衡策略？ | 第 55–65 号 | 改进路由器避免辅助损失 + 序列级平衡损失 LBal 防单序列失衡 |

## 6. 验证中发现并修复的缺陷

均为「实际运行 / 真实 API 调用」才暴露，非静态走读发现：

| # | 模块 | 缺陷 | 修复 | 回归 |
| --- | --- | --- | --- | --- |
| 1 | app/rag/chunker.py | 短章节（token ≤ overlap）整节丢失，输出 0 chunk | has_own_content 追踪，含自身内容即输出 | verify_rag_pipeline |
| 2 | app/api/routes.py | `from __future__ import annotations` + 闭包内 pydantic 类 → body 被当 query（422） | 三个 Body 模型提升模块级 | verify_infrastructure / e2e mock |
| 3 | 依赖 | starlette<0.37 的 TestClient 与 httpx>=0.28 不兼容 | requirements 锁 httpx<0.28 | verify_infrastructure |
| 4 | app/review（pipeline + citation_verifier） | 引用 index 每段从 1 编号，多段综述参考文献丢条目 | 新增 renumber_results 文档级全局编号 | verify_review_pipeline |
| 5 | app/parser/vlm_client + fallback_handler | 真实二次验证在事件循环内 asyncio.run 会崩 | 新增 recognize_page_sync（运行中 loop 检测） | verify_parser_pipeline |
| 6 | app/api/routes.py | `/upload`、`/ask`、`/review` 真实编排为占位 | 打通抽文本→分块→索引→检索→LLM | pytest API 用例 |
| 7 | app/rag/retriever.py | rank_bm25 小语料 idf≈0 全 0 分，关键词召回失效 | 全 0 分回退 Jaccard 词频 | pytest test_bm25_small_corpus_fallback |
| 8 | app/rag/embedder.py | DashScope embedding 响应为 dict，按属性解析抛 AttributeError | 兼容 dict / 对象两种返回 | pytest embedder 用例 |
| 9 | app/rag/embedder.py | batch_size=25 超上限 10，批量失败 | BATCH_SIZE=10 + 字符预算分包 + 整批失败逐条降级 | 同上 |
| 10 | app/rag/entity_extractor.py | ImageNet-1K 前缀被误抽为 model；`mAP=0.81` 抽不到；BERT / ResNet-50 缺失 | 基名白名单、指标双格式、版本号 + 全大写模型 | pytest 实体抽取用例 |
| 11 | app/agent/runner.py | 模型反复调同一工具耗尽 max_steps；tool 消息缺 `tool_call_id`，DashScope 无法关联观察 | 回填 call_id；prompt 约束收敛；预算耗尽强制作答 | verify_agent_pipeline |
| 12 | app/agent/runner.py | agent 评测判据把「假工具标记」当硬断言、工具选择集合完全一致比较 | 判据改语义关键词；工具选择改 `expected ⊆ actual` | run_agent_eval --mode real 复测 |

## 7. Agent 化改造内容（2026-10-19）

> 状态：Agent 化改造已完成，数据经本人复验（2026-10-19 实际运行记录）；mock / 真实指标的口径分界见第 8 节。

| # | 模块 | 改造内容 |
| --- | --- | --- |
| 1 | Agent 核心 | `app/agent/`：types（AgentResult / AgentStep / ToolExecution 契约 v1）、tools（4 工具注册表 + OpenAI schema）、llm（DashScope Function Calling + ScriptedMockLLM）、runner（simple ReAct + 工厂）、langgraph_runner（StateGraph，异常回退 simple） |
| 2 | API | 新增 `/agent`、`/agent/tools`、`/agent/tools/call`；`/health components` 增加 agent；`/upload` 增加 `use_vlm`（最多前 10 页）并返回 `vlm_used` / `formulas` / `parse_mode` |
| 3 | 前端 | 新增第 6 个 Tab「Agent 助手」；`ApiClient` 增加 agent 接口；API 模式展示工具调用轨迹，本地进程模式用 in-process AgentRunner |
| 4 | MCP | 手写 stdio JSON-RPC（MCP 2024-11-05）server：initialize / tools/list / tools/call，转发到后端；配套协议冒烟脚本（避免 mcp 包升级 starlette） |
| 5 | VLM 可选链路 | `/upload use_vlm=true` 走转图 → Qwen-VL → FusionEngine → FormulaBinder；默认关闭控成本，失败回退文本层（parse_mode=text_fallback） |
| 6 | compose / 依赖 / CI | compose 默认只起 app + frontend，PG / Redis 加 `profiles: ["full"]`；`requirements.lock` 锁 streamlit / plotly / langgraph / langchain-core / pdfplumber；`dev.ps1` / Makefile / CI 增加 Agent 验证步骤 |

### 7.1 记忆池 / MMR / NLI 接入 API 复验（2026-11-01，mock）

| 验收项 | 实测结果 |
| --- | --- |
| `/upload` → `/ask`（use_memory=true） | `memory_used=true`、`memory_items>0`（实测 2~4 条，随上传文本实体数变化），回答附记忆池实体；未上传 doc 时安全降级为空记忆、不报错 |
| `/ask` 真实编排（注入 FakeEmbedder + RecordingLLM，不触网） | prompt 含记忆池实体 + 会话历史（最近 3 轮），`memory_items=1` |
| `/review` 默认（2 篇，verify=true） | `citation_stats.total=2`、`verified=true`、`verification_skipped=false`；`dedup_stats={skipped:2, reduce_top_k:8}`（summaries≤8 时跳过 MMR，如实计数） |
| `/review` MMR 生效（monkeypatch `REDUCE_TOP_K=1`） | `mmr_applied=true`、`calls=1`、`input=2`、`selected=1` |
| `/review` 伪造引用 `[doc_ghost, doc_ghost#9.9]` | `citation_stats.contradiction=1`；references 中该引用 `exists=false`、`label=contradiction` |
| `/review` verify_citations=false | `verified=false`、`verification_skipped=false`、`citation_stats.total=0`，不抛异常 |

> 全部为 `create_app(mock_mode=True)` + TestClient 实测（新增 12 个 pytest 用例），未调用真实 API。

## 8. Agent 验证与真实评测口径

| 验证项 | 口径 | 结果 |
| --- | --- | --- |
| Agent 链路 | ① mock | 66/66 断言：工具注册与异常降级 / simple ReAct / LangGraph 真跑 / 边界恢复 |
| Agent 评测 mock | ① mock | 6/6 任务；success 100%、tool selection 100%、avg 2.17 步、¥0；framework=langgraph |
| MCP 冒烟 | ① mock | initialize + tools/list（4 工具）通过；`--call` 需先启动后端 |
| VLM 可选链路 | ① mock | `use_vlm=true` 返回 `vlm_used=true`、`parse_mode=vlm`、`formulas=3`；默认 `use_vlm=false` 行为不变 |
| 前端 6 Tab | ① 人工冒烟 | 本地启动后端 + 前端，确认 6 Tab 顺序、API 模式识别、后端不可达自动回退本地进程模式；仓库内无自动化 UI 测试 |
| 依赖可复现 | ① dry-run | `pip install -r requirements.lock --dry-run` exit 0（pdfplumber 0.11.9 已锁定） |
| 覆盖率扩展 | ① | context_manager 95.0% / fusion_engine 94.9% / citation_verifier 86.8%；新增 app/agent：tools 80%、runner 87%、langgraph_runner 91% |
| Agent 真实评测 | ② 真实 API | qwen-plus + **固定假工具后端**，6 条脚本化任务：success 100%、tool selection 100%、avg 2.67 步、avg latency 7.9s、¥0.0156/任务；framework=langgraph |
| Agent E2E | ② 真实 API | `POST /agent`（真实 key + Attention 论文 107 chunks）：3 步（list_documents → search_chunks → answer），答案正确（WMT 2014 英德 ~450 万句对 / BLEU 28.4），tokens 4040/341，成本 ¥0.0203，耗时 15.8s |

**口径说明（防误用）**

- Agent 评测 mock 指标是**脚本化 LLM 的编排回归指标**，不代表真实模型能力；real 模式使用真实 qwen-plus，但工具是**固定假回调**（2 篇论文 / 2 个 chunk），只反映「工具选择 + 编排 + 收敛」质量，不是端到端文档问答准确率，样本小（6 任务）不具统计意义。
- mock 与真实生成器共用同一 MMR 实现（`_MMRReduceMixin`）：MMR 是否执行取决于每章节 summaries 数与 `REDUCE_TOP_K`。默认 2 篇文献时 summaries≤8，会跳过并在 `dedup_stats.skipped` 计数（不静默）；把 summaries 数调高（测试中 monkeypatch `REDUCE_TOP_K=1`）时 mock 路径同样执行 MMR（实测 `calls=1 / input=2 / selected=1`）。
- 真实评测不纳入 CI（成本原因），日常回归使用 mock；真实命令可手动执行（需 key，会产生少量费用）。
- MCP 为手写最小协议实现（initialize / tools/list / tools/call），不等同官方 mcp 包与完整 MCP 生态；`tools/call` 依赖后端已启动并持有文档索引。
- Docker Compose：当前执行环境 `docker` CLI 不可用，**未实际构建 / 启动容器**；Dockerfile 已改为优先按 `requirements.lock` 安装依赖，容器化路径需在安装 Docker 的机器上复验。
- 真实 API 支出记录（②档）：约 ¥0.66（多次评测迭代 + 真实论文 E2E + 开发期误触发），后续真实调用统一手动执行。
- 历史实验数字与本文 ①/② 指标**不可比较**，统一见项目文档附录 B（标注 †）。

## 9. 复现命令速查

```powershell
# Windows（mock，无需 key；建议先设 $env:PYTHONIOENCODING='utf-8'）
.\scripts\dev.ps1 install
.\scripts\dev.ps1 test            # verify 套件 + 补齐模块 + Agent 验证 + E2E mock
.\scripts\dev.ps1 test-pytest     # pytest 回归
.\scripts\dev.ps1 coverage        # 覆盖率 + 核心模块门控
.\scripts\dev.ps1 eval            # 评测 quick（mock）
.\scripts\dev.ps1 verify-agent
.\scripts\dev.ps1 agent-eval
.\scripts\dev.ps1 mcp-smoke
.\scripts\dev.ps1 api             # 终端 1
.\scripts\dev.ps1 frontend        # 终端 2

# Linux / macOS
make test && make test-pytest && make coverage && make coverage-check
make eval && make verify-agent && make agent-eval && make mcp-smoke
```

```bash
# ② 真实 API（需 key + 网络，会产生少量费用）
python scripts/demo_real_paper.py
python scripts/demo_uploaded_papers.py --paper deepseek --limit 1
python tests/eval/run_eval.py --mode nightly --output report.json
python tests/eval/run_agent_eval.py --mode real
```

## 10. 工程化配套（实际落地项）

| 项 | 结果 |
| --- | --- |
| CI 覆盖真实性 | unit-tests job 从 `requirements.lock` 安装完整依赖，跑 pytest + 5 个 verify + 补齐模块验证 + Agent 验证 / mock 评测 + E2E mock；nightly 跑真实评测（失败不阻断） |
| 依赖可复现 | `requirements.lock` 直接依赖精确 pin（含 pdfplumber 0.11.9 / streamlit / plotly / langgraph / langchain-core）；CI 从 lock 安装 |
| 测试体系 | `pyproject.toml` 只收 tests/unit/test_*.py，避免误收带参 verify 脚本；conftest mock 隔离 |
| 覆盖率门控 | `scripts/check_coverage.py` + `make coverage-check`（核心 3 模块 ≥85%） |
| 静态检查 | ruff / mypy 配置于 pyproject，pre-commit 配置于 `.pre-commit-config.yaml`；**未接入 CI** |
| 仓库卫生 | `.gitignore` 忽略 .env / 覆盖率产物 / PDF（版权不入库）；dev 依赖见 `requirements-dev.txt` |
| 文档一致性 | 修正与实际不符的架构 / 依赖 / 语料规模表述；补充可选依赖、未实机构建与前端验证方式的口径 |

## 11. 诚实边界

1. 真实 VLM（Qwen-VL）按页公式 / 版面识别依赖真实 key 与网络；`/upload use_vlm=true` 为可选链路且最多前 10 页，默认走文本层，mock 可演示。
2. PgvectorRetriever / PgCostTracker / Redis 为接口预留或内存版，落地需数据库服务；docker-compose 默认只起 app + frontend。
3. 前端默认为 API 模式（ApiClient 调 FastAPI）；「本地进程模式」是离线兜底，属设计决定而非缺陷；无自动化 UI 测试。
4. 评测集内置 3 篇样例，扩展至更大规模需 `make dataset` + DashScope 真实调用。
5. Agent / MCP 为可演示的最小实现，真实工具调用需 key 与少量费用；Docker 未实机构建，无部署、无真实用户反馈。

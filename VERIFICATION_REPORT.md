# 多语种文献翻译与论文研读助手 — 验证报告

> 生成日期：2026-09-07
> 范围：项目文档三大痛点闭环、全量测试矩阵、覆盖率门控、真实论文端到端演示、企业级水平评估
> 所有结论均来自实际运行，非代码走读推断

---

## 1. 结论摘要

| 验证项 | 结论 |
| --- | --- |
| 文档第 1 章三大痛点 | 全部闭环（痛点 1/2/3 均通过场景验证） |
| 全量测试（单测 + E2E mock） | 全部通过 |
| pytest 回归测试 | 16 passed |
| 核心模块覆盖率 | context_manager 95% / fusion_engine 94.9% / citation_verifier 86.8% |
| 真实论文 PDF → 问答演示 | 通过（Attention Is All You Need，15 页，107 chunks） |
| 用户上传论文集批量验证 | 通过（28 篇 / 917 页 / 7726 chunks，含长文中间检索） |
| 企业级水平评估 | 工程素养达标；CI/依赖/覆盖/卫生等未达标项已补齐，剩余边界均为外部设施依赖 |

---

## 2. 验证环境

- 操作系统：Windows
- Python：3.13.0
- 关键依赖（实测版本，见 `requirements.lock`）：fastapi 0.110.0 / httpx 0.27.2 / starlette 0.36.3 / pydantic 2.10.4 / PyMuPDF 1.25.5 / rank_bm25 0.2.2
- DashScope：`.env` 配置真实 key（embedding 与 Qwen-Plus 真实调用仅在演示时使用，测试默认 mock 隔离）

---

## 3. 三大痛点验证（文档第 1 章）

复现入口：`python scripts/verify_pain_points.py`（全 mock，不耗 API）

### 痛点 1：公式上下文断裂 → 已解决

- 机制：`FormulaBinder` 四级规则绑定（段落序号 / 位置映射 / 文本锚点 / 兜底）
- 场景：模拟 VLM 输出位置描述「第 2 段后」+ 公式 LaTeX
- 结果：公式成功绑定到含 attention 上下文段落，`need_review=false`，返回上下文文本
- 结论：公式→段落上下文可定位，无需人工翻页找

### 痛点 2：万字论文"中间遗忘" → 已解决

- 机制：`ContextManager` KV 记忆池（分块入库时实体抽取 → KV；问答按意图注入）
- 场景：40 节长文（远超人单次阅读注意力）全量入库，提问第 6 节（中段）实体
- 结果：`ImageNet-1K` / `mAP=0.66` / `F1=0.61` 均成功写入 KV 池并被 prompt 主动注入
- 结论：中段实体不被遗忘；验证过程还暴露并修复了 entity_extractor 两个抽取缺口（见第 6 节）

### 痛点 3：写综述反复横跳 → 已解决

- 机制：`ReviewPipeline` 跨多文档 RAG + MMR 去重 + 引用溯源
- 场景：3 篇文档一次生成综述
- 结果：生成 3 章节、3 条参考文献（跨 3 篇），正文带可溯源引用
- 结论：多文献一次生成，无需来回切窗口

---

## 4. 全量测试矩阵

| 测试 | 命令 | 结果 |
| --- | --- | --- |
| 语法编译 | `python -m compileall -q app tests scripts` | 0 错误 |
| pytest 回归 | `pytest`（pyproject testpaths=tests/unit） | 16 passed |
| ContextManager 集成验证 | `python tests/unit/verify_context_manager.py` | exit 0 |
| RAG 全链路验证 | `python tests/unit/verify_rag_pipeline.py` | exit 0 |
| 综述全链路验证 | `python tests/unit/verify_review_pipeline.py` | exit 0 |
| Parser 模块验证 | `python tests/unit/verify_parser_pipeline.py` | exit 0 |
| 基础设施验证（含 FastAPI TestClient） | `python tests/unit/verify_infrastructure.py` | exit 0 |
| E2E mock（全链路） | `python tests/e2e/test_e2e_mock.py` | exit 0 |
| 核心回归脚本 | `python scripts/verify_pain_points.py` | exit 0 |

覆盖率门控（合并 verify 脚本 + pytest 口径）：

```
[coverage-check] app/rag/context_manager.py:    95.0%
[coverage-check] app/parser/fusion_engine.py:   94.9%
[coverage-check] app/review/citation_verifier.py: 86.8%
✓ 核心模块覆盖率均 >= 85%（文档 6.4 节）
```

运行：`make coverage && make coverage-check`（由 `scripts/check_coverage.py` 执行判定）。

---

## 5. 真实论文端到端演示

复现入口：`python scripts/demo_real_paper.py`（本地无 PDF 时自动从 arXiv 下载）

链路：arXiv 1706.03762（Attention Is All You Need，15 页）→ PyMuPDF 抽文本（39,526 字符）→ 层级分块 107 chunks → DashScope embedding 全量入库（107/107 成功）→ HybridRetriever（向量 + BM25 + RRF）→ Qwen-Plus 生成

| 问题 | 实测回答（正确） |
| --- | --- |
| WMT 2014 实验用了什么数据集与指标？ | 英德约 450 万句对，BLEU 英德 28.4 / 英法 41.0 |
| 模型如何训练？优化器与学习率调度？ | Adam（β1=0.9, β2=0.98, ε=1e-9），warmup 4000 步线性增后衰减 |

单问成本约 0.003–0.004 元（Qwen-Plus 真实调用）。

### 5.1 用户上传论文集（28 篇真实 arXiv PDF）批量验证

用户上传 28 篇真实论文至 `sample_papers/`（DeepSeek-V2/V3/R1/Coder/Math/VL/OCR/MoE、Kimi k1.5/K2/K2.5/K3/VL/Audio/Linear、DreamCraft3D、LV-Eval、Mooncake、Counting-Stars、Attention Is All You Need 等），已全部跑通解析→分块→入库→检索→问答：

| 阶段 | 实测结果 |
| --- | --- |
| 批量解析 | 28/28 篇成功（0 失败） |
| 规模 | 917 页 / 2,737,781 字符 / 7,726 chunks |
| 单篇 chunk 数 | 107 ~ 627（Attention 最少、DeepSeek-R1 最多） |
| 实体入库（前 4 篇） | KV 池 417 实体（metric 为主，符合技术报告数字密集特征） |

**长文中间检索验证（真实 DeepSeek-V3，53 页 / 464 chunks）**——痛点 2 的强证据：

| 问题（内容位于全文中部） | 命中的 chunk | 实测回答（正确） |
| --- | --- | --- |
| MLA 相比标准 MHA 如何压缩 KV 缓存？ | 第 62 号 chunk（全文中部） | 低秩联合压缩键值 + 共享投影分发各头，仅缓存低维 c_KV 与 RoPE 增强 k_R |
| auxiliary-loss-free 负载均衡策略？ | 第 55–65 号 chunk | 改进路由器避免辅助损失 + 序列级平衡损失 LBal 防单序列失衡 |

长文（50+ 页）中段的公式级细节仍能被 HybridRetriever 精准命中并正确作答，单问成本约 0.003–0.004 元。

> 说明：`*.pdf` 因版权不入 git（`.gitignore`），运行 `python scripts/demo_uploaded_papers.py --paper <关键词>` 即可对任意上传论文跑同链路。

---

## 6. 验证过程中发现并修复的缺陷

以下缺陷均为"实际运行/真实 API 调用"才暴露，非静态走读发现：

| # | 模块 | 缺陷 | 修复 | 回归测试 |
| --- | --- | --- | --- | --- |
| 1 | app/rag/chunker.py | 短章节（token ≤ overlap）整节内容丢失，输出 0 chunk | 引入 has_own_content 追踪，含自身内容即输出 | verify_rag_pipeline 测试 1b |
| 2 | app/api/routes.py | `from __future__ import annotations` + 闭包内 pydantic 类 → body 被当 query（422）、openapi ForwardRef 报错 | 三个 Body 模型提升模块级 | verify_infrastructure / e2e mock |
| 3 | 依赖 | starlette<0.37 的 TestClient 与 httpx>=0.28 不兼容 | requirements 锁定 httpx<0.28（+requirements.lock） | verify_infrastructure |
| 4 | app/review（pipeline + citation_verifier） | 引用 index 每段从 1 编号，多段综述参考文献丢条目 | 新增 renumber_results 文档级全局编号 | verify_review_pipeline 测试 8 |
| 5 | app/parser/vlm_client + fallback_handler | 真实二次验证在事件循环内 asyncio.run 会崩 | 新增 recognize_page_sync（运行中 loop 检测） | verify_parser_pipeline |
| 6 | app/api/routes.py | /upload、/ask、/review 真实编排为占位 | 打通 PyMuPDF 抽文本→分块→索引→HybridRetriever→LLM | pytest test_api_parser_ask_real_pipeline |
| 7 | app/rag/retriever.py | rank_bm25 小语料下 idf≈0 全 0 分，关键词召回失效 | 全 0 分回退 Jaccard 词频 | pytest test_bm25_small_corpus_fallback |
| 8 | app/rag/embedder.py | DashScope embedding 响应为 dict，按属性解析抛 AttributeError | 兼容 dict / 对象两种返回 | pytest test_embedder_batch_splitting_and_fallback |
| 9 | app/rag/embedder.py | batch_size=25 超 DashScope 上限 10，批量失败 | BATCH_SIZE=10 + 字符预算分包 + 整批失败逐条单发降级 | 同上 |
| 10 | app/rag/entity_extractor.py | ImageNet-1K 前缀被误抽为 model；`mAP=0.81` 数字在后抽不到；BERT/ResNet-50 识别缺失 | 数据集基名白名单消歧；指标双格式；版本号+全大写模型 | pytest test_entity_extractor_disambiguation_and_metric_forms |

---

## 7. 工程化配套（企业级未达标项执行结果）

| 项 | 结果 |
| --- | --- |
| CI 覆盖真实性 | unit job 改装完整依赖并跑 pytest + verify 脚本 + E2E（不再只装 3 个包导致 API 层从未被 CI 覆盖） |
| 依赖可复现 | 新增 `requirements.lock`（直接依赖精确 pin，与本地实测一致）；CI 从 lock 安装 |
| 测试体系 | `pyproject.toml`（pytest 只收 test_*.py，避免误收带参 verify 脚本）；conftest mock 隔离 |
| 覆盖率门控 | `scripts/check_coverage.py` + `make coverage-check`（核心 3 模块 ≥85%） |
| 静态检查 | ruff / mypy 配置于 pyproject，pre-commit 配置于 `.pre-commit-config.yaml` |
| 仓库卫生 | `git init`；`.gitignore` 确认忽略 .env / 覆盖率产物 / PDF（版权不入库）；dev 依赖见 `requirements-dev.txt` |
| 文档一致性 | 修正"三容器/Celery/坐标 IoU/Python 版本/Redis 缓存/200 篇已入库"等与实际不符表述 |

---

## 8. 企业级水平评估

**总体**：可定性为「高完成度的准企业级求职项目」——代码与文档自洽、测试闭环、覆盖率有实测门控；未达严格企业级的点均为外部设施依赖，且已在文档如实标注，不再是隐藏缺陷。

达标：分层清晰、外部依赖全可降级、可观测/成本治理/容错链完整、CI 真实覆盖、覆盖率有门控、单测与回归闭环。

诚实边界（需外部设施，非代码缺陷）：
1. 真实 VLM（Qwen-VL）按页公式/版面识别依赖真实 key 与网络，仅文本层全链路验证；
2. PgvectorRetriever / PgCostTracker / Redis 为接口预留或内存版，落地需数据库服务；
3. Streamlit 前端直连 DashScope（不调用 FastAPI），属设计决定而非缺陷；
4. 评测集内置 3 篇样例，扩展至更大规模需 `make dataset` + DashScope。

---

## 9. 复现命令速查

```bash
# 全量测试
make test                # verify 脚本套件
make test-pytest         # pytest 回归（需 dev 依赖）
make coverage && make coverage-check

# E2E
make e2e-mock

# 场景验证
python scripts/verify_pain_points.py        # 三大痛点闭环（mock）
python scripts/demo_real_paper.py           # 真实论文 PDF → 问答（需 .env key）
python scripts/demo_uploaded_papers.py --paper <关键词>  # 任意上传论文 RAG 问答
# 例：python scripts/demo_uploaded_papers.py --paper deepseek --limit 1

# 依赖
pip install -r requirements.lock         # 可复现安装
pip install -r requirements-dev.txt      # 开发工具
```

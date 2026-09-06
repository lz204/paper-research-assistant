"""基础设施模块集成验证脚本（全 mock，无 API Key）。

验证：llm / observability / export / review.pipeline / api 共 7 大核心模块。
运行：python tests/unit/verify_infrastructure.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 共享引导：插入项目根目录到 sys.path + 清空 API key / Langfuse 配置（强制 mock）；
# 两篇样例论文语料与分块/文献信息构建函数统一来自 _verify_common（与 verify_review_pipeline 共用）
from _verify_common import (
    PROJECT_ROOT,
    build_doc_chunks as _build_doc_chunks,
    build_doc_infos as _build_doc_infos,
)  # noqa: E402

from app.export import (  # noqa: E402
    DocxRenderResult,
    MarkdownDocxRenderer,
    PythonDocxRenderer,
    ReviewDocument,
    create_docx_renderer,
)
from app.llm import (  # noqa: E402
    BaseLLMClient,
    DashScopeLLMClient,
    LLMResponse,
    MockLLMClient,
    QACache,
    create_llm_client,
    estimate_cost_yuan,
)
from app.observability import (  # noqa: E402
    BaseCostTracker,
    BaseLangfuseClient,
    CostDashboard,
    CostRecord,
    InMemoryCostTracker,
    LogContext,
    MockLangfuseClient,
    SpanRecord,
    StdJsonLogger,
    StructlogLogger,
    StructuredLogger,
    TraceRecord,
    create_cost_tracker,
    create_langfuse_client,
    create_logger,
    get_cost_tracker,
    get_current_trace_id,
    get_langfuse_client,
    get_logger,
    new_trace_id,
    record_cost,
    set_cost_tracker,
    set_current_trace_id,
)
from app.rag import (  # noqa: E402
    FakeEmbedder,
    InMemoryVectorRetriever,
)
from app.review import (  # noqa: E402
    CitationVerifier,
    ReviewPipeline,
    create_review_pipeline,
)
from app.api import (  # noqa: E402
    AskRequest,
    AskResponse,
    HealthResponse,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    MAX_QUESTION_LEN,
    MAX_REVIEW_TOPIC_LEN,
    ReviewRequest,
    ReviewResponse,
    UploadRequest,
    UploadResponse,
    create_app,
    get_app,
)

# ---------------------------------------------------------------------------
# 测试 1：LLM 客户端（mock 模式）
# ---------------------------------------------------------------------------


def test_llm_client() -> None:
    """测试 1：LLM 客户端封装（重试+缓存+降级，文档 6.2/6.3 节）"""
    print("\n" + "=" * 70)
    print("测试 1：LLM 客户端封装（mock 模式）")
    print("=" * 70)

    # MockLLMClient
    mock = MockLLMClient()
    resp = mock.chat("什么是 Transformer？")
    print(f"MockLLMClient.chat 返回: ok={resp.ok}, model={resp.model}")
    print(f"  text: {resp.text[:60]}...")
    print(f"  tokens: input={resp.input_tokens}, output={resp.output_tokens}")
    assert resp.ok, "Mock chat 应成功"
    assert resp.model == "mock-qwen-plus"
    assert "Mock LLM 回复" in resp.text
    print("  ✓ MockLLMClient 正常工作")

    # DashScopeLLMClient 无 API key → 降级 Mock
    ds = DashScopeLLMClient()
    assert ds._mock_mode is True, "无 API key 应进入 mock 模式"
    resp2 = ds.chat("测试问题")
    assert resp2.ok, "降级 Mock 后应正常返回"
    print(f"  ✓ DashScopeLLMClient 无 API key 降级 Mock")

    # 工厂函数 auto 模式
    auto = create_llm_client("auto")
    assert isinstance(auto, MockLLMClient), "auto 无 key 应降级 Mock"
    print(f"  ✓ create_llm_client('auto') 降级为 {type(auto).__name__}")

    # QA 缓存
    cache = QACache()
    cache.put(None, "测试", 0.7, 1024, LLMResponse(text="缓存答案", model="qwen-plus"))
    cached = cache.get(None, "测试", 0.7, 1024)
    assert cached is not None, "缓存应命中"
    assert cached.from_cache is True, "from_cache 应为 True"
    assert cached.cost_yuan == 0.0, "缓存命中不计费"
    print(f"  ✓ QACache 命中（from_cache={cached.from_cache}, cost={cached.cost_yuan}）")

    # 成本估算
    cost = estimate_cost_yuan("qwen-plus", 1000, 500)
    assert cost > 0, "qwen-plus 成本应大于 0"
    print(f"  ✓ estimate_cost_yuan: qwen-plus 1000+500 tokens = {cost:.4f}元")


# ---------------------------------------------------------------------------
# 测试 2：结构化日志（文档 6.1 节）
# ---------------------------------------------------------------------------


def test_logger() -> None:
    """测试 2：结构化日志（structlog / StdJson 降级，文档 6.1 节）"""
    print("\n" + "=" * 70)
    print("测试 2：结构化日志（文档 6.1 节）")
    print("=" * 70)

    # 创建 logger（自动选择 structlog 或 StdJson 降级）
    log = create_logger(name="test_infra", trace_id="trace_test_1")
    assert isinstance(log, StructuredLogger)

    # 输出各级别日志
    log.info("测试 INFO 日志", step="test.logger", latency_ms=10, status="ok")
    log.warning("测试 WARNING 日志", step="test.logger", status="warn")
    log.error("测试 ERROR 日志", step="test.logger", status="error")
    print("  ✓ 三级别日志输出成功（含文档 6.1 节关键字段 trace_id/step/latency_ms/status）")

    # bind 链式绑定
    log2 = log.bind(user_id="u1", paper_id="p1")
    log2.info("bind 后的日志应带上 user_id 和 paper_id")
    print("  ✓ bind 链式绑定上下文（user_id / paper_id）")

    # LogContext 上下文管理器
    with LogContext(user_id="u_ctx", paper_id="p_ctx") as ctx:
        tid = get_current_trace_id()
        assert tid == ctx.trace_id, "上下文中 trace_id 应已设置"
        ctx.logger.info("上下文中的日志应自动带上 trace_id/user_id/paper_id")
    print(f"  ✓ LogContext 上下文管理器（trace_id={ctx.trace_id[:8]}...）")

    # trace_id 生成
    tid1 = new_trace_id()
    tid2 = new_trace_id()
    assert tid1 != tid2, "new_trace_id 应返回不同 ID"
    assert len(tid1) == 16, "trace_id 长度应为 16"
    print(f"  ✓ new_trace_id 生成唯一 ID（{tid1} != {tid2}）")

    # 全局 logger 单例
    g1 = get_logger()
    g2 = get_logger()
    assert g1 is g2, "get_logger 应返回单例"
    print("  ✓ get_logger 单例")


# ---------------------------------------------------------------------------
# 测试 3：成本追踪（文档 6.1/6.2 节）
# ---------------------------------------------------------------------------


def test_cost_tracker() -> None:
    """测试 3：API 成本追踪 + 看板（文档 6.1/6.2 节）"""
    print("\n" + "=" * 70)
    print("测试 3：API 成本追踪 + 看板（文档 6.1/6.2 节）")
    print("=" * 70)

    # 用独立 tracker 避免污染全局
    tracker = InMemoryCostTracker()

    # 记录 3 次调用
    from datetime import datetime
    tracker.record(CostRecord(
        ts=datetime.now(), model="qwen-plus", step="review.generate",
        input_tokens=500, output_tokens=200, cost_yuan=0.0044,
        trace_id="t1", user_id="u1", paper_id="p1", from_cache=False,
    ))
    tracker.record(CostRecord(
        ts=datetime.now(), model="qwen-plus", step="review.generate",
        input_tokens=500, output_tokens=200, cost_yuan=0.0044,
        trace_id="t1", user_id="u1", paper_id="p1", from_cache=True,
    ))
    tracker.record(CostRecord(
        ts=datetime.now(), model="text-embedding-v3", step="rag.embed",
        input_tokens=1000, output_tokens=0, cost_yuan=0.0001,
        trace_id="t1", user_id="u2", paper_id="p2", from_cache=False,
    ))
    assert len(tracker) == 3, "应有 3 条记录"
    print(f"  ✓ 记录 3 次调用（{len(tracker)} 条）")

    # 查询过滤
    user1_records = tracker.get_records(user_id="u1")
    assert len(user1_records) == 2, "u1 应有 2 条"
    print(f"  ✓ 按用户过滤：u1 有 {len(user1_records)} 条")

    # 按天聚合
    summary = tracker.summary()
    assert len(summary) == 1, "应有 1 天数据"
    today_key = list(summary.keys())[0]
    assert summary[today_key]["cost"] > 0
    assert summary[today_key]["calls"] == 3
    print(f"  ✓ 按天聚合：{today_key} = {summary[today_key]['cost']:.4f}元 / 3 调用")

    # 看板
    dashboard = CostDashboard(tracker)
    by_model = dashboard.cost_by_model()
    assert "qwen-plus" in by_model
    assert "text-embedding-v3" in by_model
    print(f"  ✓ 按模型聚合: {list(by_model.keys())}")

    by_user = dashboard.cost_by_user()
    assert "u1" in by_user
    print(f"  ✓ 按用户聚合: {list(by_user.keys())}")

    by_paper = dashboard.cost_by_paper()
    assert "p1" in by_paper
    print(f"  ✓ 按论文聚合: {list(by_paper.keys())}")

    # 缓存命中率
    rate = dashboard.cache_hit_rate()
    assert rate["total"] == 3
    assert rate["hits"] == 1
    assert abs(rate["rate"] - 1/3) < 0.01
    print(f"  ✓ 缓存命中率: {rate['rate']:.1%} ({rate['hits']}/{rate['total']})")

    # 文本看板
    text_board = dashboard.render_text()
    assert "成本看板" in text_board
    print("  ✓ render_text 文本看板生成")

    # 工厂函数
    auto_tracker = create_cost_tracker("auto")
    assert isinstance(auto_tracker, InMemoryCostTracker)
    print(f"  ✓ create_cost_tracker('auto') -> {type(auto_tracker).__name__}")

    # 便捷函数
    set_cost_tracker(InMemoryCostTracker())
    record_cost(
        model="qwen-plus", step="test",
        input_tokens=100, output_tokens=50, cost_yuan=0.001,
    )
    global_tracker = get_cost_tracker()
    assert len(global_tracker) == 1
    print("  ✓ 全局便捷函数 record_cost / get_cost_tracker")


# ---------------------------------------------------------------------------
# 测试 4：Langfuse 调用链埋点（文档 6.1 节）
# ---------------------------------------------------------------------------


def test_langfuse() -> None:
    """测试 4：Langfuse LLM 调用链埋点（文档 6.1 节）"""
    print("\n" + "=" * 70)
    print("测试 4：Langfuse LLM 调用链埋点（文档 6.1 节）")
    print("=" * 70)

    # MockLangfuseClient
    mock = MockLangfuseClient()
    trace = mock.start_trace(
        name="test.trace",
        user_id="u1",
        session_id="s1",
        paper_id="p1",
    )
    assert isinstance(trace, TraceRecord)
    assert trace.trace_id
    print(f"  ✓ start_trace: id={trace.trace_id}, name={trace.name}")

    # span 上下文管理器
    with mock.span(trace, "rag.retrieve", input={"query": "测试"}) as span:
        span.output = {"results": []}
        span.metadata["top_k"] = 5
    assert len(trace.spans) == 1
    assert trace.spans[0].status == "ok"
    assert trace.spans[0].latency_ms >= 0
    print(f"  ✓ span: name={trace.spans[0].name}, latency_ms={trace.spans[0].latency_ms}")

    # span 异常处理
    try:
        with mock.span(trace, "error.step") as err_span:
            raise ValueError("模拟错误")
    except ValueError:
        pass
    assert len(trace.spans) == 2
    assert trace.spans[1].status == "error"
    assert "模拟错误" in trace.spans[1].error
    print(f"  ✓ span 异常捕获: status={trace.spans[1].status}")

    # end_trace（控制台打印）
    mock.end_trace(trace)
    assert len(mock.traces) == 1
    print(f"  ✓ end_trace 上报（控制台已打印 trace JSON）")

    # 累计统计
    assert trace.total_cost_yuan == 0.0  # 未设置 cost
    print(f"  ✓ total_latency_ms={trace.total_latency_ms}, total_cost_yuan={trace.total_cost_yuan}")

    # 工厂函数 auto 模式
    auto = create_langfuse_client("auto")
    assert isinstance(auto, MockLangfuseClient), "auto 无 key 应降级 Mock"
    print(f"  ✓ create_langfuse_client('auto') 降级为 {type(auto).__name__}")

    # 全局单例
    g1 = get_langfuse_client()
    g2 = get_langfuse_client()
    assert g1 is g2, "get_langfuse_client 应返回单例"
    print("  ✓ get_langfuse_client 单例")


# ---------------------------------------------------------------------------
# 测试 5：Word 导出（文档第 2/3/10 章）
# ---------------------------------------------------------------------------


def test_docx_renderer() -> None:
    """测试 5：Word 导出（python-docx / Markdown 降级）"""
    print("\n" + "=" * 70)
    print("测试 5：Word 导出（文档第 2/3/10 章）")
    print("=" * 70)

    # 构造一个最小 ReviewDocument
    doc = ReviewDocument(
        topic="Transformer 在医学影像分割的应用",
        title="综述：Transformer 在医学影像分割的应用",
        outline=[
            {"section_id": "s0", "title": "研究背景", "description": "..."},
            {"section_id": "s1", "title": "核心方法", "description": "..."},
        ],
        paragraphs=[
            {
                "section_id": "s0",
                "text": "Transformer 在医学影像分割已有较多研究 [paper_1, paper_1#0.1]。",
                "rendered_text": "Transformer 在医学影像分割已有较多研究 [1]。",
                "rendered_html": "<p>...</p>",
                "citations": [
                    {"index": 1, "doc_id": "paper_1", "chunk_id": "paper_1#0.1", "label": "entailment"},
                ],
                "from_cache": False,
            },
            {
                "section_id": "s1",
                "text": "核心方法包括 Attention U-Net [paper_2, paper_2#0.0]。",
                "rendered_text": "核心方法包括 Attention U-Net [2]。",
                "rendered_html": "<p>...</p>",
                "citations": [
                    {"index": 2, "doc_id": "paper_2", "chunk_id": "paper_2#0.0", "label": "entailment"},
                ],
                "from_cache": False,
            },
        ],
        verified=True,
        full_text="综述全文...",
        full_html="<html>...</html>",
        citation_stats={"entailment": 2, "contradiction": 0, "neutral": 0, "total": 2},
        references=[
            {"index": 1, "doc_id": "paper_1", "chunk_id": "paper_1#0.1", "label": "entailment"},
            {"index": 2, "doc_id": "paper_2", "chunk_id": "paper_2#0.0", "label": "entailment"},
        ],
    )

    # Markdown 降级渲染（保证可跑）
    md_renderer = MarkdownDocxRenderer()
    result = md_renderer.render(doc)
    assert result.mode == "markdown"
    assert "# 综述" in result.markdown
    assert "## 研究背景" in result.markdown
    assert "## 参考文献" in result.markdown
    print(f"  ✓ Markdown 渲染成功（{result.size} 字节）")
    print(f"    模式: {result.mode}")

    # python-docx 渲染（可能降级）
    docx_renderer = PythonDocxRenderer()
    result2 = docx_renderer.render(doc)
    if docx_renderer.available:
        assert result2.mode == "docx"
        assert result2.size > 0
        print(f"  ✓ python-docx 渲染成功（{result2.size} 字节，模式={result2.mode}）")
    else:
        assert result2.mode == "markdown"
        print(f"  ✓ python-docx 不可用，降级为 Markdown（{result2.size} 字节）")

    # 工厂函数 auto
    auto = create_docx_renderer("auto")
    print(f"  ✓ create_docx_renderer('auto') -> {type(auto).__name__}")

    # 保存到文件
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test_review.md")
        result3 = auto.render(doc, out_path)
        assert result3.path == out_path
        assert Path(out_path).exists()
        print(f"  ✓ 保存到文件: {out_path}")


# ---------------------------------------------------------------------------
# 测试 6：综述主流程编排（文档 4.3 节 + 第 3 章架构图）
# ---------------------------------------------------------------------------


def test_review_pipeline() -> None:
    """测试 6：综述主流程编排（大纲→生成→验证→ReviewDocument）"""
    print("\n" + "=" * 70)
    print("测试 6：综述主流程编排（文档 4.3 节 + 6.1 节可观测性）")
    print("=" * 70)

    doc_chunks = _build_doc_chunks()
    doc_infos = _build_doc_infos()

    # 构造检索器（用于引用验证）
    embedder = FakeEmbedder(dim=64)
    retriever = InMemoryVectorRetriever(embedder)
    all_chunks = []
    for chunks in doc_chunks.values():
        all_chunks.extend(chunks)
    retriever.add_chunks(all_chunks)

    # 工厂函数创建 pipeline
    pipeline = create_review_pipeline(
        retriever=retriever,
        embedder=embedder,
        mode="auto",
        verify_citations=True,
    )
    assert isinstance(pipeline, ReviewPipeline)
    print(f"  ✓ create_review_pipeline 创建 ReviewPipeline")

    # 运行主流程
    t0 = time.time()
    document = pipeline.run(
        topic="深度学习模型在视觉与语言任务中的应用",
        doc_infos=doc_infos,
        doc_chunks=doc_chunks,
        num_sections=3,
        verify_citations=True,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"  ✓ 主流程运行完成（{elapsed_ms}ms）")
    print(f"    主题: {document.topic}")
    print(f"    标题: {document.title}")
    print(f"    章节数: {len(document.outline)}")
    print(f"    段落数: {len(document.paragraphs)}")
    print(f"    引用统计: {document.citation_stats}")
    print(f"    参考文献数: {len(document.references)}")
    print(f"    verified: {document.verified}")

    assert document.topic == "深度学习模型在视觉与语言任务中的应用"
    assert len(document.outline) == 3
    assert len(document.paragraphs) == 3
    assert document.verified is True
    assert document.full_text  # 非空
    assert "total" in document.citation_stats

    # 元数据
    assert document.metadata["completed"] is True
    assert document.metadata["num_sections"] == 3
    print(f"  ✓ ReviewDocument 字段完整")
    print(f"    metadata: completed={document.metadata['completed']}, "
          f"cache_hits={document.metadata['cache_hits']}, "
          f"cache_misses={document.metadata['cache_misses']}")

    # 跳过验证模式
    pipeline2 = create_review_pipeline(
        retriever=None, embedder=None, mode="auto", verify_citations=False,
    )
    doc2 = pipeline2.run(
        topic="测试无验证",
        doc_infos=doc_infos,
        doc_chunks=doc_chunks,
        num_sections=2,
        verify_citations=False,
    )
    assert doc2.verified is False
    assert doc2.full_text
    print(f"  ✓ 跳过引用验证模式（verified={doc2.verified}）")

    # 可观测性：检查成本追踪
    cost_tracker = get_cost_tracker()
    cost_records = cost_tracker.get_records(step="review.pipeline")
    assert len(cost_records) >= 1, "应至少记录 1 次 pipeline 调用"
    print(f"  ✓ 成本追踪已记录 pipeline 调用（{len(cost_records)} 条）")


# ---------------------------------------------------------------------------
# 测试 7：FastAPI 路由（文档第 3/10 章）
# ---------------------------------------------------------------------------


def test_api_routes() -> None:
    """测试 7：FastAPI 路由 schema + mock 响应"""
    print("\n" + "=" * 70)
    print("测试 7：FastAPI 路由（文档第 3/10 章 + 6.5 节安全限制）")
    print("=" * 70)

    # 文档 6.5 节安全限制常量
    assert MAX_QUESTION_LEN == 2000
    assert MAX_REVIEW_TOPIC_LEN == 500
    assert MAX_PDF_PAGES == 200
    assert MAX_PDF_SIZE_MB == 50
    print(f"  ✓ 文档 6.5 节安全限制: "
          f"question={MAX_QUESTION_LEN}, "
          f"topic={MAX_REVIEW_TOPIC_LEN}, "
          f"pdf_pages={MAX_PDF_PAGES}, "
          f"pdf_size={MAX_PDF_SIZE_MB}MB")

    # 请求/响应 schema
    upload_req = UploadRequest(filename="test.pdf", file_b64="", user_id="u1")
    assert upload_req.filename == "test.pdf"
    print(f"  ✓ UploadRequest: filename={upload_req.filename}")

    ask_req = AskRequest(paper_id="p1", question="什么是 ResNet？", user_id="u1")
    assert ask_req.question == "什么是 ResNet？"
    print(f"  ✓ AskRequest: question={ask_req.question}")

    review_req = ReviewRequest(
        topic="Transformer 综述",
        doc_ids=["p1", "p2"],
        num_sections=4,
        verify_citations=True,
    )
    assert review_req.topic == "Transformer 综述"
    assert review_req.doc_ids == ["p1", "p2"]
    print(f"  ✓ ReviewRequest: topic={review_req.topic}, doc_ids={review_req.doc_ids}")

    # 创建 FastAPI 应用（fastapi 可能未安装）
    app = create_app(mock_mode=True)
    if app is None:
        print("  ⚠ fastapi 未安装，跳过路由集成测试（schema 测试通过）")
        return

    print(f"  ✓ FastAPI 应用创建成功: title={app.title}")

    # 用 TestClient 测试端点
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  ⚠ fastapi TestClient 不可用，跳过端点测试")
        return

    client = TestClient(app)

    # /health
    resp = client.get("/health")
    assert resp.status_code == 200
    health = resp.json()
    assert health["status"] == "ok"
    assert health["mock_mode"] is True
    print(f"  ✓ GET /health: status={health['status']}, mock_mode={health['mock_mode']}")

    # /ask（mock 模式）
    resp = client.post("/ask", json={
        "paper_id": "p1",
        "question": "什么是 Transformer？",
        "user_id": "u1",
    })
    assert resp.status_code == 200
    ask_resp = resp.json()
    assert ask_resp["error"] == ""
    assert "Mock 回答" in ask_resp["answer"]
    print(f"  ✓ POST /ask: answer='{ask_resp['answer'][:50]}...'")

    # /ask 超长问题（文档 6.5 节 2000 字限制）
    resp = client.post("/ask", json={
        "paper_id": "p1",
        "question": "x" * (MAX_QUESTION_LEN + 1),
    })
    assert resp.status_code == 200
    assert "超过限制" in resp.json()["error"]
    print(f"  ✓ POST /ask 超长问题被拒（文档 6.5 节）")

    # /review 超长主题（文档 6.5 节 500 字限制）
    resp = client.post("/review", json={
        "topic": "x" * (MAX_REVIEW_TOPIC_LEN + 1),
        "doc_ids": ["p1"],
    })
    assert resp.status_code == 200
    assert "超过限制" in resp.json()["error"]
    print(f"  ✓ POST /review 超长主题被拒（文档 6.5 节）")

    # /upload（mock 模式）
    resp = client.post("/upload", json={
        "filename": "test.pdf",
        "file_b64": "",
        "user_id": "u1",
    })
    assert resp.status_code == 200
    upload_resp = resp.json()
    # 空 file_b64 → doc_id 为 "doc_empty"
    assert upload_resp["doc_id"] == "doc_empty"
    print(f"  ✓ POST /upload: doc_id={upload_resp['doc_id']}")

    # 全局单例
    app2 = get_app()
    assert app2 is not None
    print(f"  ✓ get_app 全局单例")


# ---------------------------------------------------------------------------
# 测试 8：app 顶层包导入（确认所有子包可正常 import）
# ---------------------------------------------------------------------------


def test_app_package_import() -> None:
    """测试 8：app 顶层包导入完整性"""
    print("\n" + "=" * 70)
    print("测试 8：app 顶层包导入完整性")
    print("=" * 70)

    import app
    expected_submodules = {"parser", "rag", "review", "llm", "observability", "export", "api"}
    for name in expected_submodules:
        assert hasattr(app, name), f"app.{name} 应可访问"
        sub = getattr(app, name)
        print(f"  ✓ app.{name}: {type(sub).__name__}")

    # 验证跨模块组合可用
    from app.llm import create_llm_client
    from app.observability import create_logger
    from app.export import create_docx_renderer
    from app.review import create_review_pipeline
    from app.api import create_app
    print("  ✓ 跨模块组合 import 全部成功")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("基础设施模块集成验证（项目文档 6.1-6.5 节 + 第 3/10 章）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)
    print("模式：mock（无 API key，本地联调）")

    t0 = time.time()

    # 逐项测试
    test_llm_client()
    test_logger()
    test_cost_tracker()
    test_langfuse()
    test_docx_renderer()
    test_review_pipeline()
    test_api_routes()
    test_app_package_import()

    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"\n{'=' * 70}")
    print(f"全部测试通过 ✓  总耗时: {elapsed_ms}ms")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

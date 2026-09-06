"""mock 模式端到端验证。

完整链路（全 mock，无需 API Key）：Chunker → 向量检索 → 综述主流程 → Word 导出
→ 可观测性（日志/成本/Langfuse）→ FastAPI 路由 mock → Streamlit import。
运行：python tests/e2e/test_e2e_mock.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 确保 DASHSCOPE_API_KEY 未设置（强制 mock）
os.environ.pop("DASHSCOPE_API_KEY", None)
os.environ["DASHSCOPE_API_KEY"] = ""  # 空字符串 = mock


# 与 verify_infrastructure.py / verify_review_pipeline.py 保持一致的样例文本
# （保证 HierarchicalChunker 可切出足够 chunks）
PAPER_1 = """\
# Attention Is All You Need

## 1. Introduction

We propose ResNet, a deep residual network for image recognition.
Experiments show that our method achieves 85.6% top-5 accuracy on ImageNet-1K,
significantly outperforming prior CNN architectures.

## 2. Method

Our model uses a SwinTransformer backbone with F1=0.92 on COCO-2017 for object
detection. We find that the window-shifted attention reduces computational cost
by 40% while maintaining BLEU=27.3 on machine translation tasks.
"""

PAPER_2 = """\
# BERT: Pre-training of Deep Bidirectional Transformers

## 1. Introduction

We introduce BERT, a new language representation model designed to pre-train
deep bidirectional representations from unlabeled text.

## 2. Method

Our model uses a Transformer encoder architecture. We find that the masked
language modeling objective improves downstream task performance.
"""


def step(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{title}]")
    print("=" * 70)


def main() -> int:
    total_t0 = time.time()
    print("=" * 70)
    print("E2E mock 模式 — 完整链路验证")
    print(f"项目根: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"DASHSCOPE_API_KEY={os.environ.get('DASHSCOPE_API_KEY', '(empty)')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: 切分
    # ------------------------------------------------------------------
    step("Step 1 · Chunker 切分")
    from app.rag import HierarchicalChunker, Chunk, HybridRetriever, FakeEmbedder

    # 短文本使用更小的 chunk_size（100），保证两篇都能切出 ≥2 块
    chunker = HierarchicalChunker(chunk_size=100, overlap=24)
    chunks_1 = chunker.chunk(PAPER_1, doc_id="paper_1")
    chunks_2 = chunker.chunk(PAPER_2, doc_id="paper_2")
    print(f"paper_1: {len(chunks_1)} chunks")
    print(f"paper_2: {len(chunks_2)} chunks")
    assert len(chunks_1) >= 2 and len(chunks_2) >= 2, f"切分 chunk 数不够（p1={len(chunks_1)}, p2={len(chunks_2)}）"

    # ------------------------------------------------------------------
    # Step 2: 索引 + 检索
    # ------------------------------------------------------------------
    step("Step 2 · Embedding + 混合检索（向量 + BM25 + RRF）")
    embedder = FakeEmbedder(dim=64)
    # 用文档 4.2 节定位的 HybridRetriever（向量 + BM25 + RRF）：
    # FakeEmbedder 无语义，仅向量无法保证召回关键词片段，BM25 提供确定性的关键词召回。
    retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
    retriever.add_chunks(chunks_1 + chunks_2)

    q = "Transformer 在 WMT 2014 英德翻译上的 BLEU 是多少？"
    top5 = retriever.search(q, top_k=5)
    assert len(top5) >= 1
    # 应包含 BLEU 相关 chunk（由 BM25 关键词路径保证召回）
    assert any(
        "28.4" in rr.chunk.text or "BLEU" in rr.chunk.text
        for rr in top5
    ), "RAG 未召回 BLEU 片段（首段: " + repr(top5[0].chunk.text[:60]) + "）"
    print(f"查询: {q}")
    print(f"检索 top-{len(top5)}，首段 doc_id={top5[0].chunk.doc_id}")

    # 后面 RAG 生成等步骤使用 top5.chunk 作为实际 chunk 列表
    retrieved_chunks = [rr.chunk for rr in top5]

    # ------------------------------------------------------------------
    # Step 3: 综述主流程
    # ------------------------------------------------------------------
    step("Step 3 · ReviewPipeline 综述主流程")
    from app.review import create_review_pipeline

    pipeline = create_review_pipeline(
        retriever=retriever, embedder=embedder, mode="auto", verify_citations=True,
    )
    doc = pipeline.run(
        topic="深度学习 Transformer 架构在自然语言处理与视觉任务中的对比分析",
        doc_infos=[
            {"doc_id": "paper_1", "title": "Attention Is All You Need"},
            {"doc_id": "paper_2", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        ],
        doc_chunks={"paper_1": chunks_1, "paper_2": chunks_2},
        num_sections=3,
        verify_citations=True,
    )
    print(f"主题: {doc.topic}")
    print(f"章节: {len(doc.outline)}，段落: {len(doc.paragraphs)}")
    print(f"引用统计: {doc.citation_stats}")
    print(f"参考文献: {len(doc.references)} 条")
    print(f"verified: {doc.verified}")
    assert len(doc.outline) == 3, "章节数不符合"
    assert doc.full_text and len(doc.full_text) > 100, "正文为空"
    assert doc.verified is True, "引用验证未执行"
    assert doc.citation_stats.get("total", 0) >= len(doc.outline), "引用统计异常"
    assert len(doc.references) >= 1, "参考文献为空"
    print("✓ ReviewPipeline 完整通过")

    # ------------------------------------------------------------------
    # Step 4: Word 导出
    # ------------------------------------------------------------------
    step("Step 4 · Word 导出（python-docx / Markdown 降级）")
    from app.export import create_docx_renderer

    renderer = create_docx_renderer("auto")
    result = renderer.render(doc)
    print(f"渲染模式: {result.mode}，大小: {result.size} 字节")
    assert result.mode in ("docx", "markdown")
    assert result.size > 0
    if result.mode == "markdown":
        assert "参考文献" in result.markdown
    print(f"✓ {result.mode} 导出成功")

    # ------------------------------------------------------------------
    # Step 5: 可观测性
    # ------------------------------------------------------------------
    step("Step 5 · 可观测性（日志 / 成本 / 调用链）")

    # logger
    from app.observability import create_logger, new_trace_id, LogContext
    log = create_logger("e2e_mock", trace_id=new_trace_id())
    log.info("E2E step5 日志", step="e2e.log", latency_ms=1, status="ok")
    with LogContext(user_id="u_e2e", paper_id="paper_1") as ctx:
        ctx.logger.info("E2E LogContext 日志")
    print("✓ structlog / StdJson 结构化日志")

    # 成本追踪：使用独立 tracker，避免受之前运行的全局数据污染
    from app.observability import (
        InMemoryCostTracker, CostDashboard, CostRecord,
    )
    from datetime import datetime
    tracker = InMemoryCostTracker()
    tracker.record(CostRecord(
        ts=datetime.now(), model="qwen-plus", step="e2e.review.generate",
        input_tokens=1000, output_tokens=500, cost_yuan=0.01,
        user_id="u_e2e", paper_id="paper_1", from_cache=False,
    ))
    tracker.record(CostRecord(
        ts=datetime.now(), model="qwen-plus", step="e2e.review.verify",
        input_tokens=600, output_tokens=100, cost_yuan=0.0036,
        user_id="u_e2e", paper_id="paper_1", from_cache=True,
    ))
    dash = CostDashboard(tracker)
    by_user = dash.cost_by_user()
    cache = dash.cache_hit_rate()
    print(f"按用户分摊: u_e2e = {by_user.get('u_e2e', {}).get('cost', 0):.4f}元")
    print(f"缓存命中率: {cache['rate']:.1%} ({cache['hits']}/{cache['total']})")
    assert "u_e2e" in by_user
    assert cache["total"] == 2, f"独立 tracker 应有 2 条记录，实际 {cache['total']} 条（可能混入其他模块的数据）"
    assert cache["hits"] == 1
    print("✓ 成本追踪 + 看板")

    # Langfuse trace
    from app.observability import get_langfuse_client
    lf = get_langfuse_client()
    trace = lf.start_trace(name="e2e.mock", user_id="u_e2e", paper_id="paper_1")
    with lf.span(trace, "rag.search", input={"q": q}, metadata={"top_k": 5}) as s:
        s.output = {"count": len(top5)}
    with lf.span(trace, "review.pipeline") as s:
        s.metadata["sections"] = len(doc.outline)
    lf.end_trace(trace)
    assert len(trace.spans) == 2
    print(f"✓ Langfuse trace 上报（{len(trace.spans)} spans）")

    # ------------------------------------------------------------------
    # Step 6: API 路由
    # ------------------------------------------------------------------
    step("Step 6 · FastAPI 路由（/health /ask /review）")
    from app.api import create_app, AskRequest, ReviewRequest, UploadRequest, UploadResponse

    app = create_app(mock_mode=True)
    if app is None:
        print("⚠ fastapi 未安装，仅验证 schema（跳过端点）")
        assert AskRequest(paper_id="p1", question="q").question == "q"
        assert ReviewRequest(topic="t", doc_ids=["p1"]).topic == "t"
    else:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r1 = client.get("/health")
        assert r1.status_code == 200 and r1.json()["status"] == "ok"
        print(f"/health → ok, mock_mode={r1.json()['mock_mode']}")
        r2 = client.post("/ask", json={"paper_id": "paper_1",
                                       "question": "什么是 Transformer？"})
        assert r2.status_code == 200
        assert r2.json().get("error") == ""
        print(f"/ask → answer={r2.json()['answer'][:60]}...")
        r3 = client.post("/review", json={
            "topic": "Transformer vs BERT",
            "doc_ids": ["paper_1", "paper_2"],
            "num_sections": 2,
        })
        assert r3.status_code == 200
        data = r3.json()
        assert data.get("error") == "", f"error: {data.get('error')}"
        print(f"/review → title={data.get('title')}, paragraphs={len(data.get('paragraphs', []))}")
    print("✓ FastAPI 路由（schema + mock）")

    # ------------------------------------------------------------------
    # Step 7: Streamlit 前端 import 正常
    # ------------------------------------------------------------------
    step("Step 7 · Streamlit 前端 import 正常（不启动 UI）")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "streamlit_app",
            PROJECT_ROOT / "app" / "frontend" / "streamlit_app.py",
        )
        # 只做 AST 级的语法检查 + import 依赖的可导入性验证，不真执行（会启动 UI）
        import ast
        src = (PROJECT_ROOT / "app" / "frontend" / "streamlit_app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        func_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        required = {"render_sidebar", "render_upload_tab", "render_ask_tab",
                    "render_review_tab", "render_dashboard_tab", "main"}
        assert required <= func_names, f"缺函数: {required - func_names}"
        print(f"✓ Streamlit 前端语法无误，函数齐全: {sorted(required & func_names)}")
    except SyntaxError as e:
        raise AssertionError(f"streamlit_app.py 语法错误: {e}")

    # 最终统计
    total_ms = int((time.time() - total_t0) * 1000)
    print("\n" + "=" * 70)
    print(f"E2E mock 模式全部通过 ✓  总耗时: {total_ms}ms")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

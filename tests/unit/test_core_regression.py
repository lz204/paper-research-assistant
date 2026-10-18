"""核心模块 pytest 回归测试（企业级工程化配套，mock 模式运行）。

覆盖此前发现并修复的缺陷，防止回归：
1. chunker 短章节内容丢失（token ≤ overlap 时整节被丢弃）
2. CitationVerifier 引用三色 + chunk 存在性
3. ContextManager LRU 淘汰
4. HybridRetriever / FakeEmbedder 全链路
5. MMR 去重（select 与 select_batch 结果一致）
6. 综述文档级引用编号唯一连续

运行：
    pytest tests/unit/test_core_regression.py -v
"""

from __future__ import annotations

from app.rag.chunker import HierarchicalChunker
from app.rag.context_manager import ContextManager
from app.rag.embedder import FakeEmbedder
from app.rag.mmr import MMRItem, MMRSelector, mmr_select
from app.rag.retriever import HybridRetriever, InMemoryVectorRetriever
from app.rag.reranker import NoopReranker
from app.review.citation_verifier import CitationVerifier, renumber_results
from app.review.outline_planner import MockOutlinePlanner


# ---------------------------------------------------------------------------
# 1. chunker 短章节内容不丢失（回归：曾输出 0 chunk）
# ---------------------------------------------------------------------------

def _chunk(text: str) -> list:
    return HierarchicalChunker(chunk_size=256, overlap=64).chunk(text, doc_id="d1")


def test_chunker_short_section_not_lost() -> None:
    """两个很短章节（各自 token 数远小于 overlap=64）不能整节丢弃"""
    paper = """\
# Short Paper

## 1. Intro

We propose a small method.

## 2. Conclusion

We conclude.
"""
    chunks = _chunk(paper)
    assert len(chunks) >= 1, "短章节应至少输出 1 个 chunk（回归：曾输出 0）"
    full_text = "\n".join(c.text for c in chunks)
    assert "We propose a small method." in full_text, "Intro 内容丢失"
    assert "We conclude." in full_text, "Conclusion 内容丢失"


def test_chunker_preserves_sections() -> None:
    """常规文档：多章节保留 + chunk 大小约束"""
    paper = (
        "# Title\n\n## 1. Intro\n\n" + "intro content. " * 40 + "\n\n"
        "## 2. Method\n\n" + "method content. " * 40
    )
    chunks = _chunk(paper)
    assert len(chunks) >= 2
    sections = {c.section for c in chunks}
    assert "1. Intro" in sections and "2. Method" in sections
    assert all(c.token_count <= 256 + 50 for c in chunks)


# ---------------------------------------------------------------------------
# 2. CitationVerifier 三色 + chunk 存在性
# ---------------------------------------------------------------------------

def _make_retriever() -> InMemoryVectorRetriever:
    emb = FakeEmbedder(dim=64)
    ret = InMemoryVectorRetriever(emb)
    from app.rag.chunker import Chunk
    chunks = [
        Chunk(chunk_id="p1#0.0", text="ResNet accuracy 85% on ImageNet-1K",
              token_count=8, doc_id="p1"),
        Chunk(chunk_id="p2#0.0", text="BERT pretrains language model",
              token_count=6, doc_id="p2"),
    ]
    ret.add_chunks(chunks)
    return ret


def test_citation_verifier_three_colors() -> None:
    ret = _make_retriever()
    verifier = CitationVerifier(retriever=ret, nli_reranker=NoopReranker())
    text = (
        "A uses ResNet [p1, p1#0.0]. "
        "Fake cite [p1, p1#9.9] should be red."
    )
    result = verifier.verify_paragraph(text)
    assert len(result.citations) == 2
    real = result.citations[0]
    fake = result.citations[1]
    assert real.exists is True and real.label == "neutral"  # Noop → neutral
    assert real.color == "#6b7280"
    assert fake.exists is False and fake.label == "contradiction"
    assert fake.color == "#dc2626"


# ---------------------------------------------------------------------------
# 3. ContextManager LRU
# ---------------------------------------------------------------------------

def test_context_manager_lru_eviction() -> None:
    cm = ContextManager(max_size=3)
    for cid, txt in [
        ("c1", "Use ResNet on ImageNet-1K, accuracy 85%."),
        ("c2", "SwinTransformer on COCO-2017 gets mAP=0.81."),
        ("c3", "BERT for NLP tasks, BLEU=27.3."),
        ("c4", "Experiments show IoU=0.74 on GLUE-SST."),  # 触发淘汰
    ]:
        cm.update(txt, cid)
    assert len(cm) == 3, "超过 max_size 应淘汰到 3"
    assert "ResNet" not in cm, "最久未访问的 ResNet 应被淘汰"


# ---------------------------------------------------------------------------
# 4. 混合检索全链路（FakeEmbedder + BM25 降级）
# ---------------------------------------------------------------------------

def test_hybrid_retriever_chain() -> None:
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    paper = "# P\n\n## 1. Method\n\nWe use SwinTransformer and BLEU=27.3 results.\n\n## 2. Result\n\nmAP 0.81 on COCO-2017."
    chunks = chunker.chunk(paper, doc_id="p1")
    assert len(chunks) >= 1

    ret = HybridRetriever(FakeEmbedder(dim=64), use_bm25=True, reranker=None)
    ret.add_chunks(chunks)
    results = ret.search("BLEU", top_k=3)
    assert len(results) >= 1
    # BM25 关键词路径应召回含 BLEU 的 chunk
    assert any("BLEU" in r.chunk.text for r in results)


# ---------------------------------------------------------------------------
# 5. MMR select / select_batch 一致性
# ---------------------------------------------------------------------------

def test_mmr_consistency() -> None:
    emb = FakeEmbedder(dim=32)
    query_vec = emb.embed_one("transformer attention")
    texts = [
        ("c1", "transformer self-attention mechanism"),
        ("c2", "transformer multi-head attention"),
        ("c3", "BERT pre-training language"),
        ("c4", "image classification CNN"),
    ]
    items = [MMRItem(id=cid, vec=emb.embed_one(t), payload=t) for cid, t in texts]

    single = mmr_select(items, query_vec, lambda_=0.7, k=3)
    selector = MMRSelector(lambda_=0.7)
    batch = selector.select_batch(items, query_vec, k=3)
    assert [x.id for x in single] == [x.id for x in batch]


# ---------------------------------------------------------------------------
# 6. 文档级引用全局编号唯一连续（回归：多段 index 冲突丢条目）
# ---------------------------------------------------------------------------

def test_renumber_results_global_indexing() -> None:
    ret = _make_retriever()
    verifier = CitationVerifier(retriever=ret, nli_reranker=NoopReranker())
    paras = [
        "Para1 cite A [p1, p1#0.0] and B [p2, p2#0.0]",
        "Para2 cite A again [p1, p1#0.0] and new C [p1, p1#0.0] and D [p2, p2#0.0]",
    ]
    results = [verifier.verify_paragraph(p) for p in paras]
    renumbered = renumber_results(results)

    # 展开全部 index，应为 {1,2}（p1#0.0 与 p2#0.0 各自全局唯一）
    all_idx = {c.ref.index for r in renumbered for c in r.citations}
    assert all_idx == {1, 2}, f"全局编号应唯一且从 1 连续，得到 {sorted(all_idx)}"

    # 同一 doc+chunk 跨段编号一致
    idx_para1 = {c.ref.index for c in renumbered[0].citations}
    idx_para2 = {c.ref.index for c in renumbered[1].citations}
    assert 1 in idx_para1 and 1 in idx_para2, "跨段重复引用编号应一致"
    # 每段渲染后都应有对应 marker
    assert all(any(c.marker for c in r.citations) for r in renumbered), "marker 缺失"


# ---------------------------------------------------------------------------
# 7. Mock 综述生成（大纲 → 段落 → 缓存）快速冒烟
# ---------------------------------------------------------------------------

def test_mock_review_generation_smoke() -> None:
    from app.rag.chunker import Chunk
    from app.review.generator import MockReviewGenerator

    chunks = {
        "p1": [Chunk(chunk_id="p1#0.0", text="method A content", token_count=4, doc_id="p1")],
        "p2": [Chunk(chunk_id="p2#0.0", text="method B content", token_count=4, doc_id="p2")],
    }
    plan = MockOutlinePlanner().plan(
        "test review", [{"doc_id": "p1"}, {"doc_id": "p2"}], num_sections=2
    )
    gen = MockReviewGenerator()
    result = gen.generate(topic=plan.topic, sections=plan.sections, doc_chunks=chunks)
    assert result.completed is True
    assert len(result.paragraphs) == 2
    assert result.full_text.strip()


# ---------------------------------------------------------------------------
# 8. FusionEngine 低置信度表格二次验证 + vlm_only 场景（覆盖率补齐）
# ---------------------------------------------------------------------------

def test_fusion_engine_table_reverify_and_vlm_only() -> None:
    """低置信度表格触发二次验证 + 只有 VLM 结果（无 pdfplumber）的融合路径"""
    from app.parser.fusion_engine import FusionEngine
    from app.parser.vlm_client import QwenVlmClient, VlmFormula, VlmPageResult, VlmTable

    # QwenVlmClient 无 key → mock 模式，二次验证走 mock 提升置信度分支
    vlm = QwenVlmClient()
    engine = FusionEngine(vlm_client=vlm)

    # vlm_only 场景：无 pdfplumber layout，公式应全部 unbound
    result = VlmPageResult(
        page_no=0,
        text="para one\n\npara two",
        formulas=[VlmFormula(latex=r"E=mc^2", location_desc="", confidence=0.9, formula_id="(1)")],
        tables=[VlmTable(markdown="|a|", location_desc="", confidence=0.5)],  # 低置信度 → 触发二次验证
        confidence=0.8,
        formula_resolved=True,
    )
    doc = engine.fuse(doc_id="d", vlm_results=[result], page_layouts=[])
    page = doc.pages[0]
    assert len(page.paragraphs) == 2
    assert len(page.unbound_formulas) == 1
    # mock 二次验证把表格置信度从 0.5 提到 0.85，need_review 应被清除
    assert len(page.tables) == 1
    assert page.tables[0].confidence == 0.85
    assert page.tables[0].need_review is False


def test_fusion_engine_empty_and_full_degradation() -> None:
    """空输入 + 纯 pdfplumber 降级路径（覆盖率补齐）"""
    from app.parser.fusion_engine import FusionEngine
    from app.parser.text_extractor import build_mock_layout

    engine = FusionEngine()
    # 空输入
    empty = engine.fuse(doc_id="e", vlm_results=[], page_layouts=[])
    assert len(empty.pages) == 0
    # 只有 pdfplumber → VLM 降级
    layout = build_mock_layout(page_no=0, paragraphs=["only pdfplumber text"])
    doc = engine.fuse(doc_id="e2", vlm_results=[], page_layouts=[layout])
    page = doc.pages[0]
    assert page.formula_resolved is False
    assert page.confidence == 0.5


# ---------------------------------------------------------------------------
# 9. CitationVerifier 边界：单条验证 / 无 retriever / 超长 chunk（覆盖率补齐）
# ---------------------------------------------------------------------------

def test_citation_verifier_single_and_no_retriever() -> None:
    from app.review.citation_verifier import CitationVerifier

    # 无 retriever：所有引用应判为不存在（contradiction）
    verifier = CitationVerifier(retriever=None, nli_reranker=NoopReranker())
    result = verifier.verify_paragraph("claim [p1, p1#0.0]")
    assert len(result.citations) == 1
    assert result.citations[0].exists is False
    assert result.citations[0].label == "contradiction"

    # 单条验证（verify_citation）
    ret = _make_retriever()
    verifier2 = CitationVerifier(retriever=ret, nli_reranker=NoopReranker())
    from app.review.citation_verifier import parse_citations
    ref = parse_citations("claim [p1, p1#0.0]")[0]
    vc = verifier2.verify_citation(ref, "claim")
    assert vc.exists is True


def test_citation_verifier_long_chunk_truncation() -> None:
    """超长 chunk 触发 NLI 前截断路径（覆盖率补齐）"""
    from app.rag.chunker import Chunk
    from app.review.citation_verifier import CitationVerifier

    emb = FakeEmbedder(dim=64)
    ret = InMemoryVectorRetriever(emb)
    long_text = "word " * 3000  # > 2000 字符，触发截断
    ret.add_chunks([Chunk(chunk_id="p1#0.0", text=long_text, token_count=200, doc_id="p1")])

    verifier = CitationVerifier(retriever=ret, nli_reranker=NoopReranker())
    result = verifier.verify_paragraph("claims something [p1, p1#0.0]")
    assert len(result.citations) == 1
    assert result.citations[0].exists is True


# ---------------------------------------------------------------------------
# 10. FastAPI 真实编排（/upload + /ask 链路，回归）
#    背景：run_parser_pipeline / run_ask 曾为占位；已改为
#    PyMuPDF 抽文本 → 分块 → 进程内索引 → 混合检索 → LLM。
# ---------------------------------------------------------------------------

def test_api_parser_ask_real_pipeline() -> None:
    """真实编排：解析 PDF bytes → run_ask 走 RAG（mock 组件注入）"""
    from app.api.routes import _AppState

    try:
        import fitz  # PyMuPDF
    except ImportError:
        import pytest
        pytest.skip("PyMuPDF 未安装，跳过 PDF 真实编排测试")

    # 用 PyMuPDF 构造含文本的真实 PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "We propose ResNet with 85.6% accuracy on ImageNet-1K.")
    page.insert_text((72, 120), "Experiments show BLEU 28.4 on WMT 2014.")
    pdf_bytes = doc.tobytes()
    doc.close()

    state = _AppState(mock_mode=False)  # 直接调真实编排（不经 mock 拦截）
    up = state.run_parser_pipeline("paper.pdf", pdf_bytes, "u1")
    assert up.error == "", f"解析失败: {up.error}"
    assert up.num_pages >= 1
    assert up.num_chunks >= 1

    # 注入 mock 组件，避免真实 API
    from app.llm import MockLLMClient
    state._embedder = FakeEmbedder(dim=64)
    state._llm_client = MockLLMClient()

    resp = state.run_ask(up.doc_id, "BLEU on WMT 2014?", "u1", True, 5)
    assert resp.error == "", f"run_ask 失败: {resp.error}"
    assert resp.answer, "答案为空"
    assert resp.citations, "应至少附带 1 条引用"

    # 未上传论文 → 明确报错（不再静默占位）
    resp2 = state.run_ask("doc_missing", "q", "u1", True, 5)
    assert "未找到" in resp2.error


def test_bm25_small_corpus_fallback() -> None:
    """回归：rank_bm25 在小语料下 idf≈0 全 0 分，应回退 Jaccard 保证关键词召回"""
    from app.rag.chunker import Chunk
    from app.rag.retriever import BM25Retriever

    b = BM25Retriever()
    b.add_chunks([
        Chunk(chunk_id="d#0.0", text="We propose ResNet with accuracy on ImageNet-1K. BLEU 28.4 results.",
              token_count=10, doc_id="d"),
    ])
    # 只有 1 篇文档时 BM25Okapi idf 会退化为 0；关键词仍需被召回
    results = b.search("BLEU", top_k=3)
    assert len(results) >= 1, "BM25 小语料下关键词召回不应为空（回退失效）"
    assert "BLEU" in results[0].chunk.text


def test_embedder_batch_splitting_and_fallback(monkeypatch) -> None:
    """回归：DashScope embedding 分包（batch<=10）+ 整批失败逐条降级

    真实 API 曾因 batch_size=25 超服务端 10 上限而批量失败、只 7/107 成功；
    另曾因响应解析为 dict 而抛 AttributeError。
    """
    import numpy as np
    from app.rag.embedder import DashScopeEmbedder

    emb = DashScopeEmbedder(api_key="sk-dummy")  # 不真正调 API

    # 用 stub 拦截底层调用：按真实成功向量返回，统计每次请求条数
    call_sizes: list[int] = []
    vec_dim = 8

    def fake_call(texts):
        call_sizes.append(len(texts))
        return [np.random.rand(vec_dim).astype(np.float32) for _ in texts]

    monkeypatch.setattr(emb, "_call_api_with_retry", fake_call)
    monkeypatch.setattr(emb, "_ensure_client", lambda: True)

    texts = [f"sample sentence number {i} with enough length to fill chars" for i in range(35)]
    vecs, failed = emb.embed(texts)
    assert failed == [], "分包后不应有失败"
    assert len([v for v in vecs if v is not None]) == 35
    # 任何单次请求都不应超过服务端上限 10
    assert max(call_sizes) <= 10, f"单次请求 {max(call_sizes)} 条，超过 DashScope batch<=10 上限"

    # 整批失败 → 逐条降级仍应尽量成功
    def fake_call_fail_first(texts):
        return None  # 模拟整批失败（已重试仍失败）

    monkeypatch.setattr(emb, "_call_api_with_retry", fake_call_fail_first)
    vecs2, failed2 = emb.embed(["only one text a"])
    # 单条降级也失败时返回失败索引；这里验证不抛异常、长度正确
    assert len(vecs2) == 1
    assert len(failed2) == 1


def test_entity_extractor_disambiguation_and_metric_forms() -> None:
    """回归：entity_extractor 领域消歧 + 指标双格式抽取

    曾发现：ImageNet-1K 的前缀被误抽为 model；mAP=0.81 / F1=0.92（数字在后）
    抽取不到；BERT-base / ResNet-50 带版本模型识别缺失。
    """
    from app.rag.entity_extractor import EntityType, RegexEntityExtractor

    ex = RegexEntityExtractor()
    text = (
        "BERT-base and GPT-2 on ImageNet-1K; ResNet-50 and ViT on COCO-2017; "
        "SwinTransformer gets mAP=0.81, F1=0.92 and IoU=0.74, BLEU=28.4."
    )
    ents = ex.extract(text)
    by_type: dict = {}
    for e in ents:
        by_type.setdefault(e.type, []).append(e.value)

    models = by_type.get(EntityType.MODEL, [])
    datasets = by_type.get(EntityType.DATASET, [])
    metrics = by_type.get(EntityType.METRIC, [])

    # 消歧：数据集不被误当模型
    assert "ImageNet-1K" in datasets
    assert "ImageNet" not in models, "ImageNet 不应被误抽为 model"
    # 带版本模型正确识别
    assert "BERT-base" in models
    assert "ResNet-50" in models
    assert "ViT" in models
    # 指标双格式（数字在后）
    assert "mAP=0.81" in metrics
    assert "F1=0.92" in metrics
    assert "BLEU=28.4" in metrics


# ---------------------------------------------------------------------------
# 6. 评测数值解析（P1 回归：quick 模式曾因 "18..." 抛 ValueError 中断）
# ---------------------------------------------------------------------------

def test_eval_find_floats_handles_ellipsis_and_versions() -> None:
    """回归：run_eval 数值抽取不能把省略号/多小数点当 float

    旧正则 ``[\\-]?\\d[\\d\\.]*`` 会把 "18..." 整段命中，
    ``float("18...")`` 抛 ValueError 导致 `python tests/eval/run_eval.py
    --mode quick` 在第 3/10 题中断，report.json 无法生成。
    """
    from tests.eval.run_eval import _find_floats, _number_fuzzy_match

    assert _find_floats("BLEU 28.4, top-5 85.6%, 18...") == [28.4, 5.0, 85.6, 18.0]
    assert _find_floats("d_model = 512, 8 heads, 0.1") == [512.0, 8.0, 0.1]
    assert _find_floats("no numbers here") == []
    assert _find_floats("3.13 与 -0.5") == [3.13, -0.5]
    assert _number_fuzzy_match("BLEU 28.4 (18...)", "28.4 BLEU", None) == 1.0

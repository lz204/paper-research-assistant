"""综述生成全链路集成验证脚本

模拟项目文档第 3 章架构图的综述数据流：
    Review[综述主题] --> RAG2[多文档检索] --> MMR[MMR 去重]
    MMR --> Gen[Map-Reduce 生成] --> Verify[引用事实验证]
    Verify --> Docx[Word 导出]

本脚本验证 review 模块（mock 模式，无 API key）：
    主题 → outline_planner（大纲）
                 ↓
    doc_chunks → generator（Map-Reduce + 段落级缓存）
                 ↓
    paragraphs → citation_verifier（解析 + 二次检索 + NLI + 三色标记）

运行方式（在项目根目录 paper-research-assistant/ 下）：
    python tests/unit/verify_review_pipeline.py
"""

from __future__ import annotations

import sys
import time

# 共享引导：插入项目根目录到 sys.path + 清空 API key / Langfuse 配置（强制 mock）；
# 两篇样例论文语料与分块/文献信息构建函数统一来自 _verify_common（与 verify_infrastructure 共用）
from _verify_common import (
    PROJECT_ROOT,
    SAMPLE_PAPER_1,
    SAMPLE_PAPER_2,
    build_doc_chunks as _build_doc_chunks,
    build_doc_infos as _build_doc_infos,
)  # noqa: E402

from app.rag import (  # noqa: E402
    Chunk,
    FakeEmbedder,
    HierarchicalChunker,
    HybridRetriever,
    InMemoryVectorRetriever,
    NoopReranker,
    create_reranker,
)
from app.review import (  # noqa: E402
    Citation,
    CitationRef,
    CitationVerifier,
    GenerationResult,
    LLMOutlinePlanner,
    LLMReviewGenerator,
    MockOutlinePlanner,
    MockReviewGenerator,
    OutlinePlan,
    OutlineSection,
    ParagraphCache,
    VerifiedCitation,
    VerificationResult,
    create_outline_planner,
    create_review_generator,
    parse_citations,
    render_citations_html,
)

# ---------------------------------------------------------------------------
# 测试 1：大纲规划
# ---------------------------------------------------------------------------


def test_outline_planner() -> OutlinePlan:
    """测试 1：大纲规划（Mock + LLM 降级 + 工厂函数）"""
    print("\n" + "=" * 70)
    print("测试 1：大纲规划（MockOutlinePlanner + 工厂函数降级）")
    print("=" * 70)

    doc_infos = _build_doc_infos()

    # Mock 规划器
    mock_planner = MockOutlinePlanner()
    plan = mock_planner.plan(
        topic="Transformer 在医学影像分割的应用",
        doc_infos=doc_infos,
        num_sections=4,
    )
    print(f"主题: {plan.topic}")
    print(f"章节数: {len(plan)}")
    print(f"全部文献: {plan.doc_ids}")
    for sec in plan:
        print(f"  [{sec.section_id}] {sec.title}")
        print(f"      描述: {sec.description}")
        print(f"      相关文献: {sec.related_doc_ids}")

    assert len(plan) == 4, f"应生成 4 个章节，得到 {len(plan)}"
    # 验证文献平均分配：2 篇论文分到 4 个章节
    total_assigned = sum(len(s.related_doc_ids) for s in plan)
    assert total_assigned == len(doc_infos), "全部文献应被分配到章节中"
    print("  ✓ Mock 大纲规划正确，文献分配完整")

    # LLM 规划器（无 API key → 应降级为 Mock）
    llm_planner = LLMOutlinePlanner()
    plan2 = llm_planner.plan("测试主题", doc_infos, num_sections=3)
    assert len(plan2) > 0, "LLM 降级后应仍能生成大纲"
    print(f"  ✓ LLMOutlinePlanner 无 API key 时正确降级（{len(plan2)} 个章节）")

    # 工厂函数 auto 模式（无 API key → Mock）
    auto_planner = create_outline_planner("auto")
    assert isinstance(auto_planner, MockOutlinePlanner), "auto 无 key 应降级为 Mock"
    print(f"  ✓ create_outline_planner('auto') 降级为 {type(auto_planner).__name__}")

    return plan


# ---------------------------------------------------------------------------
# 测试 2：引用解析
# ---------------------------------------------------------------------------


def test_citation_parsing() -> None:
    """测试 2：[doc_id, chunk_id] 引用标记解析"""
    print("\n" + "=" * 70)
    print("测试 2：引用标记解析（parse_citations）")
    print("=" * 70)

    # 含 3 个引用（其中 2 个相同）的段落
    text = (
        "ResNet 在 ImageNet 上取得 85.6% 准确率 [paper_1, paper_1#0.1]。"
        "BERT 通过 masked LM 改进下游任务 [paper_2, paper_2#0.2]。"
        "再次引用 ResNet [paper_1, paper_1#0.1] 应去重。"
    )
    refs = parse_citations(text)
    print(f"段落: {text}")
    print(f"解析出 {len(refs)} 个去重引用：")
    for r in refs:
        print(f"  [{r.index}] doc_id={r.doc_id}, chunk_id={r.chunk_id}, raw={r.raw}")

    assert len(refs) == 2, f"去重后应为 2 个引用，得到 {len(refs)}"
    assert refs[0].doc_id == "paper_1" and refs[0].chunk_id == "paper_1#0.1"
    assert refs[1].doc_id == "paper_2" and refs[1].chunk_id == "paper_2#0.2"
    assert refs[0].index == 1 and refs[1].index == 2, "编号应从 1 开始递增"
    print("  ✓ 引用解析正确，去重 + 编号正常")

    # 无引用的文本
    empty_refs = parse_citations("这段话没有任何引用标记")
    assert empty_refs == [], "无引用应返回空列表"
    print("  ✓ 无引用文本返回空列表")


# ---------------------------------------------------------------------------
# 测试 3：引用验证（含二次检索 + 三色标记）
# ---------------------------------------------------------------------------


def test_citation_verification(doc_chunks: dict) -> InMemoryVectorRetriever:
    """测试 3：引用事实验证（CitationVerifier + 二次检索 + 三色标记）

    Returns:
        retriever（供后续全链路测试复用）
    """
    print("\n" + "=" * 70)
    print("测试 3：引用事实验证（二次检索 + NLI + 三色标记）")
    print("=" * 70)

    # 构造检索器（含真实 chunk，用于二次检索）
    embedder = FakeEmbedder(dim=64)
    retriever = InMemoryVectorRetriever(embedder)
    all_chunks = []
    for chunks in doc_chunks.values():
        all_chunks.extend(chunks)
    retriever.add_chunks(all_chunks)
    print(f"检索器索引 {len(retriever)} 个 chunk")

    # NoopReranker：NLI 标签为空 → 全部映射为 neutral（灰色）
    # 这模拟了无 API key 的本地联调场景
    reranker = create_reranker("auto")
    verifier = CitationVerifier(retriever=retriever, nli_reranker=reranker)

    # 取一个真实 chunk_id + 一个伪造 chunk_id 构造段落
    real_chunk = all_chunks[0]
    real_chunk_id = real_chunk.chunk_id
    fake_chunk_id = "paper_1#9.9"  # 不存在的 chunk_id

    paragraph = (
        f"ResNet 在图像识别任务上表现优异 [{real_chunk_id.split('#')[0]}, {real_chunk_id}]。"
        f"另有一处伪造引用 [paper_1, {fake_chunk_id}] 应被标红。"
    )
    print(f"\n段落: {paragraph}")

    result = verifier.verify_paragraph(paragraph)
    print(f"\n验证结果（共 {len(result.citations)} 条引用）：")
    for cit in result.citations:
        print(f"  [{cit.ref.index}] chunk_id={cit.ref.chunk_id}")
        print(f"      exists={cit.exists}, label={cit.label}, marker={cit.marker}")
        print(f"      color={cit.color}, tooltip={cit.tooltip!r}")

    print(f"\n渲染纯文本:\n  {result.rendered_text}")
    print(f"\n渲染 HTML:\n  {result.rendered_html}")
    print(f"\n统计: {result.stats}")

    # 断言
    assert len(result.citations) == 2, "应解析出 2 条引用"
    # 真实 chunk：exists=True，NoopReranker → neutral（灰色）
    real_cit = result.citations[0]
    assert real_cit.exists is True, "真实 chunk_id 应存在"
    assert real_cit.label == "neutral", "NoopReranker 应映射为 neutral"
    assert real_cit.color == "#6b7280", "neutral 应为灰色"
    assert real_cit.marker == "[1?]", "neutral 标记应为 [1?]"
    print("  ✓ 真实引用：exists=True, label=neutral, 灰色 [1?]")

    # 伪造 chunk：exists=False → contradiction（红色，伪造）
    fake_cit = result.citations[1]
    assert fake_cit.exists is False, "伪造 chunk_id 应不存在"
    assert fake_cit.label == "contradiction", "伪造引用应为 contradiction"
    assert fake_cit.color == "#dc2626", "contradiction 应为红色"
    assert fake_cit.marker == "[2!]", "contradiction 标记应为 [2!]"
    print("  ✓ 伪造引用：exists=False, label=contradiction, 红色 [2!]")

    # 统计
    assert result.stats["total"] == 2
    assert result.stats["neutral"] == 1
    assert result.stats["contradiction"] == 1
    print("  ✓ 统计信息正确")

    # XSS 过滤验证（HTML 中 < > 应被转义）
    xss_text = "正常引用 [paper_1, paper_1#0.1] <script>alert(1)</script>"
    xss_result = verifier.verify_paragraph(xss_text)
    assert "<script>" not in xss_result.rendered_html, "XSS 应被过滤"
    assert "&lt;script&gt;" in xss_result.rendered_html, "< > 应被转义"
    print("  ✓ XSS 过滤正常（< > 转义）")

    return retriever


# ---------------------------------------------------------------------------
# 测试 4：便捷函数 render_citations_html
# ---------------------------------------------------------------------------


def test_render_helper(retriever) -> None:
    """测试 4：render_citations_html 便捷函数"""
    print("\n" + "=" * 70)
    print("测试 4：render_citations_html 便捷函数")
    print("=" * 70)

    text = "测试便捷函数 [paper_1, paper_1#0.1] 是否一步完成验证+渲染。"
    result = render_citations_html(text, retriever=retriever, nli_reranker=NoopReranker())
    print(f"输入: {text}")
    print(f"输出 HTML: {result.rendered_html}")
    assert len(result.citations) == 1
    assert "<span" in result.rendered_html
    print("  ✓ 便捷函数正常工作")


# ---------------------------------------------------------------------------
# 测试 5：综述生成（Mock + 段落级缓存）
# ---------------------------------------------------------------------------


def test_review_generator(doc_chunks: dict, plan: OutlinePlan) -> GenerationResult:
    """测试 5：综述生成（MockReviewGenerator + 段落级缓存）

    Returns:
        GenerationResult（供全链路测试复用）
    """
    print("\n" + "=" * 70)
    print("测试 5：综述生成（MockReviewGenerator + 段落级缓存）")
    print("=" * 70)

    # 共享缓存，验证第二次生成命中缓存
    shared_cache = ParagraphCache()
    generator = MockReviewGenerator(cache=shared_cache)

    # 第一次生成
    print("\n[第一次生成]")
    t0 = time.time()
    result = generator.generate(
        topic=plan.topic,
        sections=plan.sections,
        doc_chunks=doc_chunks,
    )
    elapsed_ms = (time.time() - t0) * 1000
    print(f"生成 {len(result.paragraphs)} 个段落，耗时 {elapsed_ms:.1f}ms")
    print(f"completed={result.completed}, cache_hits={result.cache_hits}, cache_misses={result.cache_misses}")
    print(f"failed_sections={result.failed_sections}")

    for p in result.paragraphs:
        print(f"\n  [{p.section_id}] from_cache={p.from_cache}")
        print(f"  引用数: {len(p.citations)}")
        preview = p.text[:120].replace("\n", " ")
        print(f"  正文预览: {preview}...")

    assert result.completed is True, "Mock 生成应全部完成"
    assert result.cache_misses == len(plan.sections), "首次应全部未命中"
    assert result.cache_hits == 0, "首次应无缓存命中"
    assert len(result.paragraphs) == len(plan.sections)
    print("\n  ✓ 第一次生成全部完成，缓存全部写入")

    # 第二次生成（相同输入）→ 应全部命中缓存
    print("\n[第二次生成（相同输入，应命中缓存）]")
    t0 = time.time()
    result2 = generator.generate(
        topic=plan.topic,
        sections=plan.sections,
        doc_chunks=doc_chunks,
    )
    elapsed_ms = (time.time() - t0) * 1000
    print(f"生成 {len(result2.paragraphs)} 个段落，耗时 {elapsed_ms:.1f}ms")
    print(f"cache_hits={result2.cache_hits}, cache_misses={result2.cache_misses}")

    assert result2.cache_hits == len(plan.sections), "第二次应全部命中缓存"
    assert result2.cache_misses == 0, "第二次应无未命中"
    assert all(p.from_cache for p in result2.paragraphs), "所有段落应标记 from_cache"
    print("  ✓ 第二次生成全部命中缓存（文档 6.2 节：重跑成本降 40%）")

    # LLM 生成器（无 API key → 降级 Mock）
    llm_gen = LLMReviewGenerator()
    result3 = llm_gen.generate(plan.topic, plan.sections, doc_chunks)
    assert result3.completed is True, "LLM 降级 Mock 后应正常完成"
    print(f"\n  ✓ LLMReviewGenerator 无 API key 降级为 Mock（{len(result3.paragraphs)} 段）")

    # 工厂函数
    auto_gen = create_review_generator("auto")
    assert isinstance(auto_gen, MockReviewGenerator), "auto 无 key 应降级为 Mock"
    print(f"  ✓ create_review_generator('auto') 降级为 {type(auto_gen).__name__}")

    # 缓存隔离：不同主题不命中
    other_plan = MockOutlinePlanner().plan("另一个主题", _build_doc_infos(), num_sections=2)
    result4 = generator.generate(other_plan.topic, other_plan.sections, doc_chunks)
    assert result4.cache_misses == len(other_plan.sections), "不同主题不应命中缓存"
    print(f"  ✓ 不同主题不命中缓存（缓存 key 含 topic）")

    return result


# ---------------------------------------------------------------------------
# 测试 6：全链路集成（大纲 → 生成 → 验证）
# ---------------------------------------------------------------------------


def test_full_review_pipeline(doc_chunks: dict, retriever) -> None:
    """测试 6：综述全链路集成（大纲 → Map-Reduce 生成 → 引用验证 → 三色渲染）"""
    print("\n" + "=" * 70)
    print("测试 6：综述全链路集成（大纲→生成→验证→渲染）")
    print("=" * 70)

    # 1. 大纲规划
    print("\n[1/4] 大纲规划...")
    planner = create_outline_planner("auto")
    plan = planner.plan(
        topic="深度学习模型在视觉与语言任务中的应用",
        doc_infos=_build_doc_infos(),
        num_sections=3,
    )
    print(f"  → 主题: {plan.topic}")
    print(f"  → {len(plan)} 个章节: {[s.title for s in plan]}")

    # 2. Map-Reduce 生成
    print("\n[2/4] Map-Reduce 综述生成...")
    generator = create_review_generator("auto")
    gen_result = generator.generate(
        topic=plan.topic,
        sections=plan.sections,
        doc_chunks=doc_chunks,
    )
    print(f"  → 生成 {len(gen_result.paragraphs)} 个段落")
    print(f"  → completed={gen_result.completed}, failed={gen_result.failed_sections}")

    # 3. 引用验证
    print("\n[3/4] 引用事实验证（二次检索 + NLI + 三色标记）...")
    verifier = CitationVerifier(
        retriever=retriever,
        nli_reranker=create_reranker("auto"),
    )

    total_citations = 0
    total_entailment = 0
    total_contradiction = 0
    total_neutral = 0
    verified_results: list = []

    for para in gen_result.paragraphs:
        vresult = verifier.verify_paragraph(para.text)
        verified_results.append(vresult)
        total_citations += vresult.stats["total"]
        total_entailment += vresult.stats["entailment"]
        total_contradiction += vresult.stats["contradiction"]
        total_neutral += vresult.stats["neutral"]
        print(f"  [{para.section_id}] 引用 {vresult.stats['total']} 条 "
              f"(蓝{vresult.stats['entailment']}/红{vresult.stats['contradiction']}/灰{vresult.stats['neutral']})")

    print(f"\n  引用总计: {total_citations} 条")
    print(f"  - entailment (蓝色): {total_entailment}")
    print(f"  - contradiction (红色): {total_contradiction}")
    print(f"  - neutral (灰色): {total_neutral}")

    # 4. 渲染展示
    print("\n[4/4] 三色标记渲染展示（文档 4.3 节前端展示）...")
    for para, vresult in zip(gen_result.paragraphs, verified_results):
        print(f"\n  === {para.section_id} ===")
        print(f"  渲染文本: {vresult.rendered_text[:150]}...")
        print(f"  HTML 片段: {vresult.rendered_html[:200]}...")

    # 断言
    assert gen_result.completed, "综述应全部生成完成"
    assert total_citations > 0, "应至少有 1 条引用被验证"
    # Mock 模式下 NoopReranker → 全部 neutral；但因引用的是真实 chunk，exists=True
    # 这里只验证链路跑通，不强制 NLI 标签分布
    print("\n  ✓ 全链路集成验证通过（大纲→生成→验证→渲染）")

    # 验证综合述拼接
    full_review = gen_result.full_text
    print(f"\n  综述全文长度: {len(full_review)} 字符")
    assert len(full_review) > 0
    print("  ✓ 综述全文拼接正常")


# ---------------------------------------------------------------------------
# 测试 7：LLM 中断恢复（模拟 failed_sections + retry_hint）
# ---------------------------------------------------------------------------


def test_llm_interrupt_recovery(doc_chunks: dict) -> None:
    """测试 7：LLM 中断恢复（文档 6.3 节：部分结果好过无结果）"""
    print("\n" + "=" * 70)
    print("测试 7：LLM 中断恢复（文档 6.3 节容错）")
    print("=" * 70)

    # 构造一个会触发 _LLMInterruptedError 的 LLM 生成器
    # 通过 monkey-patch _call_llm_with_retry 让其返回 None（模拟最终失败）
    generator = LLMReviewGenerator()
    # 强制 client 可用（绕过 mock 降级），但 LLM 调用始终失败
    generator._mock_mode = False
    generator._client = object()  # 占位，使 _ensure_client 返回 True

    call_count = [0]
    def fake_call(prompt):
        call_count[0] += 1
        # 第 1 次调用（第 1 节 Map）失败 → 抛中断
        return None
    generator._call_llm_with_retry = fake_call

    plan = MockOutlinePlanner().plan("测试中断", _build_doc_infos(), num_sections=2)
    result = generator.generate(plan.topic, plan.sections, doc_chunks)

    print(f"completed: {result.completed}")
    print(f"failed_sections: {result.failed_sections}")
    print(f"已生成段落: {len(result.paragraphs)}")
    print(f"retry_hint: {result.retry_hint}")

    assert result.completed is False, "LLM 中断后 completed 应为 False"
    assert len(result.failed_sections) > 0, "应有失败的章节"
    assert result.retry_hint != "", "应提供重试提示"
    assert "重试" in result.retry_hint, "重试提示应包含'重试'字样"
    print("  ✓ LLM 中断时返回部分结果 + retry_hint（文档 6.3 节）")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def test_document_level_citation_numbering() -> None:
    """测试 8：文档级引用全局编号（回归断言）

    背景：CitationVerifier 按段独立编号（每段从 1 开始），多段综述合并后
    references 按 index 去重会丢条目，且正文 [N] 与文末编号不对应。
    修复：pipeline 验证后调用 renumber_results 做文档级重排。

    本用例走真实 pipeline.run（mock），要求：
    1. references 的 index 全局唯一且从 1 连续；
    2. 渲染 HTML 中出现的引用编号集合与 references 编号一致（正文↔文末对齐）。
    """
    import re as _re

    print("\n" + "=" * 70)
    print("测试 8：文档级引用全局编号（回归断言）")
    print("=" * 70)

    from app.review import create_review_pipeline

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks_1 = chunker.chunk(SAMPLE_PAPER_1, doc_id="paper_1")
    chunks_2 = chunker.chunk(SAMPLE_PAPER_2, doc_id="paper_2")
    doc_chunks = {"paper_1": chunks_1, "paper_2": chunks_2}
    doc_infos = [
        {"doc_id": "paper_1", "title": "Attention Is All You Need"},
        {"doc_id": "paper_2", "title": "BERT"},
    ]
    assert len(chunks_1) > 0 and len(chunks_2) > 0, "分块不应为空"

    embedder = FakeEmbedder(dim=64)
    retriever = InMemoryVectorRetriever(embedder)
    retriever.add_chunks(chunks_1 + chunks_2)

    # 4 节，制造多段引用以暴露"段级编号冲突"
    pipeline = create_review_pipeline(
        retriever=retriever, embedder=embedder, mode="auto", verify_citations=True,
    )
    document = pipeline.run(
        topic="深度学习模型在视觉与语言任务中的应用",
        doc_infos=doc_infos,
        doc_chunks=doc_chunks,
        num_sections=4,
        verify_citations=True,
    )

    print(f"引用统计: {document.citation_stats}")
    print(f"参考文献 {len(document.references)} 条: "
          f"{[(r['index'], r['doc_id']) for r in document.references]}")

    # 1. references index 唯一且从 1 连续（文档级全局编号）
    ref_indexes = [r["index"] for r in document.references]
    assert ref_indexes == list(range(1, len(ref_indexes) + 1)), (
        f"参考文献编号应全局唯一且从 1 连续，实际 {ref_indexes}"
    )
    # 2. 正文渲染 marker 与参考文献编号对齐
    markers = sorted({
        int(x) for x in _re.findall(r"\[(\d+)[!?]?\]", document.full_html or "")
    })
    print(f"正文引用编号: {markers}")
    assert markers == list(range(1, len(ref_indexes) + 1)), (
        f"正文引用编号 {markers} 与参考文献编号 {ref_indexes} 不一致"
    )
    # 3. 有引用且未丢条目（stat total 应等于去重后的引用数）
    assert len(ref_indexes) >= 2, "多段综述至少应有 2 条唯一引用"
    print("  ✓ 文档级引用编号唯一连续，正文 marker 与参考文献对齐（无丢条目）")


def main() -> None:
    print("=" * 70)
    print("综述生成全链路集成验证（项目文档 4.3 / 6.2 / 6.3 / 6.5 节）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)
    print("模式：mock（无 API key，本地联调）")

    t0 = time.time()

    # 准备共享数据
    doc_chunks = _build_doc_chunks()
    print(f"\n准备测试数据：{len(doc_chunks)} 篇论文，"
          f"共 {sum(len(c) for c in doc_chunks.values())} 个 chunk")

    # 逐项测试
    plan = test_outline_planner()
    test_citation_parsing()
    retriever = test_citation_verification(doc_chunks)
    test_render_helper(retriever)
    test_review_generator(doc_chunks, plan)
    test_full_review_pipeline(doc_chunks, retriever)
    test_llm_interrupt_recovery(doc_chunks)
    test_document_level_citation_numbering()

    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n{'=' * 70}")
    print(f"全部测试通过 ✓  总耗时: {elapsed_ms:.1f}ms")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

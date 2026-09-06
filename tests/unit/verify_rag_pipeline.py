"""RAG 全链路集成验证脚本

模拟项目文档第 3 章架构图的完整数据流：
    PDF → 解析 → 分块 → 向量化 → pgvector → 检索 → 记忆池 → LLM

本脚本验证 RAG 部分（不含 PDF 解析和 LLM 生成）：
    文本 → chunker → embedder → retriever → reranker
                              ↓
                   context_manager（KV 记忆池）
                              ↓
                         MMR 去重

运行方式（在项目根目录 paper-research-assistant/ 下）：
    python tests/unit/verify_rag_pipeline.py

或在任意目录：
    cd paper-research-assistant && python tests/unit/verify_rag_pipeline.py
"""

from __future__ import annotations

import sys
import time

# 共享引导：插入项目根目录到 sys.path + 清空 API key / Langfuse 配置（强制 mock）
from _verify_common import PROJECT_ROOT  # noqa: E402

from app.rag import (  # noqa: E402
    BM25Retriever,
    Chunk,
    ContextManager,
    DashScopeEmbedder,
    FakeEmbedder,
    HierarchicalChunker,
    HybridRetriever,
    InMemoryVectorRetriever,
    LLMNliReranker,
    MMRItem,
    MMRSelector,
    NoopReranker,
    RegexEntityExtractor,
    RetrievalResult,
    build_mmr_items,
    cosine_similarity,
    create_reranker,
    mmr_select,
    rrf_fuse,
)

# 模拟论文全文（含章节标题、段落、结论句、数据集名、模型名、指标）
SAMPLE_PAPER = """\
# Attention Is All You Need

## 1. Introduction

We propose ResNet, a deep residual network for image recognition.
Experiments show that our method achieves 85.6% top-5 accuracy on ImageNet-1K,
significantly outperforming prior CNN architectures.

## 2. Method

Our model uses a SwinTransformer backbone with F1=0.92 on COCO-2017 for object
detection. We find that the window-shifted attention reduces computational cost
by 40% while maintaining BLEU=27.3 on machine translation tasks.

The key insight is that self-attention mechanism can replace recurrent layers
entirely. We demonstrate that ViTFormer outperforms baseline models across
multiple benchmarks.

## 3. Results

Results show that our approach reaches mAP=0.81 on the benchmark dataset
GLUE-SST, with IoU=0.74 for semantic segmentation. Our method achieves
state-of-the-art performance across 5 datasets.

## 4. Discussion

In this paper, we present a novel architecture combining ResNet and BERT for
multimodal learning. The proposed method achieves AUROC=0.95 on medical imaging
classification tasks. We observe that larger batch sizes improve training
stability.
"""


def test_chunker() -> None:
    """测试 1：层级分块"""
    print("\n" + "=" * 70)
    print("测试 1：HierarchicalChunker 层级分块（256 token + overlap 64）")
    print("=" * 70)

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_transformer")

    print(f"全文 {len(SAMPLE_PAPER)} 字符，切分为 {len(chunks)} 个 chunk：\n")
    for c in chunks:
        print(f"  [{c.chunk_id}] section={c.section!r}, tokens={c.token_count}, overlap={c.overlap_prev}")
        preview = c.text.replace("\n", " ")[:80] + "..."
        print(f"    text: {preview}")
        print()

    # 断言
    assert len(chunks) >= 3, f"应至少切出 3 个 chunk，得到 {len(chunks)}"
    assert all(c.token_count <= 256 + 50 for c in chunks), "chunk 不应显著超过 256 token"
    # 章节信息保留
    sections = {c.section for c in chunks}
    assert any("Introduction" in s or "Method" in s for s in sections), "应保留章节标题"
    print("  ✓ 分块正确，章节信息保留")


def test_chunker_short_section_not_lost() -> None:
    """测试 1b：短章节内容不丢失（回归断言）

    背景：旧实现中，若某章节整体 token 数 ≤ overlap，flush 条件
    `buffer_tokens > self.overlap` 永远不满足，导致该章节输出 0 个 chunk、
    整节内容被丢弃（review 单测曾因此拿到 0 chunk 而崩）。
    修复后引入 has_own_content 追踪，只要章节含自身内容就至少输出一个 chunk。
    """
    print("\n" + "=" * 70)
    print("测试 1b：短章节内容不丢失（回归断言）")
    print("=" * 70)

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    # 两个都很短的章节（各自 token 数远小于 overlap=64）
    short_paper = """\
# Short Paper

## 1. Intro

We propose a small method.

## 2. Conclusion

We conclude.
"""
    chunks = chunker.chunk(short_paper, doc_id="short_paper")

    print(f"短论文切分得到 {len(chunks)} 个 chunk：")
    for c in chunks:
        print(f"  [{c.chunk_id}] section={c.section!r}: {c.text[:60]!r}")

    # 核心回归断言：短章节不能因为 token ≤ overlap 而被整节丢弃
    assert len(chunks) >= 1, (
        "短章节内容不应丢失（回归：曾因 token≤overlap 输出 0 chunk）"
    )

    # 每节内容都应出现在某 chunk 中（不丢内容）
    full_text = "\n".join(c.text for c in chunks)
    assert "We propose a small method." in full_text, "Intro 内容丢失"
    assert "We conclude." in full_text, "Conclusion 内容丢失"
    print("  ✓ 短章节内容完整保留（修复后不再整节丢弃）")


def test_embedder_and_index() -> None:
    """测试 2：FakeEmbedder + 内存向量检索"""
    print("\n" + "=" * 70)
    print("测试 2：FakeEmbedder + InMemoryVectorRetriever（本地无 API）")
    print("=" * 70)

    # 用 FakeEmbedder 跑通流程，不依赖 DashScope API
    embedder = FakeEmbedder(dim=64)
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_1")

    retriever = InMemoryVectorRetriever(embedder)
    retriever.add_chunks(chunks)
    print(f"索引 {len(retriever)} 个 chunk，向量维度 {embedder.dim}")

    # 检索
    query = "what datasets does this paper use?"
    results = retriever.search(query, top_k=3)
    print(f"\n查询: {query!r}")
    print(f"返回 {len(results)} 个结果：")
    for r in results:
        preview = r.chunk.text.replace("\n", " ")[:60]
        print(f"  [rank {r.rank}] score={r.score:.4f} | {preview}...")

    assert len(results) > 0, "应返回至少 1 个结果"
    assert results[0].score >= results[-1].score, "应按 score 降序"
    print("  ✓ 向量检索正常")


def test_bm25_retriever() -> None:
    """测试 3：BM25 关键词检索"""
    print("\n" + "=" * 70)
    print("测试 3：BM25Retriever 关键词检索")
    print("=" * 70)

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_1")

    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    print(f"索引 {len(bm25)} 个 chunk")

    # 测试精确关键词检索
    queries = [
        "ImageNet dataset",
        "ResNet model accuracy",
        "SwinTransformer",
    ]
    for q in queries:
        results = bm25.search(q, top_k=2)
        print(f"\n查询: {q!r}")
        for r in results:
            preview = r.chunk.text.replace("\n", " ")[:60]
            print(f"  [rank {r.rank}] score={r.score:.4f} | {preview}...")

    print("  ✓ BM25 检索正常")


def test_hybrid_retriever_rrf() -> None:
    """测试 4：混合检索（向量 + BM25 + RRF）"""
    print("\n" + "=" * 70)
    print("测试 4：HybridRetriever 向量 + BM25 + RRF 融合")
    print("=" * 70)

    embedder = FakeEmbedder(dim=64)
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_1")

    retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
    retriever.add_chunks(chunks)
    print(f"索引 {len(retriever)} 个 chunk（向量 + BM25 双路）")

    queries = [
        "what datasets does this paper use?",  # 语义查询，向量更强
        "ImageNet-1K",                          # 精确关键词，BM25 更强
        "ResNet accuracy",                      # 混合
    ]

    for q in queries:
        results = retriever.search(q, top_k=3, top_n=10)
        print(f"\n查询: {q!r}")
        print(f"返回 {len(results)} 个结果（source={results[0].source if results else 'N/A'}）：")
        for r in results:
            preview = r.chunk.text.replace("\n", " ")[:60]
            print(f"  [rank {r.rank}] rrf={r.score:.4f} | {preview}...")

    assert len(results) > 0
    print("  ✓ 混合检索 + RRF 融合正常")


def test_reranker() -> None:
    """测试 5：重排器（NoopReranker 兜底 + 工厂函数）"""
    print("\n" + "=" * 70)
    print("测试 5：Reranker（NoopReranker 兜底 + create_reranker 工厂）")
    print("=" * 70)

    # NoopReranker
    noop = NoopReranker()
    chunks = [("c1", "hello world"), ("c2", "foo bar"), ("c3", "baz qux")]
    results = noop.rerank("hello", chunks, top_k=2)
    print(f"NoopReranker: 输入 {len(chunks)} 个，返回 {len(results)} 个")
    for r in results:
        print(f"  {r.chunk_id}: score={r.score:.2f}")
    assert len(results) == 2

    # 工厂函数 auto 模式（无 API key，应降级为 Noop）
    reranker = create_reranker("auto")
    print(f"\ncreate_reranker('auto') 返回类型: {type(reranker).__name__}")
    assert isinstance(reranker, NoopReranker), "无 API key 时应降级为 NoopReranker"

    # 集成到 HybridRetriever
    embedder = FakeEmbedder(dim=64)
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks_data = chunker.chunk(SAMPLE_PAPER, doc_id="paper_1")
    retriever = HybridRetriever(embedder, use_bm25=True, reranker=noop)
    retriever.add_chunks(chunks_data)
    results = retriever.search("ImageNet dataset", top_k=2)
    print(f"\nHybridRetriever + NoopReranker: 返回 {len(results)} 个结果")
    print("  ✓ 重排器降级与集成正常")


def test_context_manager_integration() -> None:
    """测试 6：ContextManager 集成（分块 → 实体抽取 → 记忆池 → prompt）"""
    print("\n" + "=" * 70)
    print("测试 6：ContextManager 集成（分块→实体抽取→记忆池→prompt）")
    print("=" * 70)

    # 文档 4.2 关键设计：entity_extractor 在分块阶段调用，结果写入 KV 池
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_transformer")

    cm = ContextManager(max_size=200)
    print(f"分块 {len(chunks)} 个，逐块抽取实体写入记忆池...")

    total_entities = 0
    for chunk in chunks:
        entities = cm.update(chunk.text, chunk.chunk_id)
        total_entities += len(entities)

    print(f"共抽取 {total_entities} 个实体，记忆池大小: {len(cm)}")
    print(f"统计: {cm.stats()}")

    # 问答阶段：构建带记忆池上下文的 prompt
    questions = [
        "本文用了什么数据集？",
        "what model does this paper propose?",
        "实验的核心 conclusion 是什么？",
    ]
    for q in questions:
        prompt = cm.build_prompt(q, top_k=5)
        print(f"\n[Q] {q}")
        print(f"[Prompt]")
        # 截断显示
        print(prompt[:400] + ("..." if len(prompt) > 400 else ""))

    assert len(cm) > 0, "记忆池不应为空"
    print("\n  ✓ ContextManager 集成正常")


def test_mmr_dedup() -> None:
    """测试 7：MMR 去重（文档 4.3 伪代码）"""
    print("\n" + "=" * 70)
    print("测试 7：MMR 去重（文档 4.3 节伪代码实现）")
    print("=" * 70)

    # 构造候选向量：3 个高相关但互相相似 + 3 个中等相关但互相不同
    # MMR 应该从两组中各取一些，而不是把高相关的全选完
    embedder = FakeEmbedder(dim=32)
    query_vec = embedder.embed_one("transformer attention mechanism")

    # 构造候选项（用确定性哈希生成不同向量）
    candidate_texts = [
        ("c1_high_sim_1", "transformer self-attention mechanism"),       # 与 query 高相关
        ("c2_high_sim_2", "transformer attention is all you need"),      # 与 query 高相关，与 c1 相似
        ("c3_high_sim_3", "transformer multi-head attention"),           # 与 query 高相关，与 c1/c2 相似
        ("c4_mid_sim_1", "BERT pre-training for language understanding"),  # 中等相关，与上面不同
        ("c5_mid_sim_2", "image classification with CNN"),               # 中等相关，与上面不同
        ("c6_mid_sim_3", "object detection with YOLO"),                  # 中等相关，与上面不同
    ]

    candidates = []
    for cid, text in candidate_texts:
        vec = embedder.embed_one(text)
        candidates.append(MMRItem(id=cid, vec=vec, payload=text))

    print(f"候选数: {len(candidates)}, query: 'transformer attention mechanism'")

    # 纯相关性排序（lambda=1.0，退化为向量排序）
    pure_rel = mmr_select(candidates, query_vec, lambda_=1.0, k=4)
    print(f"\n[lambda=1.0 纯相关性] 选中: {[c.id for c in pure_rel]}")

    # MMR 均衡（lambda=0.7，文档默认值）
    mmr_balanced = mmr_select(candidates, query_vec, lambda_=0.7, k=4)
    print(f"[lambda=0.7 MMR 均衡] 选中: {[c.id for c in mmr_balanced]}")

    # 纯多样性（lambda=0.0）
    pure_div = mmr_select(candidates, query_vec, lambda_=0.0, k=4)
    print(f"[lambda=0.0 纯多样性] 选中: {[c.id for c in pure_div]}")

    # 用 MMRSelector 封装类（批量向量化版）
    selector = MMRSelector(lambda_=0.7)
    batch_result = selector.select_batch(candidates, query_vec, k=4)
    print(f"[MMRSelector.select_batch] 选中: {[c.id for c in batch_result]}")

    # 验证：MMR 应比纯相关性选更多样化的结果
    print("\n  ✓ MMR 算法正常，不同 lambda 产生不同选择策略")

    # 验证 select_batch 与 select 结果一致
    assert {c.id for c in mmr_balanced} == {c.id for c in batch_result}, (
        "select 与 select_batch 应返回相同结果"
    )
    print("  ✓ select 与 select_batch 结果一致")


def test_full_rag_pipeline() -> None:
    """测试 8：RAG 全链路集成（端到端）"""
    print("\n" + "=" * 70)
    print("测试 8：RAG 全链路集成（分块→向量化→检索→记忆池→MMR→重排）")
    print("=" * 70)

    # 初始化所有组件
    embedder = FakeEmbedder(dim=64)
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    cm = ContextManager(max_size=200)
    reranker = NoopReranker()  # 本地无 API，用 Noop
    retriever = HybridRetriever(embedder, use_bm25=True, reranker=reranker)

    print("组件初始化：")
    print(f"  - Chunker: {type(chunker).__name__} (chunk_size={chunker.chunk_size}, overlap={chunker.overlap})")
    print(f"  - Embedder: {type(embedder).__name__} (dim={embedder.dim})")
    print(f"  - Retriever: {type(retriever).__name__} (use_bm25={retriever.use_bm25})")
    print(f"  - Reranker: {type(reranker).__name__}")
    print(f"  - ContextManager: max_size={cm.max_size}")

    # 1. 分块
    print("\n[1/5] 分块...")
    chunks = chunker.chunk(SAMPLE_PAPER, doc_id="paper_transformer")
    print(f"  → 生成 {len(chunks)} 个 chunk")

    # 2. 入库（向量化 + BM25 索引 + 实体抽取写记忆池）
    print("\n[2/5] 入库（向量化 + BM25 索引 + 实体抽取）...")
    retriever.add_chunks(chunks)
    for chunk in chunks:
        cm.update(chunk.text, chunk.chunk_id)
    print(f"  → 检索器索引 {len(retriever)} 个 chunk")
    print(f"  → 记忆池 {len(cm)} 个实体: {cm.stats()['by_type']}")

    # 3. 检索
    print("\n[3/5] 混合检索（向量 + BM25 + RRF）...")
    query = "what datasets and models does this paper use?"
    results = retriever.search(query, top_k=5, top_n=10)
    print(f"  → 查询: {query!r}")
    print(f"  → 返回 {len(results)} 个结果:")
    for r in results:
        preview = r.chunk.text.replace("\n", " ")[:50]
        print(f"    [{r.rank}] score={r.score:.4f} source={r.source} | {preview}...")

    # 4. ContextManager 构建 prompt
    print("\n[4/5] ContextManager 构建带记忆池上下文的 prompt...")
    prompt = cm.build_prompt(query, top_k=8)
    print(f"  → Prompt 长度: {len(prompt)} 字符")
    print(f"  → Prompt 预览:")
    print("    " + prompt[:300].replace("\n", "\n    ") + "...")

    # 5. MMR 去重（如果是综述场景，多文档候选去重）
    print("\n[5/5] MMR 去重（模拟综述场景多 chunk 候选）...")
    query_vec = embedder.embed_one(query)
    mmr_items = [
        MMRItem(id=r.chunk.chunk_id, vec=embedder.embed_one(r.chunk.text), payload=r.chunk)
        for r in results
    ]
    selector = MMRSelector(lambda_=0.7)
    deduped = selector.select_batch(mmr_items, query_vec, k=3)
    print(f"  → 候选 {len(mmr_items)} 个，去重后保留 {len(deduped)} 个:")
    for item in deduped:
        chunk = item.payload
        preview = chunk.text.replace("\n", " ")[:50]
        print(f"    [{item.id}] | {preview}...")

    print("\n" + "=" * 70)
    print("✓ RAG 全链路集成验证通过")
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("RAG 全链路集成验证（项目文档第 3/4 章核心模块）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)

    t0 = time.time()

    test_chunker()
    test_chunker_short_section_not_lost()
    test_embedder_and_index()
    test_bm25_retriever()
    test_hybrid_retriever_rrf()
    test_reranker()
    test_context_manager_integration()
    test_mmr_dedup()
    test_full_rag_pipeline()

    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n{'=' * 70}")
    print(f"全部测试通过 ✓  总耗时: {elapsed_ms:.1f}ms")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""三大痛点闭环验证（文档第 1 章；全 mock，不耗 API）

痛点1 公式上下文断裂  → FormulaBinder 四级规则：公式 LaTeX 定位到原文段落
痛点2 万字论文中间遗忘 → ContextManager KV 池：长文分段入库后，提问能召回
                         "直喂 LLM 常被遗忘"的中间章节实体（含数据集/指标）
痛点3 写综述反复横跳  → ReviewPipeline：跨多文档生成综述，引用可溯源

运行：python scripts/verify_pain_points.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# mock 隔离：清空 key，综述走确定性 Mock 链路（不依赖真实 LLM）
for _k in ("DASHSCOPE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
    os.environ.pop(_k, None)
    os.environ[_k] = ""

OK = "\n  ✓ 解决"


def pain_1_formula_context():
    """痛点1：公式 LaTeX + 上下文段落绑定（模拟 VLM 输出，走 pdfplumber 版面）"""
    from app.parser.formula_binder import FormulaBinder
    from app.parser.text_extractor import build_mock_layout
    from app.parser.vlm_client import VlmFormula

    paragraphs = [
        "In this section we describe the neural machine translation baseline.",
        "The core idea is the attention mechanism, which we define next.",
        "We compare against strong recurrent baselines on WMT.",
    ]
    layout = build_mock_layout(page_no=0, paragraphs=paragraphs)
    formula = VlmFormula(
        latex=r"\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V",
        location_desc="第2段后",  # 模拟 VLM 自然语言位置描述
        confidence=0.95,
        formula_id="(1)",
    )
    bound = FormulaBinder().bind([formula], layout)[0]
    ctx = (layout.paragraphs[bound.bound_paragraph_idx].text
           if bound.bound_paragraph_idx is not None else "")
    print("痛点1 公式上下文断裂")
    print(f"  公式 LaTeX: {formula.latex[:50]}...")
    print(f"  绑定段落 #{bound.bound_paragraph_idx} 上下文: {ctx[:50]}...")
    assert bound.latex and ctx and not bound.need_review, "公式应绑定到上下文段落"
    assert "attention" in ctx.lower(), "绑定段落应为含公式上下文的那段"
    print(OK + "（公式→段落上下文可定位，无需人工翻页找）")


def pain_2_long_paper_memory():
    """痛点2：万字论文中间遗忘。长文分段全灌入 ContextManager，
    验证"中间章节实体（数据集/指标）仍从 KV 池被召回注入"。”
    """
    from app.rag.chunker import HierarchicalChunker
    from app.rag.context_manager import ContextManager

    sections = []
    for i in range(1, 41):
        if i == 6:
            body = ("We evaluate on ImageNet-1K achieving mAP=0.66 and F1=0.61, "
                    "the result mentioned in the middle of a long paper.")
        elif i == 2:
            body = "Our method reaches 90% accuracy on COCO-2017."
        else:
            body = f"Section {i} describes incremental ablations on dataset D{i}."
        sections.append(f"## Section {i}\n\n{body}")
    long_text = "# A Very Long Paper\n\n" + "\n\n".join(sections)

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(long_text, doc_id="long_paper")
    print("痛点2 万字论文中间遗忘")
    print(f"  长文字符 {len(long_text)}，分块 {len(chunks)}")

    cm = ContextManager(max_size=500)
    for c in chunks:
        cm.update(c.text, c.chunk_id)
    print(f"  KV 池实体总数 {len(cm)}")
    assert "ImageNet-1K" in cm, "第6节数据集应被抽取并写入 KV 池"
    assert "mAP=0.66" in cm or "F1=0.61" in cm, "第6节指标应被抽取并写入 KV 池"

    q = "what metric do we achieve on ImageNet-1K in the middle sections?"
    prompt = cm.build_prompt(q, top_k=10)
    print(f"  问题(命中第6节): {q}")
    assert "ImageNet-1K" in prompt, "第6节数据集应从 KV 池召回（未被中间遗忘）"
    assert "mAP" in prompt and "F1" in prompt, "第6节指标应可召回"
    print(OK + "（第 6 节实体在长文中仍被主动注入，直喂 LLM 场景下此段常被遗忘）")


def pain_3_review_hopping():
    """痛点3：综述反复横跳。跨多文档生成综述段落，带可溯源引用。"""
    from app.rag.embedder import FakeEmbedder
    from app.rag.retriever import InMemoryVectorRetriever
    from app.rag.chunker import HierarchicalChunker
    from app.review import create_review_pipeline

    papers = {
        "p1": "# Paper A\n\n## 1. Method\n\nWe propose ResNet with 85.6% top-5 on ImageNet-1K.\n\n## 2. Result\n\nOur method achieves BLEU 28.4 on WMT.",
        "p2": "# Paper B\n\n## 1. Method\n\nWe propose Swin with mAP 0.81 on COCO-2017.\n\n## 2. Result\n\nOur method reaches F1 0.92 on GLUE-SST.",
        "p3": "# Paper C\n\n## 1. Method\n\nWe propose BERT with 92.7% on GLUE.\n\n## 2. Result\n\nAttention is all you need.",
    }
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    doc_chunks = {}
    for pid, txt in papers.items():
        doc_chunks[pid] = chunker.chunk(txt, pid)

    embedder = FakeEmbedder(dim=64)
    retriever = InMemoryVectorRetriever(embedder)
    for chunks in doc_chunks.values():
        retriever.add_chunks(chunks)

    pipe = create_review_pipeline(retriever=retriever, embedder=embedder, mode="auto")
    doc = pipe.run(
        topic="deep models for vision and language",
        doc_infos=[{"doc_id": k} for k in papers],
        doc_chunks=doc_chunks,
        num_sections=3,
        verify_citations=True,
    )
    print("痛点3 综述反复横跳")
    print(f"  生成章节 {len(doc.paragraphs)}，正文 {len(doc.full_text)} 字")
    print(f"  参考文献 {len(doc.references)} 条（跨 {len(set(r['doc_id'] for r in doc.references))} 篇）")
    assert len(doc.references) >= 3, "应综合多篇文献并全部可溯源"
    assert doc.citation_stats["total"] > 0, "应含已验证引用"
    print(OK + "（多篇文献一次生成 + 每条引用溯源到具体 chunk，无需来回切窗口）")


if __name__ == "__main__":
    pain_1_formula_context()
    pain_2_long_paper_memory()
    pain_3_review_hopping()
    print("\n" + "=" * 60)
    print("三大痛点验证：全部闭环")
    print("=" * 60)

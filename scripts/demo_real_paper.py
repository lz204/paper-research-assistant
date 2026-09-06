"""真实 Transformer 论文演示脚本（上传 → 解析 → 检索 → 问答）

用法（在 paper-research-assistant/ 根目录）：
    python scripts/demo_real_paper.py

流程：
  1. 读取 sample_papers/attention_is_all_you_need.pdf（真实 15 页论文）
  2. PyMuPDF 抽取文本 → HierarchicalChunker 分块（真实编排，与 /upload 同链路）
  3. HybridRetriever（向量 + BM25 + RRF）检索
     - embedding 优先 DashScope 真实 API；不可用时降级 FakeEmbedder 并提示
  4. DashScope Qwen-Plus 生成回答（失败自动 partial 降级）

只读 + 少量 API 调用，可安全重复运行。
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# 让 app.* 可 import（并触发 app/__init__ 加载 .env）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF

PDF_URL = "https://arxiv.org/pdf/1706.03762"
PDF_PATH = PROJECT_ROOT / "sample_papers" / "attention_is_all_you_need.pdf"


def ensure_pdf() -> Path:
    """本地不存在时从 arXiv 下载（版权 PDF 不入库，运行时按需获取）"""
    if PDF_PATH.exists():
        return PDF_PATH
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"未找到本地 PDF，正在从 arXiv 下载：{PDF_URL}")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)  # noqa: S310 - 仅下载公开 arXiv 论文
    return PDF_PATH


def step(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    try:
        pdf_path = ensure_pdf()
    except Exception as e:  # noqa: BLE001 - 网络不可达时给出明确提示
        print(f"PDF 下载失败：{e}")
        print(f"请手动下载 {PDF_URL} 放到 {PDF_PATH}")
        return 1

    # 1) 读取真实 PDF
    step("Step 1 · 读取真实 PDF（Attention Is All You Need, arXiv 1706.03762）")
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    page_texts = []
    for i in range(num_pages):
        page_texts.append(doc.load_page(i).get_text("text") or "")
    full_text = "\n\n".join(page_texts)
    doc.close()
    print(f"页数: {num_pages}，抽取字符: {len(full_text)}")
    print(f"首行: {full_text.strip().splitlines()[0][:80]}")

    # 2) 分块（真实编排链路：/upload 同款 chunker）
    step("Step 2 · 层级分块")
    from app.rag.chunker import HierarchicalChunker
    doc_id = "attention_is_all_you_need"
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(full_text, doc_id=doc_id)
    print(f"chunks: {len(chunks)}")
    for c in chunks[:3]:
        print(f"  [{c.chunk_id}] sec={c.section!r}: {c.text[:70]!r}...")

    # 3) Embedding + 检索
    step("Step 3 · 混合检索（向量 + BM25 + RRF）")
    has_real_key = bool(os.getenv("DASHSCOPE_API_KEY", ""))
    print(f"DASHSCOPE_API_KEY 存在: {has_real_key}")

    embedder = None
    if has_real_key:
        try:
            from app.rag.embedder import DashScopeEmbedder
            emb = DashScopeEmbedder()
            if emb._ensure_client():
                embedder = emb
                print("Embedding: DashScope text-embedding-v3（真实 API）")
        except Exception as e:  # noqa: BLE001
            print(f"DashScope embedder 初始化失败: {e}")

    if embedder is None:
        from app.rag.embedder import FakeEmbedder
        embedder = FakeEmbedder(dim=64)
        print("Embedding: FakeEmbedder（无语义，仅流程演示；配置可用的 embedding key 后自动真实）")

    from app.rag.retriever import HybridRetriever
    retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
    retriever.add_chunks(chunks)

    questions = [
        "What dataset and metrics are used in the WMT 2014 experiments?",
        "How is the model trained? What optimizer and learning rate schedule are used?",
    ]
    for q in questions:
        print(f"\n问题: {q}")
        results = retriever.search(q, top_k=3)
        print(f"召回 {len(results)} 段：")
        for r in results:
            snippet = r.chunk.text.replace("\n", " ")[:120]
            print(f"  [{r.source}] {snippet}...")

        # 4) LLM 生成
        step("Step 4 · DashScope Qwen-Plus 回答")
        from app.llm import create_llm_client
        client = create_llm_client("auto")
        ctx = "\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in results)[:3000]
        system = "你是学术问答助手。仅根据给出的论文原文片段回答问题，用中文，控制在 120 字内。若片段不足以回答请说明。"
        prompt = (
            f"论文片段：\n---\n{ctx}\n---\n\n问题：{q}\n\n请回答："
        )
        resp = client.chat(prompt, system=system, temperature=0.2, max_tokens=256)
        print(f"[回答] {resp.text.strip()}")
        if resp.partial:
            print(f"[提示] LLM 返回部分结果: {resp.error}")
        print(f"[成本] {resp.cost_yuan:.4f} 元 | from_cache={resp.from_cache}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

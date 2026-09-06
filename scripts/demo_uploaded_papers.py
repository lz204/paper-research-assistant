# -*- coding: utf-8 -*-
"""用户上传论文集 → RAG 问答流程验证（真实 PDF + 真实 Embedding/LLM）

对 sample_papers/ 下的真实 arXiv 论文执行完整流程：
  解析(页数/分块) → DashScope embedding 入库 → HybridRetriever 检索 → Qwen-Plus 回答

用法：
    python scripts/demo_uploaded_papers.py                 # 用 Attention 论文演示
    python scripts/demo_uploaded_papers.py --limit 3       # 处理前 3 篇（真实 API，注意成本）
    python scripts/demo_uploaded_papers.py --paper <关键词> # 指定论文（文件名子串）

单篇：2 次 LLM 调用、全篇一次 embedding（成本可控）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz

PDF_DIR = ROOT / "sample_papers"


def extract_pdf(path: Path):
    doc = fitz.open(str(path))
    n = len(doc)
    text = "\n\n".join(doc.load_page(i).get_text("text") or "" for i in range(n))
    doc.close()
    return n, text


def build_index(full_text: str, doc_id: str):
    """分块 + 真实 embedding + HybridRetriever"""
    from app.rag.chunker import HierarchicalChunker
    from app.rag.embedder import DashScopeEmbedder
    from app.rag.retriever import HybridRetriever

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(full_text, doc_id=doc_id)
    embedder = DashScopeEmbedder()
    retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
    retriever.add_chunks(chunks)
    return retriever


def ask(retriever, question: str, doc_id: str) -> None:
    from app.llm import create_llm_client

    results = retriever.search(question, top_k=3)
    if not results:
        print("  检索无结果")
        return
    print(f"  召回 {len(results)} 段：")
    for r in results:
        print(f"    [{r.source}] {r.chunk.text[:90].replace(chr(10), ' ')}...")
    ctx = "\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in results)[:3000]
    client = create_llm_client("auto")
    system = "你是学术问答助手，仅根据论文原文片段用中文回答，120 字内；信息不足请说明。"
    resp = client.chat(
        f"论文片段：\n---\n{ctx}\n---\n\n问题：{question}", system=system,
        temperature=0.2, max_tokens=256,
    )
    print(f"  [回答] {resp.text.strip()}")
    print(f"  [成本] {resp.cost_yuan:.4f} 元 | cache={resp.from_cache}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="attention_is_all_you_need",
                    help="论文文件名子串（默认 Attention）")
    ap.add_argument("--limit", type=int, default=None,
                    help="从第一个匹配论文开始最多处理几篇")
    args = ap.parse_args(argv)

    candidates = sorted(p for p in PDF_DIR.glob("*.pdf") if args.paper.lower() in p.name.lower())
    if not candidates:
        print(f"未找到含 '{args.paper}' 的 PDF，可用：")
        for p in sorted(PDF_DIR.glob("*.pdf"))[:30]:
            print("  ", p.name)
        return 1
    candidates = candidates[: args.limit] if args.limit else candidates[:1]
    print(f"处理论文：{len(candidates)} 篇")

    for p in candidates:
        print("\n" + "=" * 76)
        print(f"论文：{p.name}")
        pages, full_text = extract_pdf(p)
        print(f"页数 {pages}，字符 {len(full_text)}")
        doc_id = p.stem[:20]
        retriever = build_index(full_text, doc_id)
        print("已入库，开始问答：")
        # 针对各论文提问数据集/方法，验证"上传真实 PDF → 检索 → 回答"全链路
        ask(retriever, f"这篇论文 {p.stem[:40].replace('_', ' ')} 用了什么数据集与评价指标？主要方法是什么？", doc_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

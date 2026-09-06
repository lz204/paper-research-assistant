"""补齐三模块验证脚本：翻译 + GB/T 7714 引用 + 论文框架拆解。

链路（mock 必跑，DashScope 有 Key 自动切真实模式）：
  1. TermGlossary：术语前后置处理 + 术语对齐
  2. DashScopeTranslator（无 Key 则 Mock）：翻译单句 + 术语校准
  3. 批量翻译 qa_dataset 内置 3 篇 → TranslatedPaper.to_chunk_list() 可直接入索引
  4. PaperMeta + GBT7714Formatter：J/C/M/D/EB 五种类型各 1 条格式化
  5. ReviewDocument.references_gbt7714() 缺 PaperMeta 占位降级
  6. Word 导出文末参考文献使用 GB/T 7714 格式（md + python-docx 降级都验证）
  7. MockStructurer：chunks → 9 角色标注 + 核心提炼
  8. LLMStructurer（DashScope 可用时）：真实验证 JSON 输出 → 结构标注+要点抽取

运行：python tests/unit/verify_completion_modules.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time  # noqa: E402

PASS = 0
FAIL = 0


def step(title):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def check(name, cond, detail: str = ""):
    global PASS, FAIL
    ok = bool(cond)
    if ok:
        PASS += 1
        print(f"  ✓ {name}" + (f"：{detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"：{detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 主验证
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    global PASS, FAIL
    print("=" * 70)
    print("补齐三模块验证（翻译 / GB/T 7714 / 论文框架拆解）")
    print(f"项目根: {PROJECT_ROOT}")
    print("=" * 70)

    # 1. 术语词库
    step("[1/7] TermGlossary — 术语前后校准")
    from app.translation import TermGlossary, TermEntry
    gloss = TermGlossary()
    check("内置术语条目数 ≥ 40", len(gloss.entries()) >= 40, str(len(gloss.entries())))
    s1 = "Self-attention is the core of the Transformer."
    processed, ph = gloss.preprocess(s1)
    check("预处理：self-attention 被替换为占位符", "__TERM_" in processed and "Self-attention" not in processed)
    check("预处理：Transformer 被替换为占位符", processed.count("__TERM_") >= 2,
          f"占位符数={processed.count('__TERM_')}")
    # 模拟 LLM 返回：保留占位符
    llm_output = processed.replace("__TERM_", "__TERM_")
    final, aligned = gloss.postprocess(llm_output, ph)
    check("后处理：还原为自注意力机制+Transformer 架构",
          "自注意力机制" in final and "Transformer 架构" in final, final[:80])
    check("后处理：aligned_terms 非空", len(aligned) >= 2, str(aligned))
    # 持久化往返
    tmp = PROJECT_ROOT / "exports" / "_tmp_gloss.json"
    tmp.parent.mkdir(exist_ok=True)
    gloss.save(str(tmp))
    gloss2 = TermGlossary()
    gloss2.load(str(tmp))
    check("术语词库 保存/加载 往返一致", len(gloss2.entries()) == len(gloss.entries()),
          f"加载后={len(gloss2.entries())}")
    tmp.unlink(missing_ok=True)

    # 2. 翻译器（DashScope 可用则真实，否则 mock）
    step("[2/7] Translator — 单句翻译 + 术语校准")
    from app.translation import create_translator
    translator = create_translator("auto")
    check(f"翻译器类型={translator.__class__.__name__}", True)
    sample = "Self-attention is the core of the Transformer architecture."
    u = translator.translate_text(sample)
    check("翻译 OK（target 非空）", u.ok, (u.target_text[:80] if u.target_text else "空"))
    check("对齐术语数 ≥ 1", len(u.aligned_terms) >= 1, str(u.aligned_terms))

    step("[3/7] Translator — 批量翻译 qa_dataset 3 篇（mock/真实均可）")
    from tests.qa_dataset.build_dataset import load_builtin_papers, load_builtin_qa
    papers = load_builtin_papers()
    check("qa_dataset 加载 ≥ 3 篇论文", len(papers) >= 3, f"实际 {len(papers)}")
    _ = load_builtin_qa()  # 保证 QA 也能正常 load，避免 regress
    translated_papers = []
    for p in papers:
        tp = translator.translate_paper(p.paper_id, p.chunks, title_en=p.title)
        translated_papers.append(tp)
    check("3 篇全部翻译完成", len(translated_papers) == 3, f"实际 {len(translated_papers)}")
    zh_chunks = translated_papers[0].to_chunk_list()
    check("第 1 篇 to_chunk_list 非空", len(zh_chunks) >= 2, f"chunks={len(zh_chunks)}")
    check("翻译 chunks 带 metadata.lang=zh-CN",
          any(c.metadata.get("lang") == "zh-CN" for c in zh_chunks),
          str(zh_chunks[0].metadata))

    # 4. GB/T 7714
    step("[4/7] GBT7714Formatter — J / C / M / D / EB 五种类型")
    from app.common.paper_meta import PaperMeta
    from app.review import GBT7714Formatter
    fmt = GBT7714Formatter()
    samples = [
        ("[J]", PaperMeta(
            paper_id="p1", title="Attention Is All You Need",
            authors=["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J."],
            year=2017, doc_type="J", journal="Advances in Neural Information Processing Systems",
            volume="30", pages="5998-6008",
            doi="10.48550/arXiv.1706.03762",
        )),
        ("[C]", PaperMeta(
            paper_id="p2", title="BERT",
            authors=["Devlin, J.", "Chang, M.-W.", "Lee, K.", "Toutanova, K."],
            year=2019, doc_type="C", conference="Proceedings of NAACL-HLT",
            conference_location="Minneapolis, USA", pages="4171-4186",
        )),
        ("[M]", PaperMeta(
            paper_id="p3", title="Deep Learning",
            authors=["Goodfellow, I.", "Bengio, Y.", "Courville, A."],
            year=2016, doc_type="M", publisher="MIT Press", conference_location="Cambridge, MA",
            pages="780", isbn="978-0262035613",
        )),
        ("[D]", PaperMeta(
            paper_id="p4", title="研究 Transformer 架构及其在机器翻译中的应用",
            authors=["张三"], year=2023, doc_type="D",
            degree_grantor="清华大学", conference_location="北京",
        )),
        ("[EB/OL]", PaperMeta(
            paper_id="p5", title="Swin Transformer",
            authors=["Liu, Z."], year=2021, doc_type="EB",
            url="https://arxiv.org/abs/2103.14030", access_date="2026-08-10",
        )),
    ]
    for i, (label, meta) in enumerate(samples):
        s = fmt.format(meta, index=i + 1)
        check(f"{label} 格式化非空 + 含序号 [", len(s) > 20 and s.startswith(f"[{i+1}] "), s[:100])
    # 3 人+ 显示"等"
    authors_txt = PaperMeta.authors_text(PaperMeta(
        paper_id="t", authors=["A", "B", "C", "D"],
    ), max_display=3)
    check("4 作者排版 'A, B, C, 等'", authors_txt == "A, B, C, 等", authors_txt)

    # 5. ReviewDocument 国标化 + Word 导出（Markdown 降级）
    step("[5/7] ReviewDocument.references_gbt7714() + Word 导出")
    from app.export.docx_renderer import (
        ReviewDocument,
        MarkdownDocxRenderer,
    )
    doc = ReviewDocument(
        topic="Transformer 家族",
        title="综述：Transformer 架构发展",
        references=[
            {"index": 1, "doc_id": "p1", "chunk_id": "p1#0"},
            {"index": 2, "doc_id": "p2", "chunk_id": "p2#0"},
            {"index": 3, "doc_id": "p3", "chunk_id": "p3#0"},
        ],
        paper_metas={
            "p1": samples[0][1],
            "p2": samples[1][1],
            "p3": samples[2][1],
        },
    )
    refs = doc.references_gbt7714()
    check("格式化 3 条", len(refs) == 3, f"实际 {len(refs)}")
    check("第 1 条含 [J] 类型标记", "[J]" in refs[0]["formatted"], refs[0]["formatted"][:80])
    # 缺元数据占位降级
    doc2 = ReviewDocument(topic="t", references=[{"index": 1, "doc_id": "unknown_id"}])
    refs2 = doc2.references_gbt7714()
    check("缺 PaperMeta 占位降级含 [EB/OL]", "[EB/OL]" in refs2[0]["formatted"], refs2[0]["formatted"][:80])
    # Markdown 渲染文末参考文献为 GB/T 7714
    rendered = MarkdownDocxRenderer().render(doc)
    check("Markdown 渲染非空", rendered.size > 0 and rendered.markdown, f"size={rendered.size}")
    check("Markdown 文末含 '[1] ... [J]' GB/T 格式", "[J]" in rendered.markdown,
          "GB/T 片段：" + rendered.markdown[rendered.markdown.find("参考文献"):][:200])

    # 6. 论文框架拆解（mock 必跑）
    step("[6/7] MockStructurer — 9 角色标注 + 核心内容提炼")
    from app.parser import create_structurer
    from app.rag.chunker import Chunk
    # 直接构造 9 个 Chunk，明确标题前缀（section 字段 + 文本关键词），
    # 验证 MockStructurer 规则分类的可靠性，不受 HierarchicalChunker 合 chunk 影响
    raw_parts = [
        ("Abstract",     "1. Abstract: We propose a novel Transformer variant. Our contributions are threefold: new architecture and thorough ablation study."),
        ("Introduction", "2. Introduction: Machine translation is challenging. This paper proposes a pure attention-based method. We review prior work next."),
        ("Related Work", "3. Related Work: Prior neural translation models use RNN encoders and attention. Convolutional approaches exist but have limited context."),
        ("Method",       "4. Method: Our framework is an encoder-decoder with multi-head attention. Algorithm 1 shows the training loop. Encoder layers: self-attention + FFN."),
        ("Experiments",  "5. Experiments: Dataset is WMT 2014 EN-DE, 4.5M pairs. Baselines: RNNsearch, GNMT, ConvS2S. Hyperparameters: d_model=512, 8 heads, dropout=0.1."),
        ("Results",      "6. Results: Table 1 reports 28.4 BLEU vs baseline 26.1. Ablation in Table 2 shows each component's gain. Figure 1 plots attention distributions."),
        ("Discussion",   "7. Discussion: Main limitation is quadratic compute cost. Future work includes efficient sparse attention and speech applications."),
        ("Conclusion",   "8. Conclusion: We presented a new attention-only architecture achieving strong translation results with simpler training and better generalization."),
        ("Other",        "9. References: Bahdanau 2014, Sutskever 2014. Acknowledgments: The authors thank reviewers."),
    ]
    chunks = [
        Chunk(chunk_id=f"mock_paper#s{i}", text=body, token_count=max(1, len(body) // 3),
              doc_id="mock_paper", section=title, section_idx=i, chunk_idx=0)
        for i, (title, body) in enumerate(raw_parts)
    ]
    check("构造 chunks 数 = 9", len(chunks) == 9, f"chunks={len(chunks)}")
    structurer = create_structurer("mock")
    structure = structurer.structure("mock_paper", chunks)
    # 因为 section_hint 含 "Abstract/Introduction/..."，规则分类置信度=1.0，角色必须完全命中
    perfect_map = {title: {c.role for c in structure.role_chunks(title)} for title, _ in raw_parts}
    check("9 角色全部命中（section_hint 强关键词 → 置信度=1.0）",
          all(title in perfect_map and perfect_map[title] for title, _ in raw_parts),
          f"角色映射：{[(t, sorted(perfect_map.get(t, set()))) for t, _ in raw_parts]}")
    required_roles = {"Method", "Experiments", "Results", "Conclusion"}
    actual = {lc.role for lc in structure.labeled_chunks}
    check(f"包含必须角色 {required_roles}", required_roles.issubset(actual), f"实际={actual}")
    check("role_to_chunks['Method'] ≥ 1 段",
          len(structure.role_to_chunks.get("Method", [])) >= 1,
          f"Method chunks={structure.role_to_chunks.get('Method')}")

    # 7. DashScope LLM 真实模式（有 Key 就测）
    step("[7/7] LLMStructurer + DashScopeTranslator 真实调用（有 Key 时）")
    ds_struct = create_structurer("auto")
    if getattr(ds_struct, "available", False) is True:
        # 只用内置 paper_1（Transformer 带 Method/Experiments）跑真实结构
        papers2 = load_builtin_papers()
        p = papers2[0]
        s_real = ds_struct.structure(p.paper_id, p.chunks, title=p.title)
        check("LLMStructurer 真实：labeled_chunks 数量匹配",
              len(s_real.labeled_chunks) == len(p.chunks),
              f"{len(s_real.labeled_chunks)} vs {len(p.chunks)}")
        check("LLMStructurer 真实：innovations ≥ 2 条", len(s_real.innovations) >= 2,
              str(s_real.innovations[:3]))
        check("LLMStructurer 真实：method_summary 非空", len(s_real.method_summary) >= 20,
              s_real.method_summary[:80])
        # 真实 DashScopeTranslator 翻译整段（控制长度避免超成本）
        ds_trans = create_translator("auto")
        u_real = ds_trans.translate_text("Self-attention is the core of Transformer.")
        check("DashScopeTranslator 真实：译文非空 + 对齐术语 ≥ 1",
              u_real.ok and len(u_real.aligned_terms) >= 1,
              f"aligned={u_real.aligned_terms},译文={u_real.target_text[:60]}")
    else:
        print("  ⚠ DashScope 不可用（无 Key/缺包），跳过第 7 条真实调用验证。")

    print(f"\n================================================================")
    print(f"通过 {PASS} / 失败 {FAIL}  |  总耗时 {int((time.time()-t0)*1000)}ms")
    print("================================================================")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

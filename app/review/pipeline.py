"""综述主流程编排模块（项目文档 4.3 节 + 第 3 章架构图 + 6.1 节可观测性）

文档第 3 章架构图：
    Review[综述主题] --> RAG2[多文档检索] --> MMR[MMR 去重]
    MMR --> Gen[Map-Reduce 生成] --> Verify[引用事实验证]
    Verify --> Docx[Word 导出]

文档 4.3 节"综述生成 + MMR 去重"流程：
- 初版问题：Map-Reduce 直接生成，结果"车轱辘话"——10 篇文献 6 篇观点相似
- 优化一：MMR 去重（generator 已实现）
- 优化二：引用事实验证（citation_verifier 已实现）

文档 6.1 节可观测性：主流程编排中接入结构化日志 + 成本追踪 + Langfuse trace。

本模块设计：
- ReviewPipeline：综述生成主流程编排器
    1. 大纲规划（outline_planner）
    2. Map-Reduce 生成（generator）
    3. 引用事实验证（citation_verifier）
    4. 组装 ReviewDocument
- ReviewDocument 在 app.export.docx_renderer 中定义，本模块负责构造
- create_review_pipeline：工厂函数

工作流程（文档 4.3 节）：
    topic + doc_infos + doc_chunks
        ↓
    outline_planner.plan()  → OutlinePlan
        ↓
    generator.generate()    → GenerationResult（paragraphs）
        ↓
    citation_verifier.verify_paragraph() per para → VerificationResult
        ↓
    组装 ReviewDocument（outline + paragraphs + verified + full_html + citation_stats）
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 主流程编排器
# ---------------------------------------------------------------------------


class ReviewPipeline:
    """综述生成主流程编排器

    工作流程（文档 4.3 节）：
        topic + doc_infos + doc_chunks
            → [1] outline_planner.plan()    → OutlinePlan
            → [2] generator.generate()      → GenerationResult
            → [3] citation_verifier.verify_paragraph() per para
            → [4] 组装 ReviewDocument

    Args:
        outline_planner: 大纲规划器（默认 create_outline_planner("auto")）
        generator: 综述生成器（默认 create_review_generator("auto")）
        citation_verifier: 引用验证器（None 时跳过验证）
        cost_tracker: 成本追踪器（None 时不追踪）
        langfuse_client: Langfuse 客户端（None 时不埋点）

    用法：
        pipeline = ReviewPipeline()
        doc = pipeline.run(
            topic="Transformer 在医学影像分割的应用",
            doc_infos=[{"doc_id": "p1", "title": "..."}],
            doc_chunks={"p1": [Chunk(...), ...]},
        )
        print(doc.full_text)
    """

    def __init__(
        self,
        outline_planner=None,
        generator=None,
        citation_verifier=None,
        cost_tracker=None,
        langfuse_client=None,
    ):
        # 延迟导入避免循环依赖
        from .outline_planner import create_outline_planner
        from .generator import create_review_generator

        self.outline_planner = outline_planner or create_outline_planner("auto")
        self.generator = generator or create_review_generator("auto")
        self.citation_verifier = citation_verifier

        # 可观测性接入（文档 6.1 节）
        if cost_tracker is not None:
            self._cost_tracker = cost_tracker
        else:
            try:
                from ..observability import get_cost_tracker
                self._cost_tracker = get_cost_tracker()
            except ImportError:
                self._cost_tracker = None

        if langfuse_client is not None:
            self._langfuse = langfuse_client
        else:
            try:
                from ..observability import get_langfuse_client
                self._langfuse = get_langfuse_client()
            except ImportError:
                self._langfuse = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        topic: str,
        doc_infos: List[Dict],
        doc_chunks: Dict[str, List],
        num_sections: int = 4,
        title: Optional[str] = None,
        verify_citations: bool = True,
    ) -> Any:
        """运行综述生成全流程

        Args:
            topic: 综述主题（限 500 字，文档 6.5 节安全限制）
            doc_infos: 文献信息列表，每项至少含 doc_id
            doc_chunks: {doc_id: [Chunk, ...]}，每篇文献的 chunk 列表
            num_sections: 期望章节数
            title: 综述标题（None 时用主题作为标题）
            verify_citations: 是否做引用事实验证（文档 4.3 节）
        Returns:
            ReviewDocument（来自 app.export.docx_renderer）
        """
        # 延迟导入避免循环依赖
        from ..export.docx_renderer import ReviewDocument

        # 可观测性：开始 trace（文档 6.1 节）
        trace = None
        if self._langfuse is not None:
            trace = self._langfuse.start_trace(
                name="review.pipeline",
                metadata={"topic": topic[:100], "num_docs": len(doc_infos)},
            )

        t_pipeline_start = time.time()
        logger.info(
            "综述主流程开始: topic=%s, num_docs=%d, num_chunks=%d",
            topic[:50], len(doc_infos),
            sum(len(c) for c in doc_chunks.values()),
        )

        # ------------------------------------------------------------------
        # 1. 大纲规划
        # ------------------------------------------------------------------
        t0 = time.time()
        if trace is not None:
            with self._langfuse.span(trace, "outline.plan", input={"topic": topic}) as span:
                plan = self.outline_planner.plan(topic, doc_infos, num_sections)
                span.output = {"num_sections": len(plan)}
        else:
            plan = self.outline_planner.plan(topic, doc_infos, num_sections)

        outline_latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "大纲规划完成: %d 个章节, latency_ms=%d",
            len(plan), outline_latency_ms,
        )

        # ------------------------------------------------------------------
        # 2. Map-Reduce 生成
        # ------------------------------------------------------------------
        t0 = time.time()
        if trace is not None:
            with self._langfuse.span(
                trace, "review.generate",
                input={"num_sections": len(plan), "num_docs": len(doc_infos)},
            ) as span:
                gen_result = self.generator.generate(
                    topic=plan.topic,
                    sections=plan.sections,
                    doc_chunks=doc_chunks,
                )
                span.output = {
                    "completed": gen_result.completed,
                    "paragraphs": len(gen_result.paragraphs),
                    "failed_sections": gen_result.failed_sections,
                }
                span.metadata["cache_hits"] = gen_result.cache_hits
                span.metadata["cache_misses"] = gen_result.cache_misses
        else:
            gen_result = self.generator.generate(
                topic=plan.topic,
                sections=plan.sections,
                doc_chunks=doc_chunks,
            )

        gen_latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "综述生成完成: paragraphs=%d, completed=%s, failed=%s, latency_ms=%d",
            len(gen_result.paragraphs), gen_result.completed,
            gen_result.failed_sections, gen_latency_ms,
        )

        # ------------------------------------------------------------------
        # 3. 引用事实验证（文档 4.3 节）
        # ------------------------------------------------------------------
        verified_results: List[Any] = []
        all_citations: List[Dict] = []
        citation_stats = {"entailment": 0, "contradiction": 0, "neutral": 0, "total": 0}
        full_html_parts: List[str] = []
        verified = False

        if verify_citations:
            t0 = time.time()
            # 3a. 逐段验证（段落级编号，index 每段从 1 开始）
            for para in gen_result.paragraphs:
                if trace is not None:
                    with self._langfuse.span(
                        trace, f"citation.verify.{para.section_id}",
                        input={"text_preview": para.text[:100]},
                    ) as span:
                        vresult = self._verify_one(para.text)
                        span.output = {"stats": vresult.stats}
                else:
                    vresult = self._verify_one(para.text)

                verified_results.append(vresult)

            # 3b. 文档级全局编号重排（否则多段 [1][2]... 冲突，参考文献丢条目）
            from .citation_verifier import renumber_results
            verified_results = renumber_results(verified_results)

            # 3c. 用全局编号后的结果聚合渲染/统计/引用
            for vresult in verified_results:
                full_html_parts.append(vresult.rendered_html)

                # 聚合统计
                for k in citation_stats:
                    citation_stats[k] += vresult.stats.get(k, 0)

                # 收集引用
                for cit in vresult.citations:
                    all_citations.append({
                        "index": cit.ref.index,
                        "doc_id": cit.ref.doc_id,
                        "chunk_id": cit.ref.chunk_id,
                        "label": cit.label,
                        "exists": cit.exists,
                        "original_text": cit.original_text,
                        "marker": cit.marker,
                    })

            verify_latency_ms = int((time.time() - t0) * 1000)
            verified = True
            logger.info(
                "引用验证完成: total=%d, entailment=%d, contradiction=%d, neutral=%d, latency_ms=%d",
                citation_stats["total"], citation_stats["entailment"],
                citation_stats["contradiction"], citation_stats["neutral"],
                verify_latency_ms,
            )

        # ------------------------------------------------------------------
        # 4. 组装 ReviewDocument
        # ------------------------------------------------------------------
        # 段落数据（用于 Word 导出）
        paragraphs_data: List[Dict] = []
        for para, vresult in zip(gen_result.paragraphs, verified_results):
            paragraphs_data.append({
                "section_id": para.section_id,
                "text": para.text,
                "rendered_text": vresult.rendered_text,
                "rendered_html": vresult.rendered_html,
                "citations": [
                    {
                        "index": c.ref.index,
                        "doc_id": c.ref.doc_id,
                        "chunk_id": c.ref.chunk_id,
                        "label": c.label,
                    }
                    for c in vresult.citations
                ],
                "from_cache": para.from_cache,
            })

        # 如果跳过了验证，paragraphs_data 只包含原始文本
        if not verify_citations:
            paragraphs_data = [
                {
                    "section_id": para.section_id,
                    "text": para.text,
                    "rendered_text": para.text,
                    "rendered_html": "",
                    "citations": [],
                    "from_cache": para.from_cache,
                }
                for para in gen_result.paragraphs
            ]

        # 大纲数据
        outline_data = [
            {
                "section_id": s.section_id,
                "title": s.title,
                "description": s.description,
                "key_points": list(s.key_points),
                "related_doc_ids": list(s.related_doc_ids),
            }
            for s in plan.sections
        ]

        # 参考文献（按引用编号排序，去重）
        seen_indices = set()
        references: List[Dict] = []
        for cit in sorted(all_citations, key=lambda x: x["index"]):
            if cit["index"] in seen_indices:
                continue
            seen_indices.add(cit["index"])
            references.append(cit)

        full_html = "\n\n".join(full_html_parts) if full_html_parts else ""

        document = ReviewDocument(
            topic=plan.topic,
            title=title or f"综述：{plan.topic}",
            outline=outline_data,
            paragraphs=paragraphs_data,
            verified=verified,
            full_text=gen_result.full_text,
            full_html=full_html,
            citation_stats=citation_stats,
            references=references,
            metadata={
                "num_sections": len(plan),
                "num_paragraphs": len(gen_result.paragraphs),
                "num_references": len(references),
                "completed": gen_result.completed,
                "failed_sections": gen_result.failed_sections,
                "cache_hits": gen_result.cache_hits,
                "cache_misses": gen_result.cache_misses,
                "retry_hint": gen_result.retry_hint,
            },
        )

        # 成本追踪（文档 6.1 节）
        pipeline_latency_ms = int((time.time() - t_pipeline_start) * 1000)
        if self._cost_tracker is not None:
            try:
                from ..observability.cost_tracker import CostRecord
                from datetime import datetime
                self._cost_tracker.record(CostRecord(
                    ts=datetime.now(),
                    model="pipeline",
                    step="review.pipeline",
                    input_tokens=0,
                    output_tokens=len(gen_result.full_text) // 4,  # 粗估
                    cost_yuan=0.0,  # 实际成本由子步骤分别记录
                    trace_id="",
                ))
            except Exception as e:  # noqa: BLE001
                logger.debug("成本追踪失败（不阻塞主流程）: %s", e)

        # 结束 trace
        if trace is not None and self._langfuse is not None:
            self._langfuse.end_trace(trace)

        logger.info(
            "综述主流程完成: total_latency_ms=%d, paragraphs=%d, references=%d",
            pipeline_latency_ms, len(paragraphs_data), len(references),
        )

        return document

    # ------------------------------------------------------------------
    # 内部：单段落引用验证
    # ------------------------------------------------------------------

    def _verify_one(self, paragraph_text: str) -> Any:
        """验证单段综述的引用

        如果未配置 citation_verifier，则跳过验证返回空结果。
        """
        from .citation_verifier import (
            CitationVerifier,
            VerificationResult,
            _escape_html,
        )

        if self.citation_verifier is None:
            # 跳过验证
            return VerificationResult(
                paragraph_text=paragraph_text,
                citations=[],
                rendered_text=paragraph_text,
                rendered_html=_escape_html(paragraph_text),
                stats={"entailment": 0, "contradiction": 0, "neutral": 0, "total": 0},
            )

        return self.citation_verifier.verify_paragraph(paragraph_text)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_review_pipeline(
    retriever=None,
    embedder=None,
    api_key: Optional[str] = None,
    mode: str = "auto",
    verify_citations: bool = True,
) -> ReviewPipeline:
    """工厂函数：创建综述主流程编排器

    Args:
        retriever: 检索器（用于 citation_verifier 二次检索；None 时跳过验证）
        embedder: 向量化器（用于 LLM generator 的 MMR 去重）
        api_key: DashScope API key（None 时从环境变量读取）
        mode: "auto" / "llm" / "mock"
        verify_citations: 是否启用引用验证（无 retriever 时强制 False）
    Returns:
        ReviewPipeline 实例
    """
    from .outline_planner import create_outline_planner
    from .generator import create_review_generator
    from .citation_verifier import CitationVerifier

    # mock 组件的构造函数不接受 api_key（无 LLM 调用），因此 mock 模式不传；
    # 其余模式透传 api_key 供 LLM 组件使用。此前统一透传会让 mode="mock" 直接
    # TypeError——这也是 /review 长期只能走 auto 的原因之一。
    if str(mode).lower() == "mock":
        planner = create_outline_planner(mode="mock")
        generator = create_review_generator(mode="mock", embedder=embedder)
    else:
        planner = create_outline_planner(mode=mode, api_key=api_key)
        generator = create_review_generator(
            mode=mode, api_key=api_key, embedder=embedder,
        )

    citation_verifier = None
    if verify_citations and retriever is not None:
        try:
            from ..rag.reranker import create_reranker
            # mock 模式强制规则化 NLI 判断，绝不因 .env 里的真实 key 触达 API
            reranker_mode = "noop" if str(mode).lower() == "mock" else "auto"
            reranker = create_reranker(reranker_mode, api_key=api_key)
        except ImportError:
            reranker = None
        citation_verifier = CitationVerifier(
            retriever=retriever, nli_reranker=reranker,
        )
    elif verify_citations and retriever is None:
        logger.info("未提供 retriever，跳过引用验证")

    return ReviewPipeline(
        outline_planner=planner,
        generator=generator,
        citation_verifier=citation_verifier,
    )

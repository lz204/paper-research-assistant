"""多源融合 + 冲突仲裁引擎（项目文档 4.1 节核心）

文档 4.1 节多源融合与冲突处理规则：
真实冲突场景：Qwen-VL-Max 说某区域是公式 E=mc^2，pdfplumber 在同一区域抽到的是乱码 Unicode
——听谁的？按"职责单一"原则分工：

| 冲突维度               | 仲裁者                     | 理由                              |
|------------------------|----------------------------|-----------------------------------|
| 区域**类型**（公式/正文/图/表） | Qwen-VL-Max          | VLM 有视觉理解，pdfplumber 不识别版面 |
| 区域内**文本内容**     | pdfplumber（正文）/ Qwen-VL（公式） | 正文 pdfplumber 最准，公式 VLM 最准 |
| 公式区域但 VLM 置信度低 | Qwen-VL 二次验证          | 裁剪区域单独送 VLM                |
| 表格结构还原           | Qwen-VL-Max 输出 Markdown  | 置信度 < 0.8 切二次验证           |

pdfplumber 定位：作为"文本层兜底"，不参与冲突仲裁，只作为降级通道。

最终输出：每个元素带 confidence 字段，低于 0.7 标注 need_review=true，
前端黄色高亮提示用户人工核对。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .formula_binder import BoundFormula, FormulaBinder, binding_stats
from .fallback_handler import FallbackHandler
from .text_extractor import PageLayout, Paragraph
from .vlm_client import VlmFormula, VlmPageResult, VlmTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 融合后的数据结构
# ---------------------------------------------------------------------------


@dataclass
class FusedParagraph:
    """融合后的段落

    Attributes:
        idx: 段落序号（页内，0-based）
        text: 段落文本（pdfplumber 正文 + VLM 公式插入）
        source: 文本来源 "pdfplumber" / "vlm" / "fused"
        confidence: 置信度
        bound_formulas: 绑定到该段落的公式列表
    """

    idx: int
    text: str
    source: str = "fused"
    confidence: float = 1.0
    bound_formulas: List[BoundFormula] = field(default_factory=list)


@dataclass
class FusedTable:
    """融合后的表格

    Attributes:
        markdown: Markdown 表格
        confidence: 置信度
        need_review: 是否需人工复核
        location_desc: 位置描述
    """

    markdown: str
    confidence: float
    need_review: bool
    location_desc: str = ""


@dataclass
class FusedPage:
    """融合后的单页

    Attributes:
        page_no: 页码
        paragraphs: 段落列表（含绑定的公式）
        tables: 表格列表
        unbound_formulas: 未绑定到段落的公式（rule4 兜底）
        confidence: 整页置信度
        formula_resolved: 公式是否全部解析
        binding_stats: 公式绑定统计
    """

    page_no: int
    paragraphs: List[FusedParagraph] = field(default_factory=list)
    tables: List[FusedTable] = field(default_factory=list)
    unbound_formulas: List[BoundFormula] = field(default_factory=list)
    confidence: float = 0.0
    formula_resolved: bool = True
    binding_stats: dict = field(default_factory=dict)


@dataclass
class FusedDocument:
    """融合后的完整文档

    Attributes:
        doc_id: 文档 ID
        pages: 页面列表
        total_formulas: 公式总数
        total_tables: 表格总数
        avg_confidence: 平均置信度
    """

    doc_id: str
    pages: List[FusedPage] = field(default_factory=list)
    total_formulas: int = 0
    total_tables: int = 0
    avg_confidence: float = 0.0


# ---------------------------------------------------------------------------
# 融合引擎
# ---------------------------------------------------------------------------


class FusionEngine:
    """多源融合 + 冲突仲裁引擎

    工作流程（文档 4.1 节）：
    1. VLM 识别 → 得到正文 + 公式 + 表格（含位置描述）
    2. pdfplumber 抽取 → 得到文本块 + 坐标 + 段落
    3. 冲突仲裁：
       - 区域类型由 VLM 仲裁
       - 正文文本取 pdfplumber（更准）
       - 公式文本取 VLM（pdfplumber 是乱码）
    4. 公式绑定：用 FormulaBinder 四级规则把公式绑定到段落
    5. 置信度兜底：低置信度触发 FallbackHandler 二次验证
    6. 输出：FusedDocument，每个元素带 confidence + need_review

    用法：
        engine = FusionEngine(vlm_client)
        doc = engine.fuse(doc_id="paper_1", vlm_results=..., page_layouts=...)
    """

    # 文档 4.1 节：低于此阈值标注 need_review
    REVIEW_THRESHOLD = 0.7

    def __init__(
        self,
        vlm_client=None,
        formula_binder: Optional[FormulaBinder] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        """
        Args:
            vlm_client: VLM 客户端（用于二次验证，None 时跳过）
            formula_binder: 公式绑定器（None 时用默认）
            fallback_handler: 降级处理器（None 时用默认）
        """
        self.vlm_client = vlm_client
        self.formula_binder = formula_binder or FormulaBinder()
        self.fallback_handler = fallback_handler
        if vlm_client is not None and fallback_handler is None:
            self.fallback_handler = FallbackHandler(vlm_client)

    def fuse(
        self,
        doc_id: str,
        vlm_results: List[VlmPageResult],
        page_layouts: List[PageLayout],
        page_images=None,
    ) -> FusedDocument:
        """融合 VLM + pdfplumber 结果

        Args:
            doc_id: 文档 ID
            vlm_results: VLM 识别结果（每页一个）
            page_layouts: pdfplumber 抽取的版面（每页一个）
            page_images: 每页的 PageImage 列表（用于低置信度公式/表格二次验证裁剪，None 时跳过）
        Returns:
            FusedDocument
        """
        if not vlm_results and not page_layouts:
            return FusedDocument(doc_id=doc_id)

        # 按 page_no 对齐两路结果
        vlm_map = {r.page_no: r for r in vlm_results}
        layout_map = {l.page_no: l for l in page_layouts}
        image_map = {img.page_no: img for img in (page_images or [])}
        all_page_nos = sorted(set(vlm_map.keys()) | set(layout_map.keys()))

        fused_pages: List[FusedPage] = []
        total_formulas = 0
        total_tables = 0
        confidence_sum = 0.0

        for page_no in all_page_nos:
            vlm_result = vlm_map.get(page_no)
            layout = layout_map.get(page_no)
            page_image = image_map.get(page_no)

            fused_page = self._fuse_page(doc_id, page_no, vlm_result, layout, page_image)
            fused_pages.append(fused_page)
            total_formulas += sum(len(p.bound_formulas) for p in fused_page.paragraphs)
            total_formulas += len(fused_page.unbound_formulas)
            total_tables += len(fused_page.tables)
            confidence_sum += fused_page.confidence

        avg_conf = confidence_sum / len(fused_pages) if fused_pages else 0.0

        return FusedDocument(
            doc_id=doc_id,
            pages=fused_pages,
            total_formulas=total_formulas,
            total_tables=total_tables,
            avg_confidence=round(avg_conf, 3),
        )

    # ------------------------------------------------------------------
    # 单页融合
    # ------------------------------------------------------------------

    def _fuse_page(
        self,
        doc_id: str,
        page_no: int,
        vlm_result: Optional[VlmPageResult],
        layout: Optional[PageLayout],
        page_image=None,
    ) -> FusedPage:
        """融合单页

        三种场景：
        1. VLM + pdfplumber 都有 → 完整融合
        2. 只有 VLM（pdfplumber 失败）→ 用 VLM 文本
        3. 只有 pdfplumber（VLM 降级）→ 标注 formula_resolved=false
        """
        # 场景 3：VLM 降级到 pdfplumber
        if vlm_result is None and layout is not None:
            return self._fuse_pdfplumber_only(page_no, layout)

        # 场景 2：只有 VLM
        if vlm_result is not None and layout is None:
            return self._fuse_vlm_only(page_no, vlm_result)

        # 场景 1：完整融合
        if vlm_result is not None and layout is not None:
            return self._fuse_both(page_no, vlm_result, layout, page_image)

        # 两者都没有（不应发生）
        return FusedPage(page_no=page_no, confidence=0.0, formula_resolved=False)

    def _fuse_both(
        self,
        page_no: int,
        vlm_result: VlmPageResult,
        layout: PageLayout,
        page_image=None,
    ) -> FusedPage:
        """完整融合：VLM + pdfplumber 都有

        文档 4.1 节冲突仲裁：
        - 区域类型由 VLM 仲裁
        - 正文文本取 pdfplumber（更准）
        - 公式文本取 VLM（pdfplumber 是乱码）
        """
        # 1. 正文段落：以 pdfplumber 为主（文档：正文 pdfplumber 最准）
        paragraphs = [
            FusedParagraph(
                idx=p.idx,
                text=p.text,
                source="pdfplumber",
                confidence=0.95,  # pdfplumber 正文置信度高
            )
            for p in layout.paragraphs
        ]

        # 2. 公式绑定（四级规则）
        all_bound_formulas: List[BoundFormula] = []
        if vlm_result.formulas:
            # 可选：低置信度公式先二次验证
            formulas_to_bind = vlm_result.formulas
            if self.fallback_handler is not None:
                formulas_to_bind = self.fallback_handler.reverify_formulas(
                    vlm_result.formulas, page_image=page_image, page_layout=layout,
                )

            # 四级规则绑定
            all_bound_formulas = self.formula_binder.bind(formulas_to_bind, layout)

        # 3. 把绑定到段落的公式分配到对应段落
        unbound: List[BoundFormula] = []
        for bound in all_bound_formulas:
            if bound.bound_paragraph_idx is not None and 0 <= bound.bound_paragraph_idx < len(paragraphs):
                paragraphs[bound.bound_paragraph_idx].bound_formulas.append(bound)
                # 段落置信度取公式绑定置信度的最小值
                if bound.confidence < paragraphs[bound.bound_paragraph_idx].confidence:
                    paragraphs[bound.bound_paragraph_idx].confidence = bound.confidence
            else:
                unbound.append(bound)

        # 4. 表格（文档：Qwen-VL-Max 输出 Markdown，置信度 < 0.8 切二次验证）
        tables = self._fuse_tables(vlm_result.tables, page_image=page_image)

        # 5. 统计
        stats = binding_stats(all_bound_formulas)

        # 6. 整页置信度
        page_conf = vlm_result.confidence
        if all_bound_formulas:
            avg_bound_conf = sum(f.confidence for f in all_bound_formulas) / len(all_bound_formulas)
            page_conf = (vlm_result.confidence + avg_bound_conf) / 2

        return FusedPage(
            page_no=page_no,
            paragraphs=paragraphs,
            tables=tables,
            unbound_formulas=unbound,
            confidence=round(page_conf, 3),
            formula_resolved=vlm_result.formula_resolved,
            binding_stats=stats,
        )

    def _fuse_vlm_only(
        self,
        page_no: int,
        vlm_result: VlmPageResult,
    ) -> FusedPage:
        """只有 VLM 结果（pdfplumber 失败）

        文档 4.1 节：VLM 识别的正文直接用，但无法做公式绑定（无段落坐标）
        """
        # VLM 文本按换行切分为段落
        vlm_paras = [p.strip() for p in vlm_result.text.split("\n\n") if p.strip()]
        paragraphs = [
            FusedParagraph(
                idx=i,
                text=text,
                source="vlm",
                confidence=vlm_result.confidence,
            )
            for i, text in enumerate(vlm_paras)
        ]

        # 公式无法绑定（无 pdfplumber 段落），全部归为 unbound
        unbound = [
            BoundFormula(
                latex=f.latex,
                formula_id=f.formula_id,
                bound_paragraph_idx=None,
                confidence=f.confidence,
                binding_rule="rule4_unbound",
                need_review=f.confidence < self.REVIEW_THRESHOLD,
                location_desc=f.location_desc,
            )
            for f in vlm_result.formulas
        ]

        tables = self._fuse_tables(vlm_result.tables)
        stats = binding_stats(unbound)

        return FusedPage(
            page_no=page_no,
            paragraphs=paragraphs,
            tables=tables,
            unbound_formulas=unbound,
            confidence=vlm_result.confidence,
            formula_resolved=vlm_result.formula_resolved,
            binding_stats=stats,
        )

    def _fuse_pdfplumber_only(
        self,
        page_no: int,
        layout: PageLayout,
    ) -> FusedPage:
        """VLM 降级，只有 pdfplumber（文档 4.1 节兜底通道）

        文档：formula_resolved=false，前端提示用户公式未解析
        """
        paragraphs = [
            FusedParagraph(
                idx=p.idx,
                text=p.text,
                source="pdfplumber",
                confidence=0.5,  # 降级模式置信度低
            )
            for p in layout.paragraphs
        ]
        return FusedPage(
            page_no=page_no,
            paragraphs=paragraphs,
            tables=[],
            unbound_formulas=[],
            confidence=0.5,
            formula_resolved=False,  # 关键标记
            binding_stats={"total": 0, "degraded": True},
        )

    # ------------------------------------------------------------------
    # 表格融合
    # ------------------------------------------------------------------

    def _fuse_tables(self, vlm_tables: List[VlmTable], page_image=None) -> List[FusedTable]:
        """融合表格（文档：Qwen-VL-Max 输出 Markdown，置信度 < 0.8 切二次验证）"""
        result: List[FusedTable] = []
        for t in vlm_tables:
            need_review = t.confidence < 0.8
            # 低置信度：触发二次验证
            if need_review and self.fallback_handler is not None:
                reverify = self.fallback_handler.reverify_table(t, page_image=page_image)
                if reverify.success and reverify.table is not None:
                    t = reverify.table
                    need_review = t.confidence < 0.8
            result.append(
                FusedTable(
                    markdown=t.markdown,
                    confidence=t.confidence,
                    need_review=need_review,
                    location_desc=t.location_desc,
                )
            )
        return result


# ---------------------------------------------------------------------------
# 文档级统计（对接文档 6.1 可观测性）
# ---------------------------------------------------------------------------


def document_stats(doc: FusedDocument) -> dict:
    """生成文档级统计（用于 Langfuse 看板）"""
    total_paras = sum(len(p.paragraphs) for p in doc.pages)
    total_bound = sum(
        len(p.bound_formulas) for page in doc.pages for p in page.paragraphs
    )
    total_unbound = sum(len(p.unbound_formulas) for p in doc.pages)
    total_review = sum(
        1
        for page in doc.pages
        for p in page.paragraphs
        for f in p.bound_formulas
        if f.need_review
    ) + sum(
        1
        for page in doc.pages
        for f in page.unbound_formulas
        if f.need_review
    )

    return {
        "doc_id": doc.doc_id,
        "pages": len(doc.pages),
        "paragraphs": total_paras,
        "formulas": doc.total_formulas,
        "formulas_bound": total_bound,
        "formulas_unbound": total_unbound,
        "tables": doc.total_tables,
        "need_review": total_review,
        "avg_confidence": doc.avg_confidence,
        "formula_resolved_pages": sum(1 for p in doc.pages if p.formula_resolved),
    }

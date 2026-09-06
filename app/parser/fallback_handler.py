"""降级与二次验证处理器（项目文档 4.1 节 + 6.3 章）

文档 4.1 节多源融合冲突处理规则：
- 公式区域但 VLM 置信度低 → Qwen-VL 二次验证
    "裁剪该区域图片单独送 VLM，prompt 强化'仅识别公式 LaTeX'"
- 表格结构还原置信度 < 0.8 → 切二次验证

文档 6.3 节容错降级链：
- VLM 超时/限流 → 指数退避重试 2 次
- 仍失败 → pdfplumber 纯文本降级
- 标注 formula_resolved=false

文档 4.1 节 pdfplumber 定位：
- 作为"文本层兜底"，不参与冲突仲裁，只作为降级通道
- VLM API 全挂时仍能返回可用纯文本
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .text_extractor import PageLayout
from .vlm_client import QwenVlmClient, VlmFormula, VlmPageResult, VlmTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 二次验证 prompt（文档 4.1 节：强化"仅识别公式 LaTeX"）
# ---------------------------------------------------------------------------

FORMULA_REVERIFY_PROMPT = """\
请仅识别这张图片中的公式，返回 JSON：
{
  "latex": "公式 LaTeX（不含其他文字）",
  "confidence": 0.0-1.0
}
只返回 JSON。若无公式，返回 {"latex": "", "confidence": 0.0}。"""

TABLE_REVERIFY_PROMPT = """\
请识别这张图片中的表格，返回 JSON：
{
  "markdown": "Markdown 格式表格",
  "confidence": 0.0-1.0
}
只返回 JSON。若无表格，返回 {"markdown": "", "confidence": 0.0}。"""


# 文档 4.1 节：置信度阈值
FORMULA_CONFIDENCE_THRESHOLD = 0.7  # 低于此值触发二次验证
TABLE_CONFIDENCE_THRESHOLD = 0.8    # 表格阈值更严


# ---------------------------------------------------------------------------
# 降级结果
# ---------------------------------------------------------------------------


@dataclass
class FallbackResult:
    """降级处理结果

    Attributes:
        success: 是否成功（True=二次验证成功，False=最终降级到 pdfplumber）
        formula: 二次验证后的公式（公式二次验证成功时）
        table: 二次验证后的表格（表格二次验证成功时）
        reason: 降级原因
        formula_resolved: 公式是否成功解析
    """

    success: bool
    formula: Optional[VlmFormula] = None
    table: Optional[VlmTable] = None
    reason: str = ""
    formula_resolved: bool = True


# ---------------------------------------------------------------------------
# 降级处理器
# ---------------------------------------------------------------------------


class FallbackHandler:
    """置信度不足时的二次验证 + VLM 不可用时的 pdfplumber 降级

    文档 4.1 节 + 6.3 节容错链：
    1. 公式 confidence < 0.7 → 裁剪区域送 VLM 二次验证
    2. VLM 完全不可用 → pdfplumber 纯文本降级（formula_resolved=false）

    用法：
        handler = FallbackHandler(vlm_client)
        result = handler.reverify_formula(formula, page_image, region)
        if not result.success:
            # 最终降级
            degraded = handler.fallback_to_pdfplumber(page_layout)
    """

    def __init__(self, vlm_client: QwenVlmClient):
        self.vlm_client = vlm_client

    # ------------------------------------------------------------------
    # 公式二次验证
    # ------------------------------------------------------------------

    def reverify_formula(
        self,
        formula: VlmFormula,
        page_image=None,
        region_coords: Optional[tuple] = None,
    ) -> FallbackResult:
        """对低置信度公式做二次验证

        文档 4.1 节：裁剪该区域图片单独送 VLM，prompt 强化"仅识别公式 LaTeX"

        Args:
            formula: 待验证的公式
            page_image: 整页图片（PageImage），用于裁剪
            region_coords: (x0, y0, x1, y1, page_width, page_height) 裁剪区域
        Returns:
            FallbackResult
        """
        # mock 模式：直接提升置信度（本地无 API）
        if self.vlm_client._mock_mode:
            logger.info(
                "mock 模式二次验证 formula=%s: 置信度 %.2f → 0.85",
                formula.formula_id, formula.confidence,
            )
            reverified = VlmFormula(
                latex=formula.latex,
                location_desc=formula.location_desc,
                confidence=0.85,
                formula_id=formula.formula_id,
            )
            return FallbackResult(success=True, formula=reverified, formula_resolved=True)

        # 真实模式：需要 page_image 裁剪区域
        if page_image is None:
            logger.warning("二次验证需要 page_image，但未提供，降级为 pdfplumber")
            return FallbackResult(
                success=False,
                reason="no_page_image",
                formula_resolved=False,
            )

        # 裁剪区域
        cropped_b64 = ""
        if region_coords and len(region_coords) >= 6:
            from .pdf_to_image import crop_region, PageImage
            x0, y0, x1, y1, pw, ph = region_coords
            cropped = crop_region(page_image, x0, y0, x1, y1, pw, ph)
            if cropped is not None:
                cropped_b64 = cropped.image_b64

        if not cropped_b64:
            # 无法裁剪：用整页图
            cropped_b64 = page_image.image_b64

        # 调用 VLM 二次验证
        try:
            result = self.vlm_client.recognize_page_sync(
                cropped_b64,
                page_no=page_image.page_no,
                prompt=FORMULA_REVERIFY_PROMPT,
            )
            # 解析二次验证结果
            if result.formulas:
                reverified = result.formulas[0]
                return FallbackResult(
                    success=True,
                    formula=VlmFormula(
                        latex=reverified.latex or formula.latex,
                        location_desc=formula.location_desc,
                        confidence=max(reverified.confidence, formula.confidence),
                        formula_id=formula.formula_id,
                    ),
                    formula_resolved=True,
                )
            # 二次验证未返回公式：保留原公式但标记低置信度
            return FallbackResult(
                success=False,
                formula=formula,
                reason="reverify_no_formula",
                formula_resolved=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("二次验证异常 formula=%s: %s", formula.formula_id, e)
            return FallbackResult(
                success=False,
                reason=f"reverify_exception: {e}",
                formula_resolved=False,
            )

    # ------------------------------------------------------------------
    # 表格二次验证
    # ------------------------------------------------------------------

    def reverify_table(
        self,
        table: VlmTable,
        page_image=None,
        region_coords: Optional[tuple] = None,
    ) -> FallbackResult:
        """对低置信度表格做二次验证（文档 4.1 节：置信度 < 0.8 切二次验证）"""
        # mock 模式：直接提升置信度（本地无 API）
        if self.vlm_client._mock_mode:
            logger.info(
                "mock 模式二次验证 table: 置信度 %.2f → 0.85",
                table.confidence,
            )
            reverified = VlmTable(
                markdown=table.markdown,
                location_desc=table.location_desc,
                confidence=0.85,
            )
            return FallbackResult(success=True, table=reverified, formula_resolved=True)

        # 真实模式：需要 page_image 裁剪区域
        if page_image is None:
            logger.warning("表格二次验证需要 page_image，但未提供，降级保留原表格")
            return FallbackResult(
                success=False,
                table=table,
                reason="no_page_image",
                formula_resolved=False,
            )

        # 裁剪区域
        cropped_b64 = ""
        if region_coords and len(region_coords) >= 6:
            from .pdf_to_image import crop_region
            x0, y0, x1, y1, pw, ph = region_coords
            cropped = crop_region(page_image, x0, y0, x1, y1, pw, ph)
            if cropped is not None:
                cropped_b64 = cropped.image_b64

        if not cropped_b64:
            # 无法裁剪：用整页图
            cropped_b64 = page_image.image_b64

        # 调用 VLM 二次验证
        try:
            result = self.vlm_client.recognize_page_sync(
                cropped_b64,
                page_no=page_image.page_no,
                prompt=TABLE_REVERIFY_PROMPT,
            )
            # 解析二次验证结果
            if result.tables:
                reverified = result.tables[0]
                return FallbackResult(
                    success=True,
                    table=VlmTable(
                        markdown=reverified.markdown or table.markdown,
                        location_desc=table.location_desc,
                        confidence=max(reverified.confidence, table.confidence),
                    ),
                    formula_resolved=True,
                )
            # 二次验证未返回表格：保留原表格但标记低置信度
            return FallbackResult(
                success=False,
                table=table,
                reason="reverify_no_table",
                formula_resolved=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("表格二次验证异常: %s", e)
            return FallbackResult(
                success=False,
                table=table,
                reason=f"reverify_exception: {e}",
                formula_resolved=False,
            )

    # ------------------------------------------------------------------
    # pdfplumber 纯文本降级
    # ------------------------------------------------------------------

    def fallback_to_pdfplumber(
        self,
        page_layout: PageLayout,
        page_no: int,
    ) -> VlmPageResult:
        """VLM 完全不可用时回退到 pdfplumber 纯文本

        文档 4.1 节：
        - 回退到 pdfplumber 抽取纯文本（不含公式）
        - 结果标注 formula_resolved=false
        - pdfplumber 不参与冲突仲裁，只作为降级通道

        文档 6.3 节：
        - 即使云端 API 全挂，系统仍能返回可用的纯文本，不直接报错
        """
        # 拼接所有段落文本
        text = "\n\n".join(p.text for p in page_layout.paragraphs)
        logger.info(
            "pdfplumber 降级 page=%d: %d 段落, %d 字符, formula_resolved=false",
            page_no, len(page_layout.paragraphs), len(text),
        )
        return VlmPageResult(
            page_no=page_no,
            text=text,
            formulas=[],  # pdfplumber 无法识别公式
            tables=[],
            confidence=0.5,
            from_cache=False,
            formula_resolved=False,  # 关键标记：公式未解析
        )

    # ------------------------------------------------------------------
    # 判断是否需要二次验证
    # ------------------------------------------------------------------

    @staticmethod
    def needs_formula_reverify(formula: VlmFormula) -> bool:
        """文档 4.1 节：公式 confidence < 0.7 触发二次验证"""
        return formula.confidence < FORMULA_CONFIDENCE_THRESHOLD

    @staticmethod
    def needs_table_reverify(table: VlmTable) -> bool:
        """文档 4.1 节：表格 confidence < 0.8 触发二次验证"""
        return table.confidence < TABLE_CONFIDENCE_THRESHOLD

    # ------------------------------------------------------------------
    # 批量二次验证
    # ------------------------------------------------------------------

    def reverify_formulas(
        self,
        formulas: List[VlmFormula],
        page_image=None,
        page_layout: Optional[PageLayout] = None,
    ) -> List[VlmFormula]:
        """批量二次验证低置信度公式

        Args:
            formulas: 原始公式列表
            page_image: 整页图片
            page_layout: 页面版面（用于构造降级结果）
        Returns:
            验证后的公式列表（顺序与输入一致）
        """
        if not formulas:
            return []

        results: List[VlmFormula] = []
        for formula in formulas:
            if self.needs_formula_reverify(formula):
                # 触发二次验证
                reverify = self.reverify_formula(formula, page_image)
                if reverify.success and reverify.formula is not None:
                    results.append(reverify.formula)
                else:
                    # 二次验证失败：保留原公式但标记低置信度
                    results.append(formula)
                    logger.warning(
                        "公式 %s 二次验证失败: %s",
                        formula.formula_id, reverify.reason,
                    )
            else:
                results.append(formula)

        return results

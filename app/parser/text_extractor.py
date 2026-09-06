"""pdfplumber 文本+坐标抽取模块（项目文档 4.1 节）

文档 4.1 节定位：作为"文本层兜底"
- VLM API 超时或限流时，回退到 pdfplumber 抽取纯文本（不含公式）
- 结果标注 formula_resolved: false
- 不参与冲突仲裁，只作为降级通道

但在公式绑定时，pdfplumber 的文本块坐标是关键输入：
- 规则二（位置描述映射）：用页面中轴线把"左/中/右"映射到坐标区域
- 规则三（文本锚点匹配）：在 pdfplumber 文本中搜索关键词定位段落
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """pdfplumber 抽取的文本块

    Attributes:
        text: 文本内容
        x0, y0, x1, y1: 坐标边界（PDF 坐标系，原点左下）
        page_no: 页码（0-based）
        block_idx: 页内块序号
        is_paragraph_start: 是否为段落起始块
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_no: int
    block_idx: int
    is_paragraph_start: bool = False


@dataclass
class PageLayout:
    """单页版面布局

    Attributes:
        page_no: 页码
        width: 页面宽度（PDF 单位）
        height: 页面高度（PDF 单位）
        blocks: 文本块列表
        midline_x: 中轴线 x 坐标（用于规则二左/中/右映射）
        paragraphs: 按块聚合的段落列表（每段含 idx + text + block_indices）
    """

    page_no: int
    width: float
    height: float
    blocks: List[TextBlock] = field(default_factory=list)
    midline_x: float = 0.0
    paragraphs: List["Paragraph"] = field(default_factory=list)


@dataclass
class Paragraph:
    """聚合后的段落

    Attributes:
        idx: 段落序号（页内，0-based）
        text: 段落文本
        block_indices: 构成该段落的文本块序号列表
        x0, y0, x1, y1: 段落边界（取所有块的最值）
        region: 所属区域 "left" / "center" / "right"（基于中轴线）
    """

    idx: int
    text: str
    block_indices: List[int]
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    region: str = "center"


# ---------------------------------------------------------------------------
# pdfplumber 抽取器
# ---------------------------------------------------------------------------


# 段落分隔：缩进或空行
PARAGRAPH_INDENT = re.compile(r"^\s{2,}\S")
# 句子分隔
SENTENCE_END = re.compile(r"[.!?。！？]\s*$")


class PdfplumberExtractor:
    """pdfplumber 文本+坐标抽取器

    用法：
        extractor = PdfplumberExtractor()
        layouts = extractor.extract("paper.pdf")
        for layout in layouts:
            print(f"Page {layout.page_no}: {len(layout.paragraphs)} paragraphs")
    """

    def __init__(self, merge_tolerance: float = 5.0):
        """
        Args:
            merge_tolerance: 文本块合并容差（PDF 单位），
                           y 间距小于此值的相邻块合并为同一段落
        """
        self.merge_tolerance = merge_tolerance
        self._pdfplumber = None

    def _ensure_lib(self) -> bool:
        if self._pdfplumber is not None:
            return True
        try:
            import pdfplumber  # type: ignore
            self._pdfplumber = pdfplumber
            return True
        except ImportError:
            logger.warning(
                "pdfplumber 未安装，文本抽取不可用。pip install pdfplumber 后可启用"
            )
            return False

    def extract(
        self,
        pdf_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[PageLayout]:
        """抽取全文文本 + 坐标 + 段落聚合

        Args:
            pdf_path: PDF 文件路径
            page_range: (start, end) 页码区间，None 表示全部
        Returns:
            PageLayout 列表
        """
        if not self._ensure_lib():
            return []

        layouts: List[PageLayout] = []
        with self._pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            if page_range is None:
                start, end = 0, total - 1
            else:
                start, end = page_range
                start = max(0, start)
                end = min(total - 1, end)

            for page_no in range(start, end + 1):
                page = pdf.pages[page_no]
                layout = self._extract_page(page, page_no)
                layouts.append(layout)

        logger.info("pdfplumber 抽取完成: %s, %d 页", pdf_path, len(layouts))
        return layouts

    def _extract_page(self, page, page_no: int) -> PageLayout:
        """抽取单页"""
        width = float(page.width)
        height = float(page.height)
        midline_x = width / 2.0

        # 抽取文本块（含坐标）
        raw_blocks = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["size"],
        ) or []

        # 按行聚合 words → blocks
        blocks = self._words_to_blocks(raw_blocks, page_no, width, height)

        # 段落聚合
        paragraphs = self._blocks_to_paragraphs(blocks, midline_x)

        return PageLayout(
            page_no=page_no,
            width=width,
            height=height,
            blocks=blocks,
            midline_x=midline_x,
            paragraphs=paragraphs,
        )

    def _words_to_blocks(
        self,
        words: List[dict],
        page_no: int,
        page_width: float,
        page_height: float,
    ) -> List[TextBlock]:
        """把 words 聚合成文本块（按行）"""
        if not words:
            return []

        blocks: List[TextBlock] = []
        current_line: List[dict] = []
        current_y: Optional[float] = None

        for w in words:
            w_top = float(w.get("top", 0))
            if current_y is None or abs(w_top - current_y) <= self.merge_tolerance:
                current_line.append(w)
                current_y = w_top if current_y is None else current_y
            else:
                if current_line:
                    blocks.append(self._line_to_block(current_line, page_no, len(blocks)))
                current_line = [w]
                current_y = w_top

        if current_line:
            blocks.append(self._line_to_block(current_line, page_no, len(blocks)))

        return blocks

    @staticmethod
    def _line_to_block(line: List[dict], page_no: int, block_idx: int) -> TextBlock:
        """一行 words → TextBlock"""
        text = " ".join(w["text"] for w in line)
        x0 = min(float(w["x0"]) for w in line)
        y0 = min(float(w["top"]) for w in line)
        x1 = max(float(w["x1"]) for w in line)
        y1 = max(float(w["bottom"]) for w in line)
        return TextBlock(
            text=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            page_no=page_no,
            block_idx=block_idx,
        )

    def _blocks_to_paragraphs(
        self,
        blocks: List[TextBlock],
        midline_x: float,
    ) -> List[Paragraph]:
        """文本块聚合为段落（按 y 间距 + 缩进判断）"""
        if not blocks:
            return []

        paragraphs: List[Paragraph] = []
        current_blocks: List[TextBlock] = []
        current_indices: List[int] = []

        for block in blocks:
            if not current_blocks:
                current_blocks.append(block)
                current_indices.append(block.block_idx)
                continue

            prev = current_blocks[-1]
            y_gap = block.y0 - prev.y1

            # 新段落判断：y 间距大 或 段首缩进
            is_new_para = (
                y_gap > self.merge_tolerance * 2
                or (PARAGRAPH_INDENT.match(block.text) and y_gap > 0)
            )

            if is_new_para:
                paragraphs.append(self._make_paragraph(current_blocks, current_indices, midline_x, len(paragraphs)))
                current_blocks = [block]
                current_indices = [block.block_idx]
            else:
                current_blocks.append(block)
                current_indices.append(block.block_idx)

        if current_blocks:
            paragraphs.append(self._make_paragraph(current_blocks, current_indices, midline_x, len(paragraphs)))

        return paragraphs

    @staticmethod
    def _make_paragraph(
        blocks: List[TextBlock],
        indices: List[int],
        midline_x: float,
        idx: int,
    ) -> Paragraph:
        """文本块列表 → Paragraph"""
        text = " ".join(b.text for b in blocks)
        x0 = min(b.x0 for b in blocks)
        y0 = min(b.y0 for b in blocks)
        x1 = max(b.x1 for b in blocks)
        y1 = max(b.y1 for b in blocks)

        # 判断区域：基于段落中心的 x 坐标
        center_x = (x0 + x1) / 2.0
        if center_x < midline_x * 0.6:
            region = "left"
        elif center_x > midline_x * 1.4:
            region = "right"
        else:
            region = "center"

        return Paragraph(
            idx=idx,
            text=text,
            block_indices=indices,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            region=region,
        )


# ---------------------------------------------------------------------------
# 工具函数：构造 mock PageLayout（本地无 PDF 时用于验证公式绑定逻辑）
# ---------------------------------------------------------------------------


def build_mock_layout(
    page_no: int,
    paragraphs: List[str],
    width: float = 612.0,
    height: float = 792.0,
) -> PageLayout:
    """构造 mock PageLayout（本地验证用）

    Args:
        page_no: 页码
        paragraphs: 段落文本列表
        width: 页面宽度（PDF 默认 612 = 8.5inch * 72dpi）
        height: 页面高度（PDF 默认 792 = 11inch * 72dpi）
    Returns:
        PageLayout（每个段落对应一个 TextBlock + Paragraph）
    """
    midline_x = width / 2.0
    blocks: List[TextBlock] = []
    para_objs: List[Paragraph] = []

    # 每段占 80 单位高度，从顶部 50 开始
    y_cursor = 50.0
    for idx, text in enumerate(paragraphs):
        block = TextBlock(
            text=text,
            x0=72.0,
            y0=y_cursor,
            x1=width - 72.0,
            y1=y_cursor + 40.0,
            page_no=page_no,
            block_idx=idx,
            is_paragraph_start=True,
        )
        blocks.append(block)
        para_objs.append(
            Paragraph(
                idx=idx,
                text=text,
                block_indices=[idx],
                x0=block.x0,
                y0=block.y0,
                x1=block.x1,
                y1=block.y1,
                region="center",
            )
        )
        y_cursor += 80.0

    return PageLayout(
        page_no=page_no,
        width=width,
        height=height,
        blocks=blocks,
        midline_x=midline_x,
        paragraphs=para_objs,
    )

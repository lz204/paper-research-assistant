"""公式与段落绑定模块（项目文档 4.1 节核心修复）

⚠️ 坐标对齐漏洞修复说明 ⚠️
==============================
文档附录原注释为"公式与段落坐标 IoU 对齐绑定"，但文档 4.1 节明确指出：

    "Qwen-VL-Max 不返回精确坐标框，只返回自然语言位置描述
     （如'第 3 段后''公式(1)位于第二栏中部'）。
     所以我不能做真正的 IoU 对齐（没有可计算的坐标框），
     而是用'规则匹配为主、坐标辅助为辅'的四级绑定策略。"

本模块严格按文档 4.1 节修复后的方案实现四级绑定策略，
不再使用 IoU 对齐（因为 VLM 不返回坐标框，无法计算 IoU）。

四级绑定策略（文档 4.1 节）：
- 规则一 段落序号匹配（优先，置信度 0.9）：
    VLM 输出"第 3 段后"时，直接按序号绑定到 pdfplumber 对应段落
    覆盖约 70% 用例，准确率最高
- 规则二 位置描述映射（次优先，置信度 0.7）：
    VLM 输出"第二栏中部"时，用 pdfplumber 中轴线映射左/中/右区域
- 规则三 文本锚点匹配（兜底，置信度 0.5）：
    VLM 输出"softmax 计算公式如下"时，在 pdfplumber 文本中搜索锚点段落
- 规则四 兜底未绑定（置信度 0.3）：
    三条规则都未命中，标记 need_review=True，前端黄色高亮

置信度标注：低于 0.7 的绑定前端黄色高亮提示"公式位置可能不准确"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .text_extractor import PageLayout, Paragraph
from .vlm_client import VlmFormula

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class BoundFormula:
    """绑定后的公式

    Attributes:
        latex: LaTeX 文本
        formula_id: 公式编号
        bound_paragraph_idx: 绑定到的段落序号（None 表示未绑定）
        confidence: 绑定置信度 [0, 1]
        binding_rule: 绑定规则名
            - "rule1_paragraph_number": 规则一段落序号匹配
            - "rule2_position_mapping": 规则二位置描述映射
            - "rule3_text_anchor": 规则三文本锚点匹配
            - "rule4_unbound": 规则四兜底未绑定
        need_review: 是否需要人工复核（confidence < 0.7）
        location_desc: 原始 VLM 位置描述
        anchor_text: 锚点文本（规则三命中时的关键词）
    """

    latex: str
    formula_id: str
    bound_paragraph_idx: Optional[int]
    confidence: float
    binding_rule: str
    need_review: bool
    location_desc: str
    anchor_text: str = ""


# ---------------------------------------------------------------------------
# 各规则的置信度（文档 4.1 节）
# ---------------------------------------------------------------------------


class BindingRule:
    """绑定规则常量"""

    RULE1_PARAGRAPH_NUMBER = "rule1_paragraph_number"
    RULE2_POSITION_MAPPING = "rule2_position_mapping"
    RULE3_TEXT_ANCHOR = "rule3_text_anchor"
    RULE4_UNBOUND = "rule4_unbound"

    # 文档 4.1 节各规则置信度
    CONFIDENCE = {
        RULE1_PARAGRAPH_NUMBER: 0.9,
        RULE2_POSITION_MAPPING: 0.7,
        RULE3_TEXT_ANCHOR: 0.5,
        RULE4_UNBOUND: 0.3,
    }

    # 需要人工复核的阈值（文档 4.1 节：低于 0.7 黄色高亮）
    REVIEW_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# 正则规则
# ---------------------------------------------------------------------------

# 规则一：段落序号匹配
# 匹配"第3段后"、"第 3 段下方"、"paragraph 3"、"after paragraph 3"
PARAGRAPH_NUMBER_PATTERN = re.compile(
    r"(?:第\s*(\d+)\s*段(?:后|下方|下面|之后)?"
    r"|paragraph\s*(\d+)"
    r"|after\s+paragraph\s+(\d+)"
    r"|para\.?\s*(\d+))",
    re.IGNORECASE,
)

# 规则二：位置描述映射
# 匹配"第二栏中部"、"左栏"、"右栏底部"、"top of page"、"page bottom"
COLUMN_PATTERN = re.compile(
    r"(第([一二两二三])栏|左栏|右栏|双栏|单栏"
    r"|left\s*column|right\s*column|second\s*column"
    r"|top\s*of\s*page|bottom\s*of\s*page|page\s*(?:top|bottom)"
    r"|中部|顶部|底部|上部|下部)",
    re.IGNORECASE,
)

# 中文数字 → 阿拉伯数字（用于"第二栏"解析）
CN_NUM_MAP = {"一": 1, "两": 2, "二": 2, "三": 3}

# 位置词 → 垂直区域
VERTICAL_POS_MAP = {
    "顶部": "top", "上部": "top", "top": "top",
    "中部": "middle", "middle": "middle", "中": "middle",
    "底部": "bottom", "下部": "bottom", "bottom": "bottom",
}

# 规则三：文本锚点匹配
# 从 location_desc 中提取可能的锚点关键词（如"softmax 计算公式如下"）
# 锚点特征：含"如下"、"以下"、"公式"、"equation"、"式"等触发词的短语
ANCHOR_TRIGGER = re.compile(
    r"([^.!?。！？\n]{3,40}?(?:如下|以下|formula|equation|式|公式)[^.!?。！？\n]*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 公式绑定器（核心）
# ---------------------------------------------------------------------------


class FormulaBinder:
    """公式与段落绑定器（四级规则策略，非 IoU 对齐）

    文档 4.1 节明确：Qwen-VL-Max 不返回坐标框，无法做 IoU 对齐。
    本类实现"规则匹配为主、坐标辅助为辅"的四级绑定策略。

    用法：
        binder = FormulaBinder()
        bound = binder.bind(vlm_formulas, page_layout)
        for f in bound:
            if f.need_review:
                print(f"⚠️ 公式 {f.formula_id} 需人工复核: {f.location_desc}")
    """

    def bind(
        self,
        formulas: List[VlmFormula],
        page_layout: PageLayout,
    ) -> List[BoundFormula]:
        """对每个公式依次尝试四级规则绑定

        Args:
            formulas: VLM 识别出的公式列表
            page_layout: pdfplumber 抽取的页面版面
        Returns:
            BoundFormula 列表（顺序与输入一致）
        """
        if not formulas:
            return []

        results: List[BoundFormula] = []
        for formula in formulas:
            bound = self._bind_one(formula, page_layout)
            results.append(bound)

            log_extra = {
                "formula_id": bound.formula_id,
                "rule": bound.binding_rule,
                "confidence": bound.confidence,
                "paragraph_idx": bound.bound_paragraph_idx,
                "need_review": bound.need_review,
            }
            if bound.need_review:
                logger.info("公式绑定需复核: %s", log_extra)
            else:
                logger.debug("公式绑定成功: %s", log_extra)

        return results

    # ------------------------------------------------------------------
    # 四级规则（按优先级顺序尝试）
    # ------------------------------------------------------------------

    def _bind_one(
        self,
        formula: VlmFormula,
        page_layout: PageLayout,
    ) -> BoundFormula:
        """对单个公式依次尝试四级规则，首个命中的生效"""
        # 规则一：段落序号匹配（置信度 0.9，最高优先级）
        para_idx = self._rule1_paragraph_number(formula, page_layout)
        if para_idx is not None:
            return self._make_bound(
                formula, para_idx,
                BindingRule.RULE1_PARAGRAPH_NUMBER,
            )

        # 规则二：位置描述映射（置信度 0.7）
        para_idx = self._rule2_position_mapping(formula, page_layout)
        if para_idx is not None:
            return self._make_bound(
                formula, para_idx,
                BindingRule.RULE2_POSITION_MAPPING,
            )

        # 规则三：文本锚点匹配（置信度 0.5）
        para_idx, anchor = self._rule3_text_anchor(formula, page_layout)
        if para_idx is not None:
            bound = self._make_bound(
                formula, para_idx,
                BindingRule.RULE3_TEXT_ANCHOR,
            )
            bound.anchor_text = anchor
            return bound

        # 规则四：兜底未绑定（置信度 0.3）
        return self._make_bound(
            formula, None,
            BindingRule.RULE4_UNBOUND,
        )

    # ------------------------------------------------------------------
    # 规则一：段落序号匹配（置信度 0.9）
    # ------------------------------------------------------------------

    def _rule1_paragraph_number(
        self,
        formula: VlmFormula,
        page_layout: PageLayout,
    ) -> Optional[int]:
        """规则一：VLM 输出明确标注段落序号时按序号绑定

        匹配示例：
        - "第3段后" → 绑定到段落 idx=2（0-based）
        - "第 3 段下方是公式(1)" → 绑定到段落 idx=2
        - "paragraph 3" → 绑定到段落 idx=2
        - "after paragraph 3" → 绑定到段落 idx=2

        文档 4.1 节：覆盖约 70% 用例，准确率最高
        """
        desc = formula.location_desc
        if not desc:
            return None

        matches = PARAGRAPH_NUMBER_PATTERN.findall(desc)
        if not matches:
            return None

        # findall 返回元组，取第一个非空捕获组
        for match in matches:
            for g in match:
                if g:
                    try:
                        para_num = int(g)
                    except ValueError:
                        continue
                    # VLM 说"第3段"是 1-based，转为 0-based
                    # "第3段后"绑定到段落 idx=2（即第3段本身，公式在其后）
                    target_idx = para_num - 1
                    if 0 <= target_idx < len(page_layout.paragraphs):
                        return target_idx
                    # 超出范围：可能 VLM 数错了，尝试绑定到最后一段
                    if target_idx >= len(page_layout.paragraphs):
                        return len(page_layout.paragraphs) - 1
        return None

    # ------------------------------------------------------------------
    # 规则二：位置描述映射（置信度 0.7）
    # ------------------------------------------------------------------

    def _rule2_position_mapping(
        self,
        formula: VlmFormula,
        page_layout: PageLayout,
    ) -> Optional[int]:
        """规则二：VLM 输出栏位 + 垂直位置时映射到坐标区域

        匹配示例：
        - "第二栏中部" → 中轴线右侧 + 垂直中间区域的段落
        - "左栏底部" → 中轴线左侧 + 垂直底部段落
        - "page bottom" → 垂直底部段落

        文档 4.1 节：结合 pdfplumber 中轴线将"左/中/右"映射到坐标区域
        """
        desc = formula.location_desc
        if not desc:
            return None

        matches = COLUMN_PATTERN.findall(desc)
        if not matches:
            return None

        # 解析栏位（左/右）和垂直位置（顶/中/底）
        column = self._parse_column(desc, page_layout)
        vertical = self._parse_vertical(desc)

        # 在对应区域找最匹配的段落
        candidates = page_layout.paragraphs
        if column is not None:
            # 按栏位过滤
            filtered = [p for p in candidates if p.region == column]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        # 按垂直位置选择
        if vertical:
            return self._select_by_vertical(candidates, vertical, page_layout)

        # 无垂直位置信息：取该栏位第一个段落
        return candidates[0].idx

    def _parse_column(self, desc: str, page_layout: PageLayout) -> Optional[str]:
        """解析栏位描述为 'left' / 'right' / 'center'"""
        desc_lower = desc.lower()

        # 单栏文档：所有段落都是 center
        if "单栏" in desc or "single" in desc_lower:
            return "center"

        # 中文栏位
        if "左栏" in desc or "left" in desc_lower:
            return "left"
        if "右栏" in desc or "right" in desc_lower:
            return "right"

        # "第二栏" / "两栏" / "双栏" → 假设第二栏是右栏
        for cn_num, num in CN_NUM_MAP.items():
            if f"第{cn_num}栏" in desc:
                return "left" if num == 1 else "right"
        if "第二栏" in desc or "second" in desc_lower:
            return "right"
        if "第一栏" in desc or "first" in desc_lower:
            return "left"

        return None

    @staticmethod
    def _parse_vertical(desc: str) -> Optional[str]:
        """解析垂直位置为 'top' / 'middle' / 'bottom'"""
        desc_lower = desc.lower()
        for keyword, pos in VERTICAL_POS_MAP.items():
            if keyword in desc or keyword in desc_lower:
                return pos
        return None

    @staticmethod
    def _select_by_vertical(
        candidates: List[Paragraph],
        vertical: str,
        page_layout: PageLayout,
    ) -> Optional[int]:
        """按垂直位置选择段落"""
        if not candidates:
            return None

        page_height = page_layout.height
        if page_height <= 0:
            return candidates[0].idx

        # 按段落中心 y 坐标分段
        if vertical == "top":
            # 顶部：y0 < page_height * 0.33
            filtered = [p for p in candidates if p.y0 < page_height * 0.33]
        elif vertical == "bottom":
            # 底部：y0 > page_height * 0.66
            filtered = [p for p in candidates if p.y0 > page_height * 0.66]
        else:
            # 中部：page_height * 0.33 <= y0 <= page_height * 0.66
            filtered = [
                p for p in candidates
                if page_height * 0.33 <= p.y0 <= page_height * 0.66
            ]

        if filtered:
            return filtered[0].idx
        # 区域内无段落：返回该栏位第一个
        return candidates[0].idx

    # ------------------------------------------------------------------
    # 规则三：文本锚点匹配（置信度 0.5）
    # ------------------------------------------------------------------

    def _rule3_text_anchor(
        self,
        formula: VlmFormula,
        page_layout: PageLayout,
    ) -> tuple:
        """规则三：VLM 输出公式周围的上下文关键词时搜索锚点段落

        匹配示例：
        - "softmax 计算公式如下" → 在 pdfplumber 文本中搜索"softmax"所在段落
        - "the loss function is defined as follows" → 搜索"loss function"

        文档 4.1 节：兜底策略，覆盖前两条规则未命中的用例
        """
        desc = formula.location_desc
        if not desc:
            return (None, "")

        # 从 location_desc 提取锚点短语
        anchor = self._extract_anchor(desc)
        if not anchor:
            return (None, "")

        # 在 pdfplumber 段落中搜索锚点
        anchor_lower = anchor.lower()
        for para in page_layout.paragraphs:
            if anchor_lower in para.text.lower():
                return (para.idx, anchor)

        # 锚点未精确命中：尝试锚点的关键词分词后匹配
        keywords = self._extract_keywords(anchor)
        if keywords:
            best_idx = self._fuzzy_match(keywords, page_layout.paragraphs)
            if best_idx is not None:
                return (best_idx, anchor)

        return (None, "")

    @staticmethod
    def _extract_anchor(desc: str) -> str:
        """从位置描述中提取锚点短语"""
        # 优先匹配含"如下/以下/formula/equation"的短语
        matches = ANCHOR_TRIGGER.findall(desc)
        if matches:
            # 取最长的匹配（信息最丰富）
            return max(matches, key=len).strip()

        # 无触发词：取描述中的名词短语（简单启发式）
        # 去除位置词后剩余的部分
        cleaned = COLUMN_PATTERN.sub("", desc)
        cleaned = PARAGRAPH_NUMBER_PATTERN.sub("", cleaned)
        cleaned = cleaned.strip("，。,.;:、 \n\t")
        if len(cleaned) >= 3:
            return cleaned
        return ""

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """从锚点文本提取关键词（用于模糊匹配）"""
        # 简单分词：英文按空格，中文按字符
        import re as _re
        keywords = _re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", text.lower())
        # 中文连续字符段
        cn_segments = _re.findall(r"[\u4e00-\u9fff]{2,}", text)
        keywords.extend(cn_segments)
        # 过滤过短或停用词
        stop_words = {"the", "a", "an", "is", "are", "of", "and", "or", "in", "on", "to", "for"}
        return [k for k in keywords if len(k) >= 2 and k not in stop_words]

    @staticmethod
    def _fuzzy_match(
        keywords: List[str],
        paragraphs: List[Paragraph],
    ) -> Optional[int]:
        """关键词模糊匹配段落"""
        if not keywords or not paragraphs:
            return None

        best_idx: Optional[int] = None
        best_score = 0
        for para in paragraphs:
            text_lower = para.text.lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_idx = para.idx

        # 至少匹配 1 个关键词才算命中
        return best_idx if best_score > 0 else None

    # ------------------------------------------------------------------
    # 构造结果
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bound(
        formula: VlmFormula,
        para_idx: Optional[int],
        rule: str,
    ) -> BoundFormula:
        """构造 BoundFormula"""
        confidence = BindingRule.CONFIDENCE[rule]
        return BoundFormula(
            latex=formula.latex,
            formula_id=formula.formula_id,
            bound_paragraph_idx=para_idx,
            confidence=confidence,
            binding_rule=rule,
            need_review=confidence < BindingRule.REVIEW_THRESHOLD,
            location_desc=formula.location_desc,
        )


# ---------------------------------------------------------------------------
# 统计函数（用于观测）
# ---------------------------------------------------------------------------


def binding_stats(bound_formulas: List[BoundFormula]) -> dict:
    """统计四级规则的命中情况（对接文档 6.1 可观测性看板）"""
    total = len(bound_formulas)
    if total == 0:
        return {"total": 0}

    rule_counts = {
        BindingRule.RULE1_PARAGRAPH_NUMBER: 0,
        BindingRule.RULE2_POSITION_MAPPING: 0,
        BindingRule.RULE3_TEXT_ANCHOR: 0,
        BindingRule.RULE4_UNBOUND: 0,
    }
    review_count = 0
    for f in bound_formulas:
        rule_counts[f.binding_rule] = rule_counts.get(f.binding_rule, 0) + 1
        if f.need_review:
            review_count += 1

    return {
        "total": total,
        "by_rule": rule_counts,
        "need_review": review_count,
        "review_rate": round(review_count / total, 3),
        "avg_confidence": round(
            sum(f.confidence for f in bound_formulas) / total, 3
        ),
    }

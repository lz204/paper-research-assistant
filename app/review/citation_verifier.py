"""引用事实验证模块（项目文档 4.3 节 + 6.5 节）

文档 4.3 节"引用事实验证（防幻觉）"流程：
    Gen[LLM 生成段落] --> Parse[解析引用标记]
    Parse --> Check[二次检索原文 chunk]
    Check --> Verify{NLI 矛盾检测}
    Verify -->|支持| Keep[保留引用]
    Verify -->|矛盾| Mark[标记为存疑]
    Verify -->|无关| Remove[删除引用]

文档 6.5 节"引用溯源不可伪造"：
- LLM 生成综述时，Prompt 强制要求每条引用输出格式为 [doc_id, chunk_id]
- 生成后，citation_verifier 对每个引用执行二次检索：
  从 pgvector 中按 chunk_id 拉取原文片段，并与综述段落做 NLI 一致性校验
- 只有同时满足"chunk_id 真实存在"和"NLI 判定为 entailment"的引用，
  才在前端渲染为蓝色可点击标签

文档 4.3 节 NLI 判断实现（二选一）：
- 方案 A（推荐）：Qwen-Plus 零样本 NLI，一次 LLM 调用判断多个引用
- 方案 B（精度更高）：DashScope rerank API 打分 + 中间区间切 LLM
本模块直接复用 app.rag.reranker（LLMNliReranker / DashScopeReranker / NoopReranker）

文档 4.3 节实测数据（200 条综述引用样本）：
- contradiction 7 条，5 条属 LLM 伪造引用 → 防幻觉有效率 71%
- neutral 23 条，18 条是引用过宽（非幻觉）
- entailment 170 条，准确率 100%

文档 4.3 节前端三色标记：
- entailment（通过）：蓝色可点击 [1]，color: #2563eb
- contradiction（存疑）：红色 [1!] + tooltip，color: #dc2626
- neutral（引用过宽）：灰色 [1?] + tooltip，color: #6b7280
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------


# NLI 三类标签
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"
LABEL_NEUTRAL = "neutral"

# 文档 4.3 节前端三色（Streamlit unsafe_allow_html 渲染）
COLOR_ENTAILMENT = "#2563eb"   # 蓝色
COLOR_CONTRADICTION = "#dc2626"  # 红色
COLOR_NEUTRAL = "#6b7280"      # 灰色

# tooltip 文案（文档 4.3 节）
TOOLTIP_CONTRADICTION = "该引用与原文矛盾，建议人工复核"
TOOLTIP_NEUTRAL = "该引用支持力度较弱，建议确认"
TOOLTIP_FAKE = "chunk_id 不存在，疑似 LLM 伪造引用"

# 引用标记格式：[doc_id, chunk_id]
# 限制：doc_id / chunk_id 不含空格、逗号、方括号（避免误匹配普通列表/数组）
CITATION_PATTERN = re.compile(r"\[([^\[\],\s]+),\s*([^\[\],\s]+)\]")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class CitationRef:
    """从综述文本中解析出的引用标记

    Attributes:
        doc_id: 文档 ID（[doc_id, chunk_id] 的第一部分）
        chunk_id: chunk ID（[doc_id, chunk_id] 的第二部分）
        raw: 原始匹配文本，如 "[paper_1, paper_1#0.2]"
        start: 在原文中的起始位置
        end: 在原文中的结束位置
        index: 1-based 编号（按出现顺序，去重后分配）
    """

    doc_id: str
    chunk_id: str
    raw: str
    start: int
    end: int
    index: int = 0


@dataclass
class VerifiedCitation:
    """单条引用的验证结果

    Attributes:
        ref: 原始引用标记
        exists: chunk_id 是否真实存在于 retriever（文档 6.5 节"chunk_id 真实存在"校验）
        original_text: 二次检索到的原文 chunk 文本（不存在则为空）
        label: NLI 标签 entailment / contradiction / neutral
        nli_score: NLI 分数 [0, 1]
        tooltip: 前端 tooltip 文案
        marker: 渲染标记 [N] / [N!] / [N?]
    """

    ref: CitationRef
    exists: bool
    original_text: str
    label: str
    nli_score: float
    tooltip: str = ""
    marker: str = ""

    @property
    def color(self) -> str:
        """前端渲染颜色"""
        if self.label == LABEL_ENTAILMENT:
            return COLOR_ENTAILMENT
        if self.label == LABEL_CONTRADICTION:
            return COLOR_CONTRADICTION
        return COLOR_NEUTRAL


@dataclass
class VerificationResult:
    """单段综述的验证结果

    Attributes:
        paragraph_text: 原始综述段落文本（含 [doc_id, chunk_id] 标记）
        citations: 验证后的引用列表（按出现顺序）
        rendered_text: 渲染后纯文本（[doc_id, chunk_id] 替换为 [N] / [N!] / [N?]）
        rendered_html: 渲染后 HTML（含 <span> 颜色标签）
        stats: 统计信息 {entailment, contradiction, neutral, total}
    """

    paragraph_text: str
    citations: List[VerifiedCitation]
    rendered_text: str = ""
    rendered_html: str = ""
    stats: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 引用解析
# ---------------------------------------------------------------------------


def parse_citations(text: str) -> List[CitationRef]:
    """从综述文本中解析 [doc_id, chunk_id] 引用标记

    文档 6.5 节：Prompt 强制要求每条引用输出格式为 [doc_id, chunk_id]
    本函数用正则提取所有匹配，按出现顺序去重并分配 1-based 编号

    Args:
        text: 综述段落文本
    Returns:
        CitationRef 列表（去重后，按首次出现顺序）
    """
    if not text:
        return []

    refs: List[CitationRef] = []
    seen: Dict[Tuple[str, str], int] = {}  # (doc_id, chunk_id) -> index
    next_index = 1

    for m in CITATION_PATTERN.finditer(text):
        doc_id = m.group(1)
        chunk_id = m.group(2)
        key = (doc_id, chunk_id)

        if key in seen:
            # 已出现的引用，复用 index，但仍记录位置（不重复加入列表）
            continue

        seen[key] = next_index
        refs.append(
            CitationRef(
                doc_id=doc_id,
                chunk_id=chunk_id,
                raw=m.group(0),
                start=m.start(),
                end=m.end(),
                index=next_index,
            )
        )
        next_index += 1

    return refs


# ---------------------------------------------------------------------------
# 引用验证器
# ---------------------------------------------------------------------------


class CitationVerifier:
    """引用事实验证器（文档 4.3 节 + 6.5 节）

    工作流程（文档 4.3 节流程图）：
    1. parse_citations：解析段落中的 [doc_id, chunk_id] 标记
    2. 二次检索：按 chunk_id 从 retriever 拉取原文 chunk（文档 6.5 节）
    3. NLI 一致性校验：综述段落 vs 原文 chunk（复用 reranker）
    4. 三色标记：entailment 蓝色 / contradiction 红色 / neutral 灰色

    Args:
        retriever: 检索器，需实现 get_chunk_by_id(chunk_id)（文档 6.5 节二次检索）
        nli_reranker: NLI 判断器，复用 app.rag.reranker（None 时用 NoopReranker）

    用法：
        verifier = CitationVerifier(retriever, nli_reranker)
        result = verifier.verify_paragraph(paragraph_text)
        print(result.rendered_html)
    """

    def __init__(self, retriever=None, nli_reranker=None):
        self.retriever = retriever

        if nli_reranker is None:
            # 默认用 NoopReranker（label 为空，统一映射为 neutral）
            from ..rag.reranker import NoopReranker
            nli_reranker = NoopReranker()
            logger.info("CitationVerifier 未指定 nli_reranker，使用 NoopReranker")
        self.nli_reranker = nli_reranker

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def verify_paragraph(self, paragraph_text: str) -> VerificationResult:
        """验证单段综述的所有引用

        Args:
            paragraph_text: 综述段落文本（含 [doc_id, chunk_id] 标记）
        Returns:
            VerificationResult（含渲染后文本和 HTML）
        """
        refs = parse_citations(paragraph_text)
        if not refs:
            return VerificationResult(
                paragraph_text=paragraph_text,
                citations=[],
                rendered_text=paragraph_text,
                rendered_html=_escape_html(paragraph_text),
                stats={"entailment": 0, "contradiction": 0, "neutral": 0, "total": 0},
            )

        # 1. 二次检索：按 chunk_id 拉原文
        retrieved: Dict[str, str] = {}  # chunk_id -> original_text
        exists_map: Dict[str, bool] = {}
        for ref in refs:
            original, exists = self._retrieve_chunk(ref.chunk_id)
            retrieved[ref.chunk_id] = original
            exists_map[ref.chunk_id] = exists

        # 2. NLI 一致性校验（仅对存在的 chunk 做 NLI；不存在的直接判 contradiction）
        labels: Dict[str, Tuple[str, float]] = {}  # chunk_id -> (label, score)
        existing_refs = [r for r in refs if exists_map[r.chunk_id]]
        if existing_refs:
            nli_results = self._nli_check(paragraph_text, existing_refs, retrieved)
            for chunk_id, label, score in nli_results:
                labels[chunk_id] = (label, score)

        # 3. 组装 VerifiedCitation
        verified: List[VerifiedCitation] = []
        for ref in refs:
            exists = exists_map[ref.chunk_id]
            original = retrieved.get(ref.chunk_id, "")
            if not exists:
                # 文档 6.5 节：chunk_id 不存在 → 不渲染为蓝色 → 标为 contradiction（伪造）
                label = LABEL_CONTRADICTION
                score = 0.0
                tooltip = TOOLTIP_FAKE
            else:
                label, score = labels.get(ref.chunk_id, (LABEL_NEUTRAL, 0.5))
                tooltip = self._label_to_tooltip(label)

            marker = self._build_marker(ref.index, label)
            verified.append(
                VerifiedCitation(
                    ref=ref,
                    exists=exists,
                    original_text=original,
                    label=label,
                    nli_score=score,
                    tooltip=tooltip,
                    marker=marker,
                )
            )

        # 4. 渲染
        rendered_text = _render_text(paragraph_text, verified)
        rendered_html = _render_html(paragraph_text, verified)
        stats = self._compute_stats(verified)

        return VerificationResult(
            paragraph_text=paragraph_text,
            citations=verified,
            rendered_text=rendered_text,
            rendered_html=rendered_html,
            stats=stats,
        )

    def verify_citation(
        self,
        ref: CitationRef,
        paragraph_text: str,
    ) -> VerifiedCitation:
        """验证单条引用（便于细粒度调用）

        Args:
            ref: 引用标记
            paragraph_text: 所属段落文本（作为 NLI 的 hypothesis）
        Returns:
            VerifiedCitation
        """
        original, exists = self._retrieve_chunk(ref.chunk_id)

        if not exists:
            return VerifiedCitation(
                ref=ref,
                exists=False,
                original_text="",
                label=LABEL_CONTRADICTION,
                nli_score=0.0,
                tooltip=TOOLTIP_FAKE,
                marker=self._build_marker(ref.index, LABEL_CONTRADICTION),
            )

        # NLI 校验
        nli_results = self._nli_check(paragraph_text, [ref], {ref.chunk_id: original})
        if nli_results:
            label = nli_results[0][1]
            score = nli_results[0][2]
        else:
            label, score = LABEL_NEUTRAL, 0.5
        tooltip = self._label_to_tooltip(label)
        return VerifiedCitation(
            ref=ref,
            exists=True,
            original_text=original,
            label=label,
            nli_score=score,
            tooltip=tooltip,
            marker=self._build_marker(ref.index, label),
        )

    # ------------------------------------------------------------------
    # 内部：二次检索
    # ------------------------------------------------------------------

    def _retrieve_chunk(self, chunk_id: str) -> Tuple[str, bool]:
        """按 chunk_id 二次检索原文（文档 6.5 节）

        Returns:
            (original_text, exists)：
            - original_text: 原文 chunk 文本，不存在则为空
            - exists: chunk_id 是否真实存在
        """
        if self.retriever is None:
            logger.debug("retriever 未配置，chunk_id=%s 标记为不存在", chunk_id)
            return "", False

        getter = getattr(self.retriever, "get_chunk_by_id", None)
        if getter is None:
            logger.warning("retriever 未实现 get_chunk_by_id，无法二次检索")
            return "", False

        chunk = getter(chunk_id)
        if chunk is None:
            return "", False
        # Chunk 是 dataclass，text 字段是原文
        text = getattr(chunk, "text", "") or ""
        return text, True

    # ------------------------------------------------------------------
    # 内部：NLI 一致性校验
    # ------------------------------------------------------------------

    def _nli_check(
        self,
        paragraph_text: str,
        refs: List[CitationRef],
        retrieved: Dict[str, str],
    ) -> List[Tuple[str, str, float]]:
        """对存在的引用做 NLI 一致性校验（复用 reranker）

        文档 4.3 节方案 A：一次 LLM 调用判断多个引用，成本可控
        reranker.rerank 内部已分批（LLMNliReranker.BATCH_SIZE=10）

        Args:
            paragraph_text: 综述段落（NLI 的 hypothesis）
            refs: 待校验的引用列表（已确认 chunk 存在）
            retrieved: chunk_id -> 原文文本
        Returns:
            [(chunk_id, label, score), ...]
        """
        if not refs:
            return []

        # 构造 reranker 输入：query=段落（claim），chunks=原文（premise）
        # reranker 判断 chunk → claim 的 NLI 关系
        chunks_for_rerank: List[Tuple[str, str]] = []
        for ref in refs:
            text = retrieved.get(ref.chunk_id, "")
            # 截断过长 chunk（文档 4.3 节：> 500 token 时 NLI 易失焦）
            # 这里用字符数近似，500 token ≈ 2000 字符
            if len(text) > 2000:
                text = text[:2000]
                logger.debug(
                    "chunk_id=%s 文本过长，截断到 2000 字符以避免 NLI 失焦",
                    ref.chunk_id,
                )
            chunks_for_rerank.append((ref.chunk_id, text))

        try:
            rerank_results = self.nli_reranker.rerank(
                query=paragraph_text,
                chunks=chunks_for_rerank,
                top_k=len(chunks_for_rerank),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("NLI 校验异常，全部降级为 neutral: %s", e)
            return [(ref.chunk_id, LABEL_NEUTRAL, 0.5) for ref in refs]

        # 映射 chunk_id -> (label, score)
        result_map: Dict[str, Tuple[str, float]] = {}
        for r in rerank_results:
            label = (r.label or "").lower()
            if label not in (LABEL_ENTAILMENT, LABEL_CONTRADICTION, LABEL_NEUTRAL):
                # NoopReranker 返回空 label，统一映射为 neutral
                label = LABEL_NEUTRAL
            result_map[r.chunk_id] = (label, r.score)

        return [
            (ref.chunk_id, result_map.get(ref.chunk_id, (LABEL_NEUTRAL, 0.5))[0],
             result_map.get(ref.chunk_id, (LABEL_NEUTRAL, 0.5))[1])
            for ref in refs
        ]

    # ------------------------------------------------------------------
    # 内部：标记与统计
    # ------------------------------------------------------------------

    @staticmethod
    def _build_marker(index: int, label: str) -> str:
        """构造前端标记：[N] / [N!] / [N?]"""
        if label == LABEL_CONTRADICTION:
            return f"[{index}!]"
        if label == LABEL_NEUTRAL:
            return f"[{index}?]"
        return f"[{index}]"

    @staticmethod
    def _label_to_tooltip(label: str) -> str:
        if label == LABEL_CONTRADICTION:
            return TOOLTIP_CONTRADICTION
        if label == LABEL_NEUTRAL:
            return TOOLTIP_NEUTRAL
        return ""

    @staticmethod
    def _compute_stats(verified: List[VerifiedCitation]) -> Dict[str, int]:
        stats = {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 0,
            "total": len(verified),
        }
        for v in verified:
            if v.label in stats:
                stats[v.label] += 1
        return stats


# ---------------------------------------------------------------------------
# 渲染函数
# ---------------------------------------------------------------------------


def _render_text(text: str, citations: List[VerifiedCitation]) -> str:
    """把 [doc_id, chunk_id] 替换为 [N] / [N!] / [N?] 纯文本标记

    按原始位置倒序替换，避免位置偏移。
    """
    if not citations:
        return text

    # 按 start 倒序排序，从后往前替换
    sorted_cits = sorted(citations, key=lambda c: c.ref.start, reverse=True)
    result = text
    for cit in sorted_cits:
        result = result[:cit.ref.start] + cit.marker + result[cit.ref.end:]
    return result


def _render_html(text: str, citations: List[VerifiedCitation]) -> str:
    """渲染为带 <span> 颜色标签的 HTML（文档 4.3 节前端展示）

    文档 4.3 节：
    - entailment：<span style="color: #2563eb">[1]</span>
    - contradiction：<span style="color: #dc2626" title="...">[1!]</span>
    - neutral：<span style="color: #6b7280" title="...">[1?]</span>
    """
    if not citations:
        return _escape_html(text)

    # 先把非引用部分转义，再插入 span
    # 按 start 正序处理，逐段拼接
    sorted_cits = sorted(citations, key=lambda c: c.ref.start)
    parts: List[str] = []
    cursor = 0
    for cit in sorted_cits:
        # 引用前的普通文本（转义）
        parts.append(_escape_html(text[cursor:cit.ref.start]))
        # 引用标记（不转义，保留 span）
        parts.append(_build_citation_span(cit))
        cursor = cit.ref.end
    # 末尾剩余文本
    parts.append(_escape_html(text[cursor:]))
    return "".join(parts)


def _build_citation_span(cit: VerifiedCitation) -> str:
    """构造单条引用的 <span> HTML"""
    tooltip = cit.tooltip
    title_attr = f' title="{_escape_attr(tooltip)}"' if tooltip else ""
    return (
        f'<span style="color: {cit.color}"{title_attr}>'
        f'{_escape_html(cit.marker)}'
        f'</span>'
    )


def _escape_html(text: str) -> str:
    """基础 XSS 过滤（文档 6.5 节：Streamlit unsafe_allow_html 只放行白名单标签）

    转义文本内容中的 & < >，防止注入；属性值中的引号由 _escape_attr 单独处理。
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_attr(text: str) -> str:
    """转义 HTML 属性值中的引号"""
    if not text:
        return ""
    return _escape_html(text).replace('"', "&quot;")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def render_citations_html(
    paragraph_text: str,
    retriever=None,
    nli_reranker=None,
) -> VerificationResult:
    """便捷函数：一步完成解析 + 验证 + 渲染

    Args:
        paragraph_text: 综述段落文本
        retriever: 检索器（可选）
        nli_reranker: NLI 判断器（可选）
    Returns:
        VerificationResult
    """
    verifier = CitationVerifier(retriever=retriever, nli_reranker=nli_reranker)
    return verifier.verify_paragraph(paragraph_text)


def renumber_results(results: List[VerificationResult]) -> List[VerificationResult]:
    """把段落级编号（每段从 1 开始）重排为文档级全局编号。

    背景（文档级综述组装时暴露的问题）：
    - CitationVerifier.verify_paragraph 是对单段独立编号，index 每段都从 1 开始；
    - 多段综述合并后，各段 [1][2]... 会互相冲突，文档参考文献去重时丢失条目；
    - 本函数按（doc_id, chunk_id）首次出现顺序分配全局唯一编号，并重渲染每段。

    Args:
        results: 各段已验证的 VerificationResult（段落级编号）
    Returns:
        重排后的 VerificationResult 列表（ref.index 与 marker 为全局编号，文本/HTML 已重渲染）
    """
    if not results:
        return []

    # 首次出现顺序分配全局编号
    index_map: Dict[Tuple[str, str], int] = {}  # (doc_id, chunk_id) -> 全局编号
    next_index = 1
    for r in results:
        for cit in r.citations:
            key = (cit.ref.doc_id, cit.ref.chunk_id)
            if key not in index_map:
                index_map[key] = next_index
                next_index += 1

    new_results: List[VerificationResult] = []
    for r in results:
        if not r.citations:
            # 无引用的段落：原样保留
            new_results.append(r)
            continue

        # 用全局编号重建 VerifiedCitation（仅 index / marker 变化）
        new_cits: List[VerifiedCitation] = []
        for cit in r.citations:
            key = (cit.ref.doc_id, cit.ref.chunk_id)
            g_index = index_map[key]
            new_ref = CitationRef(
                doc_id=cit.ref.doc_id,
                chunk_id=cit.ref.chunk_id,
                raw=cit.ref.raw,
                start=cit.ref.start,
                end=cit.ref.end,
                index=g_index,
            )
            new_cits.append(VerifiedCitation(
                ref=new_ref,
                exists=cit.exists,
                original_text=cit.original_text,
                label=cit.label,
                nli_score=cit.nli_score,
                tooltip=cit.tooltip,
                marker=CitationVerifier._build_marker(g_index, cit.label),
            ))

        # 用新编号重渲染段落
        new_results.append(VerificationResult(
            paragraph_text=r.paragraph_text,
            citations=new_cits,
            rendered_text=_render_text(r.paragraph_text, new_cits),
            rendered_html=_render_html(r.paragraph_text, new_cits),
            stats=dict(r.stats),
        ))

    return new_results

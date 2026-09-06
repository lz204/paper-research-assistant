"""GB/T 7714-2015《信息与文献 参考文献著录规则》格式化（顺序编码制）。

实现 5 种最常见类型：
  [J] 期刊论文
  [C] 会议论文
  [M] 专著
  [D] 学位论文
  [EB/OL] 电子预印本（arXiv / URL 在线资源，默认兜底）

格式遵循 GB/T 7714-2015 顺序编码制常见写法（不做严格到句号/冒号空格级校验，
满足工程场景下"人读+Word 直接导出"即可）。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.common.paper_meta import PaperMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 格式化器
# ---------------------------------------------------------------------------


class GBT7714Formatter:
    """把 PaperMeta 格式化为 GB/T 7714 顺序编码制条目。"""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def format(self, meta: PaperMeta, index: Optional[int] = None) -> str:
        """格式化单条引用。

        Args:
            meta: 文献元数据
            index: 引用序号（None 时不加 [1] 前缀）
        """
        t = meta.doc_type.upper() if meta.doc_type else "EB"
        body_map = {
            "J": self._fmt_journal,
            "C": self._fmt_conference,
            "M": self._fmt_monograph,
            "D": self._fmt_thesis,
        }
        fn = body_map.get(t, self._fmt_electronic)
        body = fn(meta)
        prefix = f"[{index}] " if index is not None else ""
        return f"{prefix}{body}"

    def format_many(self, metas_with_index: Dict[int, PaperMeta]) -> Dict[int, str]:
        """批量格式化：{index: PaperMeta} → {index: 格式化字符串}。"""
        return {idx: self.format(meta, index=idx) for idx, meta in metas_with_index.items()}

    # ------------------------------------------------------------------
    # 各类型格式化实现
    # ------------------------------------------------------------------

    def _fmt_journal(self, m: PaperMeta) -> str:
        # [序号] 作者. 题名[J]. 刊名, 年, 卷(期): 页码.
        parts = []
        parts.append(m.authors_text())
        parts.append(f"{m.title or '[题名不详]'}[J]")
        tail = ""
        if m.journal:
            tail += m.journal
        if m.year:
            tail += f", {m.year}" if tail else str(m.year)
        if m.volume:
            if m.issue:
                tail += f", {m.volume}({m.issue})"
            else:
                tail += f", {m.volume}"
        if m.pages:
            tail += f": {m.pages}"
        if tail:
            parts.append(tail)
        return self._join(parts) + self._doi_url_tail(m)

    def _fmt_conference(self, m: PaperMeta) -> str:
        # 作者. 题名[C]//会议名, 会议地点, 会议年: 页码.
        parts = [m.authors_text(), f"{m.title or '[题名不详]'}[C]"]
        body = ""
        if m.conference:
            body += f"//{m.conference}"
        loc_year = ", ".join(x for x in [m.conference_location, str(m.year) if m.year else ""] if x)
        if loc_year:
            body += f", {loc_year}" if body else loc_year
        if m.pages:
            body += f": {m.pages}"
        if body:
            parts.append(body)
        return self._join(parts) + self._doi_url_tail(m)

    def _fmt_monograph(self, m: PaperMeta) -> str:
        # 作者. 书名[M]. 出版地: 出版社, 出版年: 页码.
        parts = [m.authors_text(), f"{m.title or '[书名不详]'}[M]"]
        pub = ""
        location = m.conference_location or ""
        if location and m.publisher:
            pub += f"{location}: {m.publisher}"
        elif m.publisher:
            pub += m.publisher
        if m.year:
            pub += f", {m.year}" if pub else str(m.year)
        if m.pages:
            pub += f": {m.pages}"
        if pub:
            parts.append(pub)
        return self._join(parts) + self._doi_url_tail(m)

    def _fmt_thesis(self, m: PaperMeta) -> str:
        # 作者. 题名[D]. 学位授予单位所在地: 学位授予单位, 年份.
        parts = [m.authors_text(), f"{m.title or '[题名不详]'}[D]"]
        grant = ""
        location = m.conference_location or ""
        if location and m.degree_grantor:
            grant += f"{location}: {m.degree_grantor}"
        elif m.degree_grantor:
            grant += m.degree_grantor
        if m.year:
            grant += f", {m.year}" if grant else str(m.year)
        if grant:
            parts.append(grant)
        return self._join(parts) + self._doi_url_tail(m)

    def _fmt_electronic(self, m: PaperMeta) -> str:
        # 作者. 题名[EB/OL]. (发布日期)[访问日期]. URL. / DOI
        parts = [m.authors_text(), f"{m.title or '[题名不详]'}[EB/OL]"]
        date_parts = []
        if m.year:
            ymd = f"{m.year:04d}"
            if m.month:
                ymd += f"-{m.month:02d}"
            date_parts.append(ymd)
        mid = ""
        if date_parts:
            mid = f"({date_parts[0]})"
        if m.access_date:
            mid += f"[{m.access_date}]" if mid else f"[{m.access_date}]"
        if mid:
            parts.append(mid)
        return self._join(parts) + self._doi_url_tail(m, force_url=True)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _join(parts) -> str:
        # 每段之间用 ". " 连接，末尾补句号
        cleaned = [p.strip() for p in parts if p and p.strip()]
        body = ". ".join(cleaned)
        if not body.endswith("."):
            body += "."
        return body

    @staticmethod
    def _doi_url_tail(m: PaperMeta, force_url: bool = False) -> str:
        extras: list[str] = []
        if m.doi:
            extras.append(f"DOI: {m.doi}")
        if (force_url or not extras) and m.url:
            extras.append(m.url)
        if not extras:
            return ""
        return " " + " / ".join(extras)

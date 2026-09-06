"""公共数据结构与辅助类。

PaperMeta：一篇文献的"书目元数据"（作者/年份/期刊/卷期/页码/DOI/…），
用于生成 GB/T 7714 标准参考文献格式，以及在 Word 导出文末列出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PaperMeta:
    """单篇文献的书目元数据（GB/T 7714 格式化所需字段）。

    文献类型 doc_type 对应国标标识：
        J  期刊论文
        C  会议论文
        M  专著
        D  学位论文
        R  科技报告
        S  标准
        P  专利
        DB 数据库
        EB 电子文献（含 arXiv: preprint 视为 EB/OL）
    """

    paper_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    month: Optional[int] = None
    doc_type: str = "EB"          # 默认：电子预印本
    # 期刊/会议/学位授予单位
    journal: str = ""             # [J] 期刊名
    conference: str = ""          # [C] 会议名
    conference_location: str = "" # [C] 会议地点
    publisher: str = ""           # [M] 出版社 / [D] 学位授予单位所在名
    degree_grantor: str = ""      # [D] 学位授予单位
    # 卷、期、页码
    volume: str = ""
    issue: str = ""
    pages: str = ""               # 如 "123-145"
    # 标识
    doi: str = ""
    issn: str = ""
    isbn: str = ""
    # 电子资源
    url: str = ""
    access_date: str = ""         # 访问日期，如 "2026-08-10"
    # 论文自身字段
    paper_language: str = "en"
    paper_keywords: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 构造辅助
    # ------------------------------------------------------------------

    @classmethod
    def from_paper(cls, paper_obj, **overrides) -> "PaperMeta":
        """从 qa_dataset 的 Paper 对象或任何携带 paper_id/title/year/authors/url/keywords 的对象构造。

        未知字段用 LLM 补（由调用方决定，PaperMeta 本身不调 LLM）。
        """
        meta = PaperMeta(
            paper_id=getattr(paper_obj, "paper_id", ""),
            title=getattr(paper_obj, "title", ""),
            year=getattr(paper_obj, "year", None),
            authors=list(getattr(paper_obj, "authors", []) or []),
            url=getattr(paper_obj, "url", ""),
            paper_keywords=list(getattr(paper_obj, "keywords", []) or []),
        )
        # url 若包含 "arxiv" 则判定为预印本
        if "arxiv.org" in meta.url.lower() or meta.url.lower().startswith("arxiv:"):
            meta.doc_type = "EB"
            if not meta.access_date:
                from datetime import date
                meta.access_date = date.today().isoformat()
        for k, v in overrides.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        return meta

    def authors_text(self, max_display: int = 3) -> str:
        """作者列表排版：多于 3 人时 "张三, 李四, 王五, 等"。"""
        authors = [a for a in self.authors if a]
        if not authors:
            return "[著者不详]"
        # GB/T 7714：作者 1~3 全部列出，4 人及以上列前 3 + "等"
        if len(authors) <= max_display:
            return ", ".join(authors)
        return ", ".join(authors[:max_display]) + ", 等"

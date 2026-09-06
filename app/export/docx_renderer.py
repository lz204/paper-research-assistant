"""Word 导出模块（项目文档第 2/3/10 章）

文档第 2 章"核心功能"：
- 综述草稿生成：跨多篇文献 RAG 检索 + 引用溯源，生成带引用的综述初稿

文档第 3 章架构图：
    Review[综述主题] --> RAG2[多文档检索] --> MMR[MMR 去重]
    MMR --> Gen[Map-Reduce 生成] --> Verify[引用事实验证]
    Verify --> Docx[Word 导出]

文档第 10 章技术栈：
- 导出 | python-docx | Word 综述

容错：python-docx 未安装时降级返回 markdown 文本，
保证无第三方依赖也能跑通全链路（文档 6.3 节降级思想）。

设计：
- DocxRenderResult dataclass：渲染结果（path / bytes / markdown 降级）
- BaseDocxRenderer ABC：支持 python-docx / Markdown 降级热插拔
- PythonDocxRenderer：python-docx 真实实现
- MarkdownDocxRenderer：python-docx 不可用时的降级实现（输出 .md）
- create_docx_renderer：工厂函数

文档要求：
- 章节标题（Heading 1/2）+ 正文段落 + 引用编号
- 引用编号 [N] / [N?] / [N!]（与 citation_verifier 一致，三色标记仅在 HTML 生效）
- 文末附"参考文献"列表，按引用顺序列出 doc_id / chunk_id
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ReviewDocument:
    """完整综述文档（用于 Word 导出和前端展示）"""

    topic: str
    title: str = ""
    outline: List[Dict] = field(default_factory=list)
    paragraphs: List[Dict] = field(default_factory=list)
    verified: bool = False
    full_text: str = ""
    full_html: str = ""
    citation_stats: Dict[str, int] = field(default_factory=dict)
    references: List[Dict] = field(default_factory=list)
    # 外部注入：每篇 doc_id 对应的 PaperMeta（来自 GB/T 7714 格式化）
    paper_metas: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # GB/T 7714 参考文献格式化
    # ------------------------------------------------------------------

    def references_gbt7714(self, paper_metas: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """把当前 references 用 GB/T 7714 格式化，返回 [{index, doc_id, formatted}]。

        若调用方传入 paper_metas（推荐，按 doc_id 索引 PaperMeta）则优先使用；
        否则使用 self.paper_metas；再无可用信息则降级为占位条目。
        """
        metas = paper_metas or self.paper_metas or {}
        try:
            from app.review.citation_formatter import GBT7714Formatter
            from app.common.paper_meta import PaperMeta
        except ImportError:
            logger.exception("引用格式化组件缺失，降级返回原始引用")
            return [{"index": r.get("index", i + 1),
                     "doc_id": r.get("doc_id", ""),
                     "formatted": f"{r.get('doc_id', '')} (无元数据)"}
                    for i, r in enumerate(self.references)]
        fmt = GBT7714Formatter()
        out: List[Dict] = []
        for ref in self.references:
            index = ref.get("index")
            doc_id = ref.get("doc_id", "")
            meta = metas.get(doc_id)
            if isinstance(meta, PaperMeta):
                formatted = fmt.format(meta, index=index)
            else:
                # 无 PaperMeta：构造最小占位条目（doc_id 作为标题）
                placeholder = PaperMeta(
                    paper_id=doc_id,
                    title=doc_id or "无题名文献",
                    url=ref.get("url", ""),
                )
                formatted = fmt.format(placeholder, index=index)
            out.append({"index": index, "doc_id": doc_id, "formatted": formatted})
        return out



@dataclass
class DocxRenderResult:
    """Word 渲染结果

    Attributes:
        path: 渲染文件路径（保存到文件时）
        bytes: 二进制内容（保存到内存时）
        markdown: 降级模式下的 markdown 文本
        mode: 实际使用的渲染模式（"docx" / "markdown"）
        size: 文件大小（字节）
    """

    path: str = ""
    bytes: bytes = b""
    markdown: str = ""
    mode: str = "docx"
    size: int = 0


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseDocxRenderer(ABC):
    """Word 渲染抽象基类

    设计目的：python-docx / Markdown 降级热插拔。
    """

    @abstractmethod
    def render(
        self,
        document: ReviewDocument,
        output_path: Optional[str] = None,
    ) -> DocxRenderResult:
        """渲染综述文档

        Args:
            document: 综述文档数据
            output_path: 输出文件路径（None 时返回 bytes）
        Returns:
            DocxRenderResult
        """
        ...


# ---------------------------------------------------------------------------
# python-docx 实现（生产环境首选）
# ---------------------------------------------------------------------------


class PythonDocxRenderer(BaseDocxRenderer):
    """python-docx 渲染实现（文档第 10 章）

    工作流程：
    1. 创建 Document，写标题
    2. 逐章节写 Heading + 段落正文
    3. 文末附参考文献列表（按引用顺序）
    4. 保存到文件或返回 bytes

    容错：python-docx 未安装时由 MarkdownDocxRenderer 降级。

    用法：
        renderer = PythonDocxRenderer()
        result = renderer.render(review_doc, "review.docx")
    """

    def __init__(self):
        try:
            import docx  # type: ignore
            self._docx = docx
            self._available = True
        except ImportError:
            logger.warning("python-docx 未安装，PythonDocxRenderer 不可用")
            self._docx = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def render(
        self,
        document: ReviewDocument,
        output_path: Optional[str] = None,
    ) -> DocxRenderResult:
        if not self._available:
            logger.info("python-docx 不可用，降级为 MarkdownDocxRenderer")
            return MarkdownDocxRenderer().render(document, output_path)

        doc = self._docx.Document()

        # 1. 主标题
        title = document.title or f"综述：{document.topic}"
        doc.add_heading(title, level=0)

        # 元信息
        meta_para = doc.add_paragraph()
        meta_para.add_run(f"主题：{document.topic}\n")
        meta_para.add_run(f"生成时间：{document.created_at}\n")
        stats = document.citation_stats
        if stats:
            meta_para.add_run(
                f"引用统计：共 {stats.get('total', 0)} 条 "
                f"(支持 {stats.get('entailment', 0)} / "
                f"存疑 {stats.get('contradiction', 0)} / "
                f"过宽 {stats.get('neutral', 0)})"
            )

        # 2. 章节内容
        outline_map: Dict[str, Dict] = {
            o.get("section_id", ""): o for o in document.outline
        }
        for para in document.paragraphs:
            section_id = para.get("section_id", "")
            outline = outline_map.get(section_id, {})
            section_title = outline.get("title") or f"章节 {section_id}"

            doc.add_heading(section_title, level=1)

            # 段落正文（优先用渲染后纯文本，去掉 [doc_id, chunk_id] 标记）
            text = para.get("rendered_text") or para.get("text", "")
            if text:
                doc.add_paragraph(text)

            # 引用列表（章节内）
            citations = para.get("citations", [])
            if citations:
                ref_para = doc.add_paragraph()
                ref_para.add_run("本节引用：").bold = True
                for cit in citations:
                    ref_para.add_run(
                        f"\n  [{cit.get('index', '')}] "
                        f"doc_id={cit.get('doc_id', '')}, "
                        f"chunk_id={cit.get('chunk_id', '')}, "
                        f"label={cit.get('label', '')}"
                    )

        # 3. 参考文献汇总（GB/T 7714）
        if document.references:
            doc.add_heading("参考文献", level=1)
            gbt_refs = document.references_gbt7714()
            # 保证顺序
            gbt_refs.sort(key=lambda r: (r.get("index") is None, r.get("index") or 0))
            for item in gbt_refs:
                doc.add_paragraph(item.get("formatted", ""))

        # 4. 输出
        if output_path:
            # 保存到文件
            try:
                docx_dir = os.path.dirname(output_path)
                if docx_dir and not os.path.exists(docx_dir):
                    os.makedirs(docx_dir, exist_ok=True)
                doc.save(output_path)
                size = os.path.getsize(output_path)
                logger.info("Word 文档已保存: %s (%d bytes)", output_path, size)
                return DocxRenderResult(
                    path=output_path,
                    mode="docx",
                    size=size,
                )
            except (OSError, PermissionError) as e:
                logger.warning("文件保存失败，改为返回 bytes: %s", e)
                output_path = None

        # 保存到内存
        buffer = io.BytesIO()
        doc.save(buffer)
        data = buffer.getvalue()
        return DocxRenderResult(
            bytes=data,
            mode="docx",
            size=len(data),
        )


# ---------------------------------------------------------------------------
# Markdown 降级实现（python-docx 不可用时）
# ---------------------------------------------------------------------------


class MarkdownDocxRenderer(BaseDocxRenderer):
    """Markdown 降级渲染（python-docx 不可用时）

    输出标准 Markdown 格式（.md），保证内容可读、可后续转换为 docx。
    """

    def render(
        self,
        document: ReviewDocument,
        output_path: Optional[str] = None,
    ) -> DocxRenderResult:
        lines: List[str] = []

        # 标题
        title = document.title or f"综述：{document.topic}"
        lines.append(f"# {title}")
        lines.append("")

        # 元信息
        lines.append(f"- 主题：{document.topic}")
        lines.append(f"- 生成时间：{document.created_at}")
        stats = document.citation_stats
        if stats:
            lines.append(
                f"- 引用统计：共 {stats.get('total', 0)} 条 "
                f"(支持 {stats.get('entailment', 0)} / "
                f"存疑 {stats.get('contradiction', 0)} / "
                f"过宽 {stats.get('neutral', 0)})"
            )
        lines.append("")

        # 章节内容
        outline_map: Dict[str, Dict] = {
            o.get("section_id", ""): o for o in document.outline
        }
        for para in document.paragraphs:
            section_id = para.get("section_id", "")
            outline = outline_map.get(section_id, {})
            section_title = outline.get("title") or f"章节 {section_id}"

            lines.append(f"## {section_title}")
            lines.append("")

            text = para.get("rendered_text") or para.get("text", "")
            if text:
                lines.append(text)
                lines.append("")

            citations = para.get("citations", [])
            if citations:
                lines.append("**本节引用：**")
                for cit in citations:
                    lines.append(
                        f"- [{cit.get('index', '')}] "
                        f"doc_id={cit.get('doc_id', '')}, "
                        f"chunk_id={cit.get('chunk_id', '')}, "
                        f"label={cit.get('label', '')}"
                    )
                lines.append("")

        # 参考文献（GB/T 7714）
        if document.references:
            lines.append("## 参考文献")
            lines.append("")
            gbt_refs = document.references_gbt7714()
            gbt_refs.sort(key=lambda r: (r.get("index") is None, r.get("index") or 0))
            for item in gbt_refs:
                lines.append(item.get("formatted", ""))
            lines.append("")

        markdown = "\n".join(lines)

        # 输出
        if output_path:
            try:
                md_dir = os.path.dirname(output_path)
                if md_dir and not os.path.exists(md_dir):
                    os.makedirs(md_dir, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                size = os.path.getsize(output_path)
                logger.info("Markdown 文档已保存: %s (%d bytes)", output_path, size)
                return DocxRenderResult(
                    path=output_path,
                    markdown=markdown,
                    mode="markdown",
                    size=size,
                )
            except (OSError, PermissionError) as e:
                logger.warning("文件保存失败，返回内存内容: %s", e)

        return DocxRenderResult(
            markdown=markdown,
            mode="markdown",
            size=len(markdown.encode("utf-8")),
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_docx_renderer(mode: str = "auto", **kwargs) -> BaseDocxRenderer:
    """工厂函数：按模式创建 Word 渲染器

    Args:
        mode: "docx" / "markdown" / "auto"
            - auto：优先 python-docx，未安装降级 Markdown
        **kwargs: 传给具体 renderer 的参数
    Returns:
        BaseDocxRenderer 实例
    """
    mode = mode.lower()

    if mode == "markdown":
        return MarkdownDocxRenderer(**kwargs)

    if mode == "docx":
        return PythonDocxRenderer(**kwargs)

    if mode == "auto":
        renderer = PythonDocxRenderer(**kwargs)
        if renderer.available:
            return renderer
        logger.info("auto 模式：python-docx 不可用，降级为 MarkdownDocxRenderer")
        return MarkdownDocxRenderer()

    raise ValueError(f"未知 docx renderer mode: {mode}")

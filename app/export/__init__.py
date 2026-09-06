"""Word 导出模块（项目文档第 2/3/10 章）

包含：综述 Word 文档渲染（python-docx）+ Markdown 降级。

模块总览（项目文档附录）：
- docx_renderer.py    python-docx 模板渲染
"""

from .docx_renderer import (
    BaseDocxRenderer,
    DocxRenderResult,
    MarkdownDocxRenderer,
    PythonDocxRenderer,
    ReviewDocument,
    create_docx_renderer,
)

__all__ = [
    "BaseDocxRenderer",
    "DocxRenderResult",
    "MarkdownDocxRenderer",
    "PythonDocxRenderer",
    "ReviewDocument",
    "create_docx_renderer",
]

"""PDF 解析模块（API-first 架构，项目文档 4.1 节）

模块总览（项目文档附录）：
- pdf_to_image.py       PyMuPDF 按页转图
- vlm_client.py         Qwen-VL-Max API 调用 + 并发控制 + Redis 缓存
- text_extractor.py     pdfplumber 文本+坐标抽取
- fusion_engine.py      多源融合 + 冲突仲裁（核心）
- formula_binder.py     公式与段落绑定（四级规则策略，非 IoU 对齐）
- fallback_handler.py   置信度<0.7 时切二次验证

⚠️ 坐标对齐漏洞修复说明：
文档附录原注释为"formula_binder.py # 公式与段落坐标 IoU 对齐绑定"，
但文档 4.1 节明确 Qwen-VL-Max 不返回坐标框，不能做 IoU 对齐。
formula_binder.py 已实现为"规则匹配为主、坐标辅助为辅"的四级绑定策略。
"""

from .pdf_to_image import PageImage, PdfToImageConverter, crop_region
from .text_extractor import (
    PageLayout,
    Paragraph,
    PdfplumberExtractor,
    TextBlock,
    build_mock_layout,
)
from .vlm_client import (
    QwenVlmClient,
    VlmFormula,
    VlmPageResult,
    VlmTable,
)
from .formula_binder import (
    BindingRule,
    BoundFormula,
    FormulaBinder,
    binding_stats,
)
from .fallback_handler import FallbackHandler, FallbackResult
from .fusion_engine import (
    FusedDocument,
    FusedPage,
    FusedParagraph,
    FusedTable,
    FusionEngine,
    document_stats,
)
from .paper_structure import (
    CHUNK_ROLES,
    BaseStructurer,
    LabeledChunk,
    LLMStructurer,
    MockStructurer,
    PaperStructure,
    create_structurer,
)

__all__ = [
    # pdf_to_image
    "PageImage",
    "PdfToImageConverter",
    "crop_region",
    # text_extractor
    "PageLayout",
    "Paragraph",
    "PdfplumberExtractor",
    "TextBlock",
    "build_mock_layout",
    # vlm_client
    "QwenVlmClient",
    "VlmFormula",
    "VlmPageResult",
    "VlmTable",
    # formula_binder
    "BindingRule",
    "BoundFormula",
    "FormulaBinder",
    "binding_stats",
    # fallback_handler
    "FallbackHandler",
    "FallbackResult",
    # fusion_engine
    "FusedDocument",
    "FusedPage",
    "FusedParagraph",
    "FusedTable",
    "FusionEngine",
    "document_stats",
    # paper_structure
    "CHUNK_ROLES",
    "BaseStructurer",
    "LabeledChunk",
    "LLMStructurer",
    "MockStructurer",
    "PaperStructure",
    "create_structurer",
]

"""综述生成模块（项目文档 4.3 / 6.5 节）

包含：大纲规划、Map-Reduce 分段生成、引用事实验证、主流程编排。

模块总览（项目文档附录）：
- outline_planner.py    综述大纲规划（按主题分节）
- generator.py          Map-Reduce 分段生成（含段落级缓存）
- citation_verifier.py  NLI 引用事实验证（三色标记）
- pipeline.py           主流程编排（大纲→生成→验证→ReviewDocument）
"""

from .outline_planner import (
    BaseOutlinePlanner,
    LLMOutlinePlanner,
    MockOutlinePlanner,
    OutlinePlan,
    OutlineSection,
    create_outline_planner,
)
from .generator import (
    BaseReviewGenerator,
    Citation,
    GenerationResult,
    LLMReviewGenerator,
    MapSummary,
    MockReviewGenerator,
    Paragraph,
    ParagraphCache,
    create_review_generator,
)
from .citation_verifier import (
    CitationRef,
    CitationVerifier,
    VerifiedCitation,
    VerificationResult,
    parse_citations,
    render_citations_html,
)
from .citation_formatter import GBT7714Formatter
from .pipeline import (
    ReviewPipeline,
    create_review_pipeline,
)

__all__ = [
    # outline_planner
    "BaseOutlinePlanner",
    "LLMOutlinePlanner",
    "MockOutlinePlanner",
    "OutlinePlan",
    "OutlineSection",
    "create_outline_planner",
    # generator
    "BaseReviewGenerator",
    "Citation",
    "GenerationResult",
    "LLMReviewGenerator",
    "MapSummary",
    "MockReviewGenerator",
    "Paragraph",
    "ParagraphCache",
    "create_review_generator",
    # citation_verifier
    "CitationRef",
    "CitationVerifier",
    "VerifiedCitation",
    "VerificationResult",
    "parse_citations",
    "render_citations_html",
    # citation_formatter
    "GBT7714Formatter",
    # pipeline
    "ReviewPipeline",
    "create_review_pipeline",
]

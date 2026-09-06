"""多语种文献翻译 + 专业术语统一校准。

用法：
    from app.translation import create_translator, TermGlossary, TermEntry
    translator = create_translator("auto")
    result = translator.translate_text("Self-attention is the core of Transformer.")
    print(result.target_text, result.aligned_terms)
"""

from .translator import (
    DashScopeTranslator,
    BaseTranslator,
    MockTranslator,
    TermEntry,
    TermGlossary,
    TranslatedPaper,
    TranslationUnit,
    create_translator,
)

__all__ = [
    "DashScopeTranslator",
    "BaseTranslator",
    "MockTranslator",
    "TermEntry",
    "TermGlossary",
    "TranslatedPaper",
    "TranslationUnit",
    "create_translator",
]

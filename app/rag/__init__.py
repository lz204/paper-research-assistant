"""RAG 核心模块

包含：分块、向量化、检索、重排、记忆池、实体抽取、MMR 去重。
本包内的 context_manager 与 entity_extractor 是项目核心亮点（文档 4.2 节）。

模块总览（项目文档附录）：
- chunker.py          层级分块（256 token + overlap 64）
- embedder.py         DashScope text-embedding-v3 API 封装
- retriever.py        混合检索（向量 + BM25 + RRF，pgvector）
- reranker.py         rerank API / LLM NLI 重排序
- context_manager.py  KV 记忆池（核心亮点）
- entity_extractor.py 正则+spaCy 实体抽取
- mmr.py              最大边际相关性去重
"""

from .chunker import (
    BaseChunker,
    Chunk,
    HierarchicalChunker,
    TokenCounter,
)
from .embedder import (
    BaseEmbedder,
    DashScopeEmbedder,
    FakeEmbedder,
)
from .retriever import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    InMemoryVectorRetriever,
    PgvectorRetriever,
    RetrievalResult,
    rrf_fuse,
)
from .reranker import (
    BaseReranker,
    DashScopeReranker,
    LLMNliReranker,
    NoopReranker,
    RerankResult,
    create_reranker,
)
from .context_manager import ContextManager
from .entity_extractor import (
    BaseEntityExtractor,
    Entity,
    EntityType,
    HybridEntityExtractor,
    RegexEntityExtractor,
    SpacyEntityExtractor,
)
from .mmr import (
    MMRItem,
    MMRSelector,
    build_mmr_items,
    cosine_similarity,
    mmr_select,
)

__all__ = [
    # chunker
    "BaseChunker",
    "Chunk",
    "HierarchicalChunker",
    "TokenCounter",
    # embedder
    "BaseEmbedder",
    "DashScopeEmbedder",
    "FakeEmbedder",
    # retriever
    "BaseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "InMemoryVectorRetriever",
    "PgvectorRetriever",
    "RetrievalResult",
    "rrf_fuse",
    # reranker
    "BaseReranker",
    "DashScopeReranker",
    "LLMNliReranker",
    "NoopReranker",
    "RerankResult",
    "create_reranker",
    # context_manager
    "ContextManager",
    # entity_extractor
    "BaseEntityExtractor",
    "Entity",
    "EntityType",
    "HybridEntityExtractor",
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    # mmr
    "MMRItem",
    "MMRSelector",
    "build_mmr_items",
    "cosine_similarity",
    "mmr_select",
]

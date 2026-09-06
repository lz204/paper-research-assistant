"""混合检索模块（项目文档第 3/5 章 + 附录）

文档选型：
- pgvector（HNSW 查询 < 50ms，5 万级向量够用）
- 混合检索：向量 + BM25 + RRF 融合

为什么不用 Milvus（文档第 3 章）：
- pgvector 复用 PG，省中间件
- 元数据和向量在同一 PG 事务内提交，避免数据不一致
- Milvus 要带 Etcd + MinIO，吃 4GB+ 内存，8GB 机器扛不住

本模块设计：
- BaseRetriever 抽象基类
- InMemoryVectorRetriever：内存版兜底（无 PG 时可用，本地联调）
- BM25Retriever：基于 rank_bm25 的关键词检索（可选依赖）
- HybridRetriever：向量 + BM25 + RRF 融合（项目默认）
- PgvectorRetriever：生产环境 pgvector 实现（接口预留，依赖 PG）

RRF 算法（Reciprocal Rank Fusion）：
    score(d) = Σ 1 / (k + rank_i(d))
    其中 k=60 是经验值，rank_i(d) 是文档 d 在第 i 个检索器中的排名
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunker import Chunk
from .embedder import BaseEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """单个检索结果

    Attributes:
        chunk: 检索到的 chunk
        score: 综合相关性分数（越高越相关）
        source: 检索来源（"vector" / "bm25" / "hybrid"）
        rank: 在最终结果列表中的排名（0 开始）
    """

    chunk: Chunk
    score: float
    source: str = "hybrid"
    rank: int = 0


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """检索器抽象基类"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """索引 chunk 列表（入库阶段调用）"""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """检索

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
        Returns:
            RetrievalResult 列表，按 score 降序
        """
        ...


# ---------------------------------------------------------------------------
# RRF 融合算法
# ---------------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion 融合多个检索器的排名

    Args:
        ranked_lists: 每个 retrieval 的 chunk_id 排序列表
        k: RRF 平滑参数（经验值 60）
        weights: 每个检索器的权重（默认等权）
    Returns:
        {chunk_id: rrf_score}
    """
    if not ranked_lists:
        return {}

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights 长度 {len(weights)} 与 ranked_lists 长度 {len(ranked_lists)} 不匹配"
        )

    scores: Dict[str, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked):
            # rank 从 0 开始，RRF 公式用 rank+1
            scores[chunk_id] += w / (k + rank + 1)

    return dict(scores)


# ---------------------------------------------------------------------------
# 内存向量检索器（本地兜底，无 PG 依赖）
# ---------------------------------------------------------------------------


class InMemoryVectorRetriever(BaseRetriever):
    """内存版向量检索器

    用 numpy 做暴力余弦相似度检索，适用于：
    - 本地开发联调（8GB CPU 笔记本）
    - 小规模数据（< 1 万 chunk）
    - 单元测试

    生产环境应替换为 PgvectorRetriever（HNSW 索引，5 万级向量 < 50ms）
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._chunks: List[Chunk] = []
        self._vectors: List[np.ndarray] = []
        self._id_to_idx: Dict[str, int] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """向量化并索引 chunks"""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vecs, failed = self.embedder.embed(texts)

        added = 0
        for chunk, vec in zip(chunks, vecs):
            if vec is None:
                logger.warning(
                    "chunk %s embedding 失败，跳过索引（待重试）",
                    chunk.chunk_id,
                )
                continue
            self._id_to_idx[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            self._vectors.append(vec)
            added += 1

        logger.info(
            "InMemoryVectorRetriever.add_chunks: total=%d, added=%d, failed=%d",
            len(chunks), added, len(chunks) - added,
        )

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._chunks:
            return []

        query_vec = self.embedder.embed_one(query)
        if query_vec is None:
            logger.warning("query embedding 失败，返回空结果")
            return []

        # 暴力余弦相似度
        scores = self._cosine_all(query_vec)
        # 取 top_k
        top_indices = np.argsort(-scores)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    source="vector",
                    rank=rank,
                )
            )
        return results

    def _cosine_all(self, query_vec: np.ndarray) -> np.ndarray:
        """计算 query 与所有已索引向量的余弦相似度"""
        if not self._vectors:
            return np.array([])
        mat = np.vstack(self._vectors)  # (n, d)
        q = query_vec.reshape(1, -1)    # (1, d)

        # L2 归一化
        mat_norm = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_norm_safe = np.where(mat_norm == 0, 1.0, mat_norm)
        mat_normed = mat / mat_norm_safe

        q_norm = float(np.linalg.norm(q))
        q_normed = q / q_norm if q_norm > 0 else q

        return (mat_normed @ q_normed.T).flatten()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """按 chunk_id 获取 chunk（用于引用溯源二次检索）"""
        idx = self._id_to_idx.get(chunk_id)
        if idx is None:
            return None
        return self._chunks[idx]

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# BM25 检索器（关键词检索）
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
    """BM25 关键词检索器

    依赖 rank_bm25 库（pip install rank_bm25），未安装时降级为简单词频匹配。
    用于与向量检索互补——向量擅长语义，BM25 擅长精确关键词（数据集名、模型名、术语）。
    """

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: 自定义分词函数，默认用空格 + 简单中文切分
        """
        self.tokenizer = tokenizer or self._default_tokenizer
        self._chunks: List[Chunk] = []
        self._id_to_idx: Dict[str, int] = {}
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """简单分词：英文按空格+标点，中文按字符"""
        import re
        # 英文 token
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
        # 中文字符
        tokens.extend(c for c in text if "\u4e00" <= c <= "\u9fff")
        return tokens

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            self._bm25_cls = BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 未安装，BM25Retriever 降级为简单词频匹配。"
                "pip install rank_bm25 后可启用完整 BM25"
            )
            self._bm25_cls = None

        for chunk in chunks:
            self._id_to_idx[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            self._tokenized_corpus.append(self.tokenizer(chunk.text))

        if self._bm25_cls is not None and self._tokenized_corpus:
            self._bm25 = self._bm25_cls(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._chunks:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            # rank_bm25 的 BM25Okapi 在小语料下（文档数少）idf 会退化为 0，
            # 导致所有文档分数为 0、关键词召回完全失效。此时回退为简单词频匹配，
            # 保证 BM25 关键词路径在任意语料规模下都有确定性召回。
            if not np.any(scores > 0):
                logger.debug(
                    "BM25Okapi 全 0 分（语料过小 idf≈0），回退 Jaccard 词频匹配"
                )
                scores = self._simple_tf_scores(query_tokens)
        else:
            # 降级：简单词频匹配
            scores = self._simple_tf_scores(query_tokens)

        # 取 top_k
        top_indices = np.argsort(-scores)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    source="bm25",
                    rank=rank,
                )
            )
        return results

    def _simple_tf_scores(self, query_tokens: List[str]) -> np.ndarray:
        """降级模式：简单词频匹配（rank_bm25 不可用时）"""
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        query_set = set(query_tokens)
        for i, doc_tokens in enumerate(self._tokenized_corpus):
            doc_set = set(doc_tokens)
            # Jaccard 相似度作为分数
            if not query_set or not doc_set:
                continue
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)
            scores[i] = intersection / union if union > 0 else 0.0
        return scores

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._id_to_idx.get(chunk_id)
        if idx is None:
            return None
        return self._chunks[idx]

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# 混合检索器（向量 + BM25 + RRF，项目默认）
# ---------------------------------------------------------------------------


class HybridRetriever(BaseRetriever):
    """混合检索器：向量 + BM25 + RRF 融合（项目文档默认实现）

    工作流程：
    1. 向量检索器返回 top_n 个语义相关 chunk
    2. BM25 检索器返回 top_n 个关键词相关 chunk
    3. RRF 融合两个排名列表
    4. （可选）reranker 重排融合后的 top_k

    文档第 3 章架构图：User → RAG 检索 → 记忆池 → LLM
    本类对应"RAG 检索"环节

    Args:
        embedder: 向量化器
        use_bm25: 是否启用 BM25（默认 True）
        reranker: 可选重排器，None 表示不重排
        vector_weight: 向量检索在 RRF 中的权重
        bm25_weight: BM25 检索在 RRF 中的权重
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        use_bm25: bool = True,
        reranker=None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.vector_retriever = InMemoryVectorRetriever(embedder)
        self.use_bm25 = use_bm25
        self.bm25_retriever: Optional[BM25Retriever] = (
            BM25Retriever() if use_bm25 else None
        )
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """索引 chunks（向量 + BM25 双路入库）"""
        self.vector_retriever.add_chunks(chunks)
        if self.bm25_retriever is not None:
            self.bm25_retriever.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5, top_n: int = 20) -> List[RetrievalResult]:
        """混合检索

        Args:
            query: 查询文本
            top_k: 最终返回数量
            top_n: 每路检索器各自取的候选数（融合后再裁剪到 top_k）
        Returns:
            RetrievalResult 列表，按综合分数降序
        """
        if len(self.vector_retriever) == 0 and not (self.bm25_retriever and len(self.bm25_retriever) > 0):
            return []

        # 1. 向量检索
        vec_results = self.vector_retriever.search(query, top_k=top_n)
        vec_ranked = [r.chunk.chunk_id for r in vec_results]
        vec_score_map = {r.chunk.chunk_id: r.score for r in vec_results}

        # 2. BM25 检索
        bm25_ranked: List[str] = []
        bm25_score_map: Dict[str, float] = {}
        bm25_results: List[RetrievalResult] = []
        if self.bm25_retriever is not None and len(self.bm25_retriever) > 0:
            bm25_results = self.bm25_retriever.search(query, top_k=top_n)
            bm25_ranked = [r.chunk.chunk_id for r in bm25_results]
            bm25_score_map = {r.chunk.chunk_id: r.score for r in bm25_results}

        # 3. RRF 融合
        ranked_lists = [vec_ranked]
        weights = [self.vector_weight]
        if self.bm25_retriever is not None and bm25_ranked:
            ranked_lists.append(bm25_ranked)
            weights.append(self.bm25_weight)

        rrf_scores = rrf_fuse(ranked_lists, k=60, weights=weights)

        # 4. 构造融合结果
        all_chunks: Dict[str, Chunk] = {}
        for r in vec_results:
            all_chunks[r.chunk.chunk_id] = r.chunk
        for r in bm25_results:
            if r.chunk.chunk_id not in all_chunks:
                all_chunks[r.chunk.chunk_id] = r.chunk

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_ids = sorted_ids[:top_k]

        results: List[RetrievalResult] = []
        for rank, (cid, score) in enumerate(sorted_ids):
            chunk = all_chunks.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    source="hybrid",
                    rank=rank,
                )
            )

        # 5. 可选 reranker 重排
        if self.reranker is not None and results:
            chunks_for_rerank = [(r.chunk.chunk_id, r.chunk.text) for r in results]
            rerank_results = self.reranker.rerank(query, chunks_for_rerank, top_k=top_k)

            # 用 rerank 分数更新 RetrievalResult
            rerank_map = {r.chunk_id: r for r in rerank_results}
            new_results: List[RetrievalResult] = []
            for r in results:
                if r.chunk.chunk_id in rerank_map:
                    r.score = rerank_map[r.chunk.chunk_id].score
                    r.source = f"hybrid+rerank({rerank_map[r.chunk.chunk_id].label})"
                new_results.append(r)
            new_results.sort(key=lambda x: x.score, reverse=True)
            for i, r in enumerate(new_results):
                r.rank = i
            results = new_results[:top_k]

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """按 chunk_id 获取 chunk（用于引用溯源二次检索，文档 6.5 节）"""
        chunk = self.vector_retriever.get_chunk_by_id(chunk_id)
        if chunk is not None:
            return chunk
        if self.bm25_retriever is not None:
            return self.bm25_retriever.get_chunk_by_id(chunk_id)
        return None

    def __len__(self) -> int:
        return len(self.vector_retriever)


# ---------------------------------------------------------------------------
# PgvectorRetriever 接口预留（生产环境用）
# ---------------------------------------------------------------------------


class PgvectorRetriever(BaseRetriever):
    """pgvector 检索器（生产环境用，依赖 PostgreSQL + pgvector 扩展）

    本类为接口预留，实际部署时需要：
    1. 安装 pgvector 扩展：CREATE EXTENSION vector;
    2. 建表：CREATE TABLE chunks (id TEXT, doc_id TEXT, section TEXT,
              text TEXT, embedding VECTOR(1024), metadata JSONB);
    3. 建 HNSW 索引：CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

    文档第 3 章：pgvector 复用 PG，元数据和向量在同一事务内提交，避免数据不一致

    注意：本地开发用 InMemoryVectorRetriever 即可，无需启动 PG
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        dsn: str = "postgresql://postgres:postgres@localhost:5432/pra",
        table_name: str = "chunks",
    ):
        self.embedder = embedder
        self.dsn = dsn
        self.table_name = table_name
        self._conn = None
        logger.warning(
            "PgvectorRetriever 为接口预留，本地开发请用 InMemoryVectorRetriever"
        )

    def _ensure_conn(self) -> bool:
        """懒加载 PG 连接"""
        if self._conn is not None:
            return True
        try:
            import psycopg2  # type: ignore
            self._conn = psycopg2.connect(self.dsn)
            return True
        except ImportError:
            logger.warning("psycopg2 未安装，PgvectorRetriever 不可用")
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning("PG 连接失败: %s", e)
            return False

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not self._ensure_conn():
            logger.warning("PG 不可用，跳过索引（待 PG 恢复后重试）")
            return

        texts = [c.text for c in chunks]
        vecs, failed = self.embedder.embed(texts)

        with self._conn.cursor() as cur:
            for chunk, vec in zip(chunks, vecs):
                if vec is None:
                    continue
                # 文档第 3 章：元数据和向量在同一事务内提交
                cur.execute(
                    f"INSERT INTO {self.table_name} "
                    f"(id, doc_id, section, text, embedding, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s) "
                    f"ON CONFLICT (id) DO NOTHING",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.section,
                        chunk.text,
                        vec.tolist(),
                        chunk.metadata,
                    ),
                )
        self._conn.commit()

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._ensure_conn():
            return []

        query_vec = self.embedder.embed_one(query)
        if query_vec is None:
            return []

        with self._conn.cursor() as cur:
            # HNSW 余弦相似度检索（文档第 3 章：< 50ms for 5 万级向量）
            cur.execute(
                f"SELECT id, doc_id, section, text, metadata, "
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table_name} "
                f"ORDER BY embedding <=> %s::vector "
                f"LIMIT %s",
                (query_vec.tolist(), query_vec.tolist(), top_k),
            )
            rows = cur.fetchall()

        results: List[RetrievalResult] = []
        for rank, row in enumerate(rows):
            chunk = Chunk(
                chunk_id=row[0],
                text=row[3],
                token_count=0,  # 从 PG 取出时不重新计数
                doc_id=row[1],
                section=row[2] or "",
                metadata=row[4] or {},
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(row[5]),
                    source="pgvector",
                    rank=rank,
                )
            )
        return results

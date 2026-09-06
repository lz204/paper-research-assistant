"""Embedding 模块（项目文档第 3/5/6 章）

选型（文档第 5 章）：
- DashScope text-embedding-v3 API（替代本地 bge-large-zh）
- 理由：bge 要 2GB 显存，纯 CPU 跑不动；API 中文论文术语召回与 bge 持平，不占本地资源
- 代价：依赖外部 API

成本治理（文档 6.2 节）：
- 文本去重后向量化 + 结果缓存
- 相同 chunk 不重复 embedding
- 缓存命中率约 60%（同论文二次打开免费）

容错（文档 6.3 节）：
- Embedding API 失败时标记 chunk 待重试，不阻塞入库
- 指数退避重试
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Embedding 抽象基类

    设计目的：支持 embedder 热插拔
    （DashScope API / 本地 bge / 哈希 fake 都可替换）
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """批量向量化

        Args:
            texts: 文本列表
        Returns:
            (vectors, failed_indices)
            - vectors: 与 texts 等长的向量列表，失败的对应位置为 None
            - failed_indices: 失败的文本下标列表（用于后续重试）
        """
        ...

    def embed_one(self, text: str) -> Optional[np.ndarray]:
        """单文本向量化（便捷方法）"""
        vecs, failed = self.embed([text])
        if failed:
            return None
        return vecs[0]


# ---------------------------------------------------------------------------
# DashScope Embedding 实现
# ---------------------------------------------------------------------------


class DashScopeEmbedder(BaseEmbedder):
    """DashScope text-embedding-v3 API 封装

    成本治理（文档 6.2 节）：
    1. 文本去重：相同文本只 embedding 一次
    2. 结果缓存：内存 LRU（可扩展到 Redis）
    3. 批量请求：一次 API 调用处理多条文本

    容错（文档 6.3 节）：
    1. 指数退避重试（max_retry 次）
    2. 最终失败的文本返回 None + failed_indices，不阻塞入库

    用法：
        embedder = DashScopeEmbedder(api_key="sk-xxx")
        vecs, failed = embedder.embed(["hello", "world"])
    """

    # DashScope text-embedding-v3 默认维度 1024
    DEFAULT_DIM = 1024
    # API 单次请求最大文本数：DashScope 明确限制 batch <= 10
    # （此前 25 会触发 400 "batch size ... larger than 10"）
    BATCH_SIZE = 10
    # API 单次请求字符数预算（text-embedding-v3 有字符上限，
    # 长文本下按字符预算再拆分，避免单请求超限）
    BATCH_CHAR_BUDGET = 4000
    # 重试次数（文档 6.3：指数退避重试 2 次）
    MAX_RETRY = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-v3",
        dim: int = DEFAULT_DIM,
        cache_size: int = 10000,
    ):
        """
        Args:
            api_key: DashScope API key，默认从 DASHSCOPE_API_KEY 环境变量读取
            model: 模型名，默认 text-embedding-v3
            dim: 向量维度
            cache_size: 内存缓存最大条目数
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self._dim = dim
        # 内存缓存：{text_md5: vector}
        # 生产环境可替换为 Redis（key: embed:{model}:{md5}）
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._client = None  # lazy init

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_client(self) -> bool:
        """懒加载 DashScope client"""
        if self._client is not None:
            return True
        if not self.api_key:
            logger.warning(
                "DASHSCOPE_API_KEY 未配置，DashScopeEmbedder 不可用。"
                "请配置环境变量或使用 FakeEmbedder 做本地测试"
            )
            return False
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning(
                "dashscope 包未安装，pip install dashscope 后可启用。"
                "本地测试可用 FakeEmbedder 替代"
            )
            return False

    def embed(self, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        if not texts:
            return [], []

        # 1. 文本去重 + 缓存命中检查（文档 6.2 成本治理）
        text_hashes = [self._hash_text(t) for t in texts]
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        to_fetch: Dict[str, str] = {}  # {hash: text}，待 API 获取

        for i, h in enumerate(text_hashes):
            if h in self._cache:
                results[i] = self._cache[h].copy()
            elif texts[i].strip():
                to_fetch[h] = texts[i]
            else:
                # 空文本直接返回零向量
                results[i] = np.zeros(self._dim, dtype=np.float32)

        if not to_fetch:
            return results, []

        # 2. 检查 API 可用性
        if not self._ensure_client():
            # API 不可用：所有待获取的标记为失败
            failed = [i for i, h in enumerate(text_hashes) if h in to_fetch]
            return results, failed

        # 3. 批量调用 API（按条数 + 字符预算自适应分包）
        all_failed_hashes: set = set()
        hash_to_vec: Dict[str, np.ndarray] = {}

        fetch_items = list(to_fetch.items())
        batches = self._split_batches(fetch_items)
        for batch in batches:
            batch_hashes = [h for h, _ in batch]
            batch_texts = [t for _, t in batch]

            vecs = self._call_api_with_retry(batch_texts)
            if vecs is None:
                # 整批失败：降级为逐条单发（单条长文本成功率更高，避免整批丢弃）
                for h, t in zip(batch_hashes, batch_texts):
                    single_vecs = self._call_api_with_retry([t])
                    if single_vecs and single_vecs[0] is not None:
                        hash_to_vec[h] = single_vecs[0]
                        self._put_cache(h, single_vecs[0])
                    else:
                        all_failed_hashes.add(h)
                continue

            for h, v in zip(batch_hashes, vecs):
                if v is not None:
                    hash_to_vec[h] = v
                    self._put_cache(h, v)

        # 4. 回填结果
        failed_indices: List[int] = []
        for i, h in enumerate(text_hashes):
            if results[i] is not None:
                continue
            if h in hash_to_vec:
                results[i] = hash_to_vec[h].copy()
            else:
                failed_indices.append(i)

        if failed_indices:
            logger.warning(
                "embedding 部分失败：total=%d, failed=%d（不阻塞入库，待重试）",
                len(texts), len(failed_indices),
            )

        return results, failed_indices

    def _split_batches(self, fetch_items: List[tuple]) -> List[List[tuple]]:
        """按条数上限 + 字符预算把 (hash, text) 拆成多个批次

        DashScope text-embedding-v3 单请求有字符上限，长文本（如论文 chunk）
        下固定 25 条/批可能超限被服务端拒绝。这里同时约束条数与累计字符数，
        并用贪心回填减少批次数。
        """
        batches: List[List[tuple]] = []
        current: List[tuple] = []
        current_chars = 0

        for item in fetch_items:
            chars = len(item[1])
            # 单条超预算：单独一批（服务端若仍拒绝由调用方逐条降级）
            if chars > self.BATCH_CHAR_BUDGET:
                if current:
                    batches.append(current)
                    current, current_chars = [], 0
                batches.append([item])
                continue
            if (len(current) >= self.BATCH_SIZE
                    or current_chars + chars > self.BATCH_CHAR_BUDGET):
                batches.append(current)
                current, current_chars = [], 0
            current.append(item)
            current_chars += chars

        if current:
            batches.append(current)
        return batches

    def _call_api_with_retry(self, texts: List[str]) -> Optional[List[Optional[np.ndarray]]]:
        """调用 DashScope API，指数退避重试

        Returns:
            向量列表（每项可能为 None）；整批失败返回 None
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRY + 1):
            try:
                resp = self._client.TextEmbedding.call(
                    model=self.model,
                    input=texts,
                )
                # dashscope SDK 的 status_code 可能为 int 或可比较对象
                status = getattr(resp, "status_code", None)
                code = getattr(resp, "code", None)
                if status is not None and status != 200:
                    raise RuntimeError(
                        f"DashScope embedding API 错误: code={code}, "
                        f"msg={getattr(resp, 'message', '')}"
                    )
                # 兼容 dict / 对象两种返回：
                #   resp.output.embeddings   （旧对象形态）
                #   resp.output["embeddings"]（新版 dict 形态）
                output = getattr(resp, "output", None)
                if isinstance(output, dict):
                    emb_items = output.get("embeddings", [])
                else:
                    emb_items = getattr(output, "embeddings", []) or []
                if not emb_items:
                    raise RuntimeError(
                        f"DashScope embedding API 返回为空（code={code}）"
                    )
                # emb_items 是 list of {"text_index": int, "embedding": list}
                vecs: List[Optional[np.ndarray]] = [None] * len(texts)
                for item in emb_items:
                    if isinstance(item, dict):
                        idx = item["text_index"]
                        vecs[idx] = np.asarray(item["embedding"], dtype=np.float32)
                    else:  # 对象形态
                        idx = item.text_index
                        vecs[idx] = np.asarray(item.embedding, dtype=np.float32)
                return vecs
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        "embedding API 失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.MAX_RETRY + 1, wait, e,
                    )
                    time.sleep(wait)
        logger.error("embedding API 最终失败: %s", last_error)
        return None

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _put_cache(self, h: str, v: np.ndarray) -> None:
        """写入缓存，超出容量时简单清空一半（生产环境可换 LRU）"""
        if len(self._cache) >= self._cache_size:
            # 简化处理：清空一半最旧的（按 dict 插入顺序）
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]
        self._cache[h] = v

    def cache_stats(self) -> Dict:
        """缓存命中率统计（对接文档 6.1 可观测性看板）"""
        return {
            "cache_size": len(self._cache),
            "cache_capacity": self._cache_size,
        }


# ---------------------------------------------------------------------------
# FakeEmbedder（本地测试用，无 API 依赖）
# ---------------------------------------------------------------------------


class FakeEmbedder(BaseEmbedder):
    """哈希向量 embedder，本地测试用

    用确定性哈希把文本映射到固定维度的向量，
    保证相同文本得到相同向量（语义无关，仅用于流程联调）。
    在 8GB CPU 笔记本上不依赖任何 API 即可跑通 RAG 全链路。
    """

    def __init__(self, dim: int = 64, seed: int = 42):
        self._dim = dim
        self.seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        results: List[np.ndarray] = []
        for t in texts:
            results.append(self._hash_to_vec(t))
        return results, []

    def _hash_to_vec(self, text: str) -> np.ndarray:
        """确定性哈希：相同文本 → 相同向量"""
        # 用文本 hash 做 seed，生成确定性向量
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16) ^ self.seed
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        # L2 归一化，方便后续余弦相似度
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

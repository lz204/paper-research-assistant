"""重排模块（项目文档 4.3 节 + 第 5 章选型）

文档方案 A（推荐，省一次调用）：
- 直接用 Qwen-Plus 做零样本 NLI
- prompt 让模型判断"query vs chunk"是 entailment / contradiction / neutral
- 一次 LLM 调用判断多个 chunk，成本可控

文档方案 B（精度更高）：
- 用 DashScope rerank API 对 "query vs chunk" 打分
- 分数 < 0.3 判 neutral，> 0.7 判 entailment，中间区间切 LLM 二次判断

选型（文档第 5 章）：
- Rerank/NLI: Qwen-Plus 零样本 NLI / DashScope rerank API

容错：API 不可用时降级为 NoopReranker（保持原顺序），不阻塞主流程
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    """单个 chunk 的重排结果

    Attributes:
        chunk_id: chunk 唯一标识
        score: 重排分数 [0, 1]，越高越相关
        label: NLI 标签（entailment / contradiction / neutral），仅 LLM NLI 模式有
        original_rank: 原始排名（0 开始）
    """

    chunk_id: str
    score: float
    label: str = ""
    original_rank: int = 0


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """重排器抽象基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """对候选 chunks 重排

        Args:
            query: 用户查询
            chunks: [(chunk_id, chunk_text), ...] 候选列表
            top_k: 返回前 k 个
        Returns:
            RerankResult 列表，按 score 降序
        """
        ...


# ---------------------------------------------------------------------------
# NoopReranker：不重排（兜底，API 不可用时降级）
# ---------------------------------------------------------------------------


class NoopReranker(BaseReranker):
    """不重排，按原顺序返回

    作为 DashScope/LLM 不可用时的降级实现（文档 6.3 容错思想）。
    """

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        results = [
            RerankResult(
                chunk_id=cid,
                score=1.0 - i * 0.01,  # 简单递减分数
                label="",
                original_rank=i,
            )
            for i, (cid, _) in enumerate(chunks)
        ]
        return results[:top_k]


# ---------------------------------------------------------------------------
# DashScope Rerank API 实现（文档方案 B）
# ---------------------------------------------------------------------------


class DashScopeReranker(BaseReranker):
    """DashScope rerank API 封装（文档方案 B，精度更高）

    评分规则（文档 4.3 节）：
    - score > 0.7 → entailment
    - score < 0.3 → neutral
    - 0.3 ~ 0.7 → 中间区间（实际场景可再切 LLM 二次判断，这里简化为 neutral）

    用法：
        reranker = DashScopeReranker(api_key="sk-xxx")
        results = reranker.rerank(query, [(cid, text), ...], top_k=5)
    """

    MAX_RETRY = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gte-rerank",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self._client = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未配置，DashScopeReranker 不可用")
            return False
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning("dashscope 包未安装，pip install dashscope 后可启用")
            return False

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        if not chunks:
            return []

        if not self._ensure_client():
            # 降级为 NoopReranker
            logger.warning("DashScopeReranker 不可用，降级为 NoopReranker")
            return NoopReranker().rerank(query, chunks, top_k)

        # 调用 API
        texts = [t for _, t in chunks]
        ids = [cid for cid, _ in chunks]
        scores = self._call_api_with_retry(query, texts)

        if scores is None:
            logger.warning("rerank API 失败，降级为 NoopReranker")
            return NoopReranker().rerank(query, chunks, top_k)

        # 构造结果 + NLI 标签映射
        results: List[RerankResult] = []
        for i, (cid, score) in enumerate(zip(ids, scores)):
            label = self._score_to_label(score)
            results.append(
                RerankResult(
                    chunk_id=cid,
                    score=score,
                    label=label,
                    original_rank=i,
                )
            )

        # 按 score 降序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _call_api_with_retry(
        self,
        query: str,
        texts: List[str],
    ) -> Optional[List[float]]:
        """调用 DashScope rerank API，指数退避重试"""
        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRY + 1):
            try:
                resp = self._client.TextReRank.call(
                    model=self.model,
                    query=query,
                    documents=texts,
                    top_n=len(texts),
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DashScope rerank API 错误: code={resp.code}, msg={resp.message}"
                    )
                # resp.output.results 是 list of {"index": int, "relevance_score": float}
                scores = [0.0] * len(texts)
                for item in resp.output.results:
                    idx = item["index"]
                    scores[idx] = float(item["relevance_score"])
                return scores
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt
                    logger.warning(
                        "rerank API 失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.MAX_RETRY + 1, wait, e,
                    )
                    time.sleep(wait)
        logger.error("rerank API 最终失败: %s", last_error)
        return None

    @staticmethod
    def _score_to_label(score: float) -> str:
        """分数 → NLI 标签（文档 4.3 节规则）"""
        if score > 0.7:
            return "entailment"
        if score < 0.3:
            return "neutral"
        return "neutral"  # 中间区间简化为 neutral


# ---------------------------------------------------------------------------
# LLM NLI Reranker（文档方案 A，推荐）
# ---------------------------------------------------------------------------


class LLMNliReranker(BaseReranker):
    """用 Qwen-Plus 做零样本 NLI 重排（文档方案 A，省一次调用）

    工作流程：
    1. 把 query + 多个 chunk 拼成一个 prompt
    2. LLM 一次性返回每个 chunk 的 NLI 标签 + 相关性分数
    3. 按 entailment > neutral > contradiction 排序

    优点（文档 4.3 节）：一次 LLM 调用判断多个引用，成本可控

    用法：
        reranker = LLMNliReranker(api_key="sk-xxx")
        results = reranker.rerank(query, [(cid, text), ...], top_k=5)
    """

    MAX_RETRY = 2
    # 单次 NLI 判断的最大 chunk 数（控制 prompt 长度）
    BATCH_SIZE = 10

    # NLI 标签 → 数值分数（用于排序）
    LABEL_SCORE: Dict[str, float] = {
        "entailment": 0.9,
        "neutral": 0.5,
        "contradiction": 0.1,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self._client = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未配置，LLMNliReranker 不可用")
            return False
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning("dashscope 包未安装，pip install dashscope 后可启用")
            return False

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        if not chunks:
            return []

        if not self._ensure_client():
            logger.warning("LLMNliReranker 不可用，降级为 NoopReranker")
            return NoopReranker().rerank(query, chunks, top_k)

        # 分批调用 LLM（BATCH_SIZE 一批）
        all_results: List[RerankResult] = []
        for batch_start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[batch_start: batch_start + self.BATCH_SIZE]
            batch_results = self._rerank_batch(query, batch, batch_start)
            all_results.extend(batch_results)

        # 按 score 降序
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def _rerank_batch(
        self,
        query: str,
        batch: List[Tuple[str, str]],
        base_rank: int,
    ) -> List[RerankResult]:
        """对一批 chunks 调用 LLM 做 NLI 判断"""
        prompt = self._build_prompt(query, batch)
        response_text = self._call_llm_with_retry(prompt)

        if response_text is None:
            # LLM 失败：本批降级为 NoopReranker
            return [
                RerankResult(
                    chunk_id=cid,
                    score=0.5,
                    label="neutral",
                    original_rank=base_rank + i,
                )
                for i, (cid, _) in enumerate(batch)
            ]

        # 解析 LLM 输出
        labels = self._parse_response(response_text, len(batch))

        results: List[RerankResult] = []
        for i, (cid, _) in enumerate(batch):
            label = labels[i] if i < len(labels) else "neutral"
            score = self.LABEL_SCORE.get(label, 0.5)
            results.append(
                RerankResult(
                    chunk_id=cid,
                    score=score,
                    label=label,
                    original_rank=base_rank + i,
                )
            )
        return results

    def _build_prompt(self, query: str, batch: List[Tuple[str, str]]) -> str:
        """构造 NLI 判断 prompt"""
        lines = [
            "你是一个严格的 NLI（自然语言推理）判断器。",
            "对于给定的查询和若干文本片段，判断每段文本是否能推出查询所述内容。",
            "只返回 JSON 数组，每个元素是 {\"index\": int, \"label\": \"entailment|contradiction|neutral\"}。",
            "",
            f"查询：{query}",
            "",
            "文本片段：",
        ]
        for i, (_, text) in enumerate(batch):
            # 截断过长 chunk，控制 prompt 长度
            preview = text[:300] + ("..." if len(text) > 300 else "")
            lines.append(f"[{i}] {preview}")
        lines.append("")
        lines.append("返回 JSON 数组：")
        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """调用 Qwen-Plus LLM，指数退避重试"""
        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRY + 1):
            try:
                resp = self._client.Generation.call(
                    model=self.model,
                    prompt=prompt,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"LLM API 错误: code={resp.code}, msg={resp.message}"
                    )
                # DashScope Generation 返回 resp.output.text
                return resp.output.text
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM NLI 失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.MAX_RETRY + 1, wait, e,
                    )
                    time.sleep(wait)
        logger.error("LLM NLI 最终失败: %s", last_error)
        return None

    @staticmethod
    def _parse_response(text: str, expected_count: int) -> List[str]:
        """解析 LLM 返回的 JSON 数组，提取 labels"""
        labels: List[str] = []
        try:
            # LLM 输出可能含 markdown 代码块，提取 JSON 部分
            text = text.strip()
            if text.startswith("```"):
                # 去掉 ```json ... ``` 包裹
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "label" in item:
                        label = str(item["label"]).lower()
                        if label in ("entailment", "contradiction", "neutral"):
                            labels.append(label)
                        else:
                            labels.append("neutral")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("LLM NLI 响应解析失败: %s, 原始: %s", e, text[:200])

        # 补齐到 expected_count
        while len(labels) < expected_count:
            labels.append("neutral")
        return labels


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_reranker(mode: str = "auto", **kwargs) -> BaseReranker:
    """工厂函数：按模式创建 reranker

    Args:
        mode: "dashscope" / "llm_nli" / "noop" / "auto"
            - auto：优先 LLM NLI，不可用降级 DashScope，再不可用降级 Noop
        **kwargs: 传给具体 reranker 的参数
    Returns:
        BaseReranker 实例
    """
    mode = mode.lower()

    if mode == "noop":
        return NoopReranker()

    if mode == "dashscope":
        return DashScopeReranker(**kwargs)

    if mode == "llm_nli":
        return LLMNliReranker(**kwargs)

    if mode == "auto":
        # 优先 LLM NLI（文档推荐方案 A）
        reranker = LLMNliReranker(**kwargs)
        if reranker._ensure_client():
            return reranker
        # 降级 DashScope rerank（方案 B）
        reranker = DashScopeReranker(**kwargs)
        if reranker._ensure_client():
            return reranker
        # 最终降级 Noop
        logger.warning("所有 reranker 不可用，降级为 NoopReranker")
        return NoopReranker()

    raise ValueError(f"未知 reranker mode: {mode}")

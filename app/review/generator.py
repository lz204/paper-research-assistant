"""综述 Map-Reduce 分段生成模块（项目文档 4.3 节 + 6.2/6.3 章）

文档 4.3 节"综述生成 + MMR 去重"流程：
- 初版问题：Map-Reduce 直接生成，结果"车轱辘话"——10 篇文献 6 篇观点相似
- 优化一：MMR 去重（在 Reduce 阶段选片段时惩罚已选片段相似度）
  效果：重复占比 38% → 9%，lambda_=0.7
- 优化二：引用事实现证（防幻觉，见 citation_verifier.py）

文档 4.3 节 Map-Reduce：
- Map 阶段：逐文献抽要点（key points + 引用 [doc_id, chunk_id]）
- Reduce 阶段：合并去重（MMR 选片段）→ LLM 按章节生成段落

文档 6.2 节成本治理：
- 综述生成 | 段落级缓存，同主题同文献组可复用 | 综述重跑成本降 40%

文档 6.3 节容错与降级：
- 综述生成是分段流式的，每段生成完立即写缓存
- LLM 中断时返回已生成段落 + 提示"段落 X-Y 因网络问题未完成，可点击重试"
- 关键思想："部分结果好过无结果"

本模块设计：
- MapSummary / Paragraph / GenerationResult dataclass
- BaseReviewGenerator ABC：支持 LLM / Mock 热插拔
- LLMReviewGenerator：Qwen-Plus Map-Reduce + 段落级缓存 + 中断恢复
- MockReviewGenerator：无 API key 时降级，从 chunks 拼接模板段落
- 复用 app.rag.mmr（MMR 去重）和 app.rag.embedder（向量化 map summaries）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Reduce 阶段 MMR 保留的最大要点数（控制段落长度；文档 4.3 节）
# 模块级常量供 /review 响应 dedup_stats.reduce_top_k 直接透出，
# 生成器类通过 _MMRReduceMixin.REDUCE_TOP_K 使用同一默认值（测试可临时调小）。
REDUCE_TOP_K = 8


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """LLM 生成的引用标记（文档 6.5 节强制格式 [doc_id, chunk_id]）

    Attributes:
        doc_id: 文档 ID
        chunk_id: chunk ID
    """

    doc_id: str
    chunk_id: str

    def to_marker(self) -> str:
        """渲染为 [doc_id, chunk_id] 文本标记，嵌入综述段落"""
        return f"[{self.doc_id}, {self.chunk_id}]"


@dataclass
class MapSummary:
    """Map 阶段单文献产物

    Attributes:
        doc_id: 文献 ID
        section_id: 所属章节 ID
        key_points: 该文献在本章节下的要点列表
        citations: 每个要点对应的引用 [doc_id, chunk_id]
        text: 拼接后的要点文本（用于 Reduce 阶段向量化）
    """

    doc_id: str
    section_id: str
    key_points: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    text: str = ""

    def __post_init__(self):
        if not self.text and self.key_points:
            self.text = " ".join(self.key_points)


@dataclass
class Paragraph:
    """Reduce 阶段单章节产物

    Attributes:
        section_id: 章节 ID
        text: 综述段落文本（含 [doc_id, chunk_id] 标记）
        citations: 段落内全部引用
        from_cache: 是否来自段落级缓存
    """

    section_id: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    from_cache: bool = False


@dataclass
class GenerationResult:
    """综述生成总结果（文档 6.3 节：支持部分结果）

    Attributes:
        topic: 综述主题
        paragraphs: 已生成的段落列表（含缓存命中）
        completed: 是否全部章节生成完毕
        failed_sections: 生成失败的章节 ID 列表（LLM 中断时非空）
        retry_hint: 重试提示文案（文档 6.3 节）
        cache_hits: 缓存命中数
        cache_misses: 缓存未命中数
    """

    topic: str
    paragraphs: List[Paragraph] = field(default_factory=list)
    completed: bool = True
    failed_sections: List[str] = field(default_factory=list)
    retry_hint: str = ""
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def full_text(self) -> str:
        """拼接全部段落为完整综述"""
        return "\n\n".join(p.text for p in self.paragraphs)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseReviewGenerator(ABC):
    """综述生成器抽象基类

    设计目的：支持 LLM / Mock 热插拔，本地无 API key 也能跑通全链路。
    """

    @abstractmethod
    def generate(
        self,
        topic: str,
        sections: List,
        doc_chunks: Dict[str, List],
    ) -> GenerationResult:
        """按大纲生成综述

        Args:
            topic: 综述主题
            sections: 章节列表（OutlineSection，含 section_id / title / description / related_doc_ids）
            doc_chunks: {doc_id: [Chunk, ...]}，每篇文献的 chunk 列表（Map 阶段输入）
        Returns:
            GenerationResult
        """
        ...


# ---------------------------------------------------------------------------
# Reduce 阶段 MMR 去重（LLM / Mock 生成器共用）
# ---------------------------------------------------------------------------


class _MMRReduceMixin:
    """Reduce 阶段 MMR 去重 + 统计（LLM 与 Mock 生成器共用）

    文档 4.3 节：在 Reduce 阶段选片段时惩罚已选片段的相似度（MMR），
    把"多篇文献观点高度相似"的重复内容压下去。

    可观测性：embedder 缺失、候选项不足、mmr 导入失败、向量化失败等跳过分支
    必须计数（skipped）并记录 debug 原因——不允许静默降级，
    /review 会把统计透出到 dedup_stats，便于确认 MMR 是否真正执行。

    Attributes:
        REDUCE_TOP_K: Reduce 阶段保留的最大要点数（子类可覆盖，便于测试调小）
        mmr_lambda: 相关性 vs 多样性权衡系数（文档 4.3 节调参默认 0.7）
        mmr_stats: MMR 执行统计
            {calls, input_summaries, selected_summaries, skipped}
    """

    # Reduce 阶段保留的最大要点数：默认值与模块级 REDUCE_TOP_K 一致；
    # 子类 / 测试可通过类属性覆盖（如 monkeypatch 成 1 以验证 MMR 真正执行）
    REDUCE_TOP_K = 8
    mmr_lambda: float = 0.7

    @staticmethod
    def _new_mmr_stats() -> Dict[str, int]:
        """新建 MMR 统计字典（每个生成器实例独立，避免跨请求串号）"""
        return {
            "calls": 0,
            "input_summaries": 0,
            "selected_summaries": 0,
            "skipped": 0,
        }

    def _skip_mmr(
        self,
        reason: str,
        summaries: List[MapSummary],
    ) -> List[MapSummary]:
        """跳过 MMR 并计数（logger.debug 记录原因，不允许静默降级）"""
        self.mmr_stats["skipped"] += 1
        logger.debug(
            "MMR 跳过（%s）: input=%d, REDUCE_TOP_K=%d",
            reason, len(summaries), self.REDUCE_TOP_K,
        )
        return summaries

    def _reduce_with_mmr(
        self,
        summaries: List[MapSummary],
        topic: str,
        section,
    ) -> List[MapSummary]:
        """Reduce 阶段：MMR 去重（复用 app.rag.mmr）

        Args:
            summaries: Map 阶段各文献要点
            topic: 综述主题（仅用于日志）
            section: 章节对象（需有 title / description，作为 MMR 的 query 文本）
        Returns:
            去重后的要点列表（跳过 MMR 时原样返回全部 summaries）
        """
        # 每次调用都统计输入规模（无论最终是否跳过），保证 dedup_stats 可观测
        self.mmr_stats["input_summaries"] += len(summaries)

        if self.embedder is None:
            return self._skip_mmr("embedder 未配置", summaries)
        if len(summaries) <= self.REDUCE_TOP_K:
            return self._skip_mmr(
                f"summaries({len(summaries)}) <= REDUCE_TOP_K({self.REDUCE_TOP_K})",
                summaries,
            )
        try:
            from ..rag.mmr import MMRItem, MMRSelector
        except ImportError:
            return self._skip_mmr("无法导入 app.rag.mmr", summaries)

        # 向量化每个 summary（失败项跳过，不阻塞主流程）
        texts = [s.text for s in summaries]
        try:
            vecs, _failed = self.embedder.embed(texts)
        except Exception as e:  # noqa: BLE001
            return self._skip_mmr(f"summary 向量化异常: {e}", summaries)

        items: List = []
        for s, vec in zip(summaries, vecs):
            if vec is None:
                continue
            items.append(MMRItem(id=s.doc_id, vec=vec, payload=s))
        if not items:
            return self._skip_mmr("无可用 summary 向量", summaries)

        # 用章节标题 + 描述作为 MMR 的 query 向量
        query_text = f"{section.title} {section.description}"
        try:
            query_vec = self.embedder.embed_one(query_text)
        except Exception as e:  # noqa: BLE001
            return self._skip_mmr(f"query 向量化异常: {e}", summaries)
        if query_vec is None:
            return self._skip_mmr("query 向量化失败", summaries)

        selector = MMRSelector(lambda_=self.mmr_lambda)
        selected_items = selector.select_batch(items, query_vec, k=self.REDUCE_TOP_K)

        # 成功选出：calls + selected_summaries 计数
        self.mmr_stats["calls"] += 1
        self.mmr_stats["selected_summaries"] += len(selected_items)
        logger.debug(
            "MMR 去重完成: topic=%s, input=%d, selected=%d, lambda=%.2f",
            (topic or "")[:30], len(summaries), len(selected_items), self.mmr_lambda,
        )
        return [it.payload for it in selected_items]


# ---------------------------------------------------------------------------
# 段落级缓存（文档 6.2 节）
# ---------------------------------------------------------------------------


class ParagraphCache:
    """段落级缓存（文档 6.2 节：综述重跑成本降 40%）

    缓存 key：(topic, section_id, sorted(doc_ids)) 的哈希
    缓存 value：Paragraph

    生产环境替换为 Redis，本地用内存 dict 即可。
    """

    def __init__(self):
        self._store: Dict[str, Paragraph] = {}

    @staticmethod
    def _make_key(topic: str, section_id: str, doc_ids: List[str]) -> str:
        """构造缓存 key（同主题 + 同章节 + 同文献组 → 命中）"""
        raw = f"{topic}||{section_id}||{','.join(sorted(doc_ids))}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, topic: str, section_id: str, doc_ids: List[str]) -> Optional[Paragraph]:
        key = self._make_key(topic, section_id, doc_ids)
        cached = self._store.get(key)
        if cached is not None:
            cached.from_cache = True
            logger.debug("段落缓存命中: section=%s", section_id)
        return cached

    def put(self, topic: str, section_id: str, doc_ids: List[str], paragraph: Paragraph) -> None:
        """写入缓存（文档 6.3 节：每段生成完立即写缓存）"""
        key = self._make_key(topic, section_id, doc_ids)
        # 存副本，避免外部修改影响缓存
        self._store[key] = Paragraph(
            section_id=paragraph.section_id,
            text=paragraph.text,
            citations=list(paragraph.citations),
            from_cache=True,
        )

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# LLM 综述生成器（Map-Reduce + 段落级缓存）
# ---------------------------------------------------------------------------


MAP_PROMPT_TEMPLATE = """\
你是一位学术论文综述撰写助手。请从下列文献片段中，提取与指定章节相关的要点。

综述主题：{topic}
章节标题：{section_title}
章节描述：{section_description}

文献 {doc_id} 的片段：
{chunks_text}

要求：
1. 提取 2-4 个与该章节直接相关的要点
2. 每个要点必须标注引用来源，格式为 [doc_id, chunk_id]
3. 只返回 JSON，不要其他文字

返回格式：
{{
  "key_points": [
    {{"text": "要点内容", "chunk_id": "引用的 chunk_id"}}
  ]
}}
"""


REDUCE_PROMPT_TEMPLATE = """\
你是一位学术论文综述撰写助手。请根据下列要点，撰写综述的"{section_title}"章节段落。

综述主题：{topic}
章节描述：{section_description}

来自多篇文献的要点（已去重）：
{summaries_text}

要求：
1. 撰写一段 200-400 字的综述段落，逻辑连贯
2. 每个观点必须标注引用来源，格式为 [doc_id, chunk_id]
3. 不要简单罗列，要体现文献间的对比和联系
4. 只返回段落正文，不要标题

综述段落：
"""


class LLMReviewGenerator(_MMRReduceMixin, BaseReviewGenerator):
    """LLM 综述生成器（文档 4.3 节 Map-Reduce）

    工作流程：
    1. 段落级缓存命中 → 直接返回（文档 6.2 节）
    2. Map 阶段：逐文献调 LLM 抽要点 + 引用
    3. Reduce 阶段：MMR 去重（复用 app.rag.mmr）→ LLM 生成段落
    4. 写缓存（文档 6.3 节：每段生成完立即写缓存）
    5. LLM 中断 → 记录 failed_sections，返回已生成段落 + retry_hint

    Args:
        api_key: DashScope API key
        model: LLM 模型名
        embedder: 用于 MMR 向量化 map summaries（None 时跳过 MMR）
        mmr_lambda: MMR 多样性权衡系数（文档默认 0.7）
        cache: 段落级缓存（None 时新建内存缓存）
        max_retry: LLM 调用重试次数（文档 6.3 节默认 2）

    用法：
        generator = LLMReviewGenerator(api_key="sk-xxx", embedder=embedder)
        result = generator.generate(topic, sections, doc_chunks)
        if not result.completed:
            print(result.retry_hint)
    """

    MAX_RETRY = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        embedder=None,
        mmr_lambda: float = 0.7,
        cache: Optional[ParagraphCache] = None,
        max_retry: int = MAX_RETRY,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.embedder = embedder
        self.mmr_lambda = mmr_lambda
        self.cache = cache or ParagraphCache()
        self.max_retry = max_retry
        # MMR 执行统计（/review 的 dedup_stats 数据源，跳过也要计数）
        self.mmr_stats = self._new_mmr_stats()
        self._client = None
        self._mock_mode = not bool(self.api_key)
        if self._mock_mode:
            logger.warning(
                "DASHSCOPE_API_KEY 未配置，LLMReviewGenerator 进入 mock 模式"
            )

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.api_key:
            return False
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning("dashscope 包未安装，LLMReviewGenerator 不可用")
            return False

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate(
        self,
        topic: str,
        sections: List,
        doc_chunks: Dict[str, List],
    ) -> GenerationResult:
        # mock 模式或 API 不可用 → 降级（保留 embedder，让 Reduce 阶段 MMR 仍可执行）
        if self._mock_mode or not self._ensure_client():
            logger.info("LLMReviewGenerator 不可用，降级为 MockReviewGenerator")
            fallback = MockReviewGenerator(
                embedder=self.embedder, mmr_lambda=self.mmr_lambda,
            )
            return fallback.generate(topic, sections, doc_chunks)

        paragraphs: List[Paragraph] = []
        failed_sections: List[str] = []
        cache_hits = 0
        cache_misses = 0

        for section in sections:
            section_id = section.section_id
            related_doc_ids = list(section.related_doc_ids)
            if not related_doc_ids:
                related_doc_ids = list(doc_chunks.keys())

            # 1. 段落级缓存命中检查
            cached = self.cache.get(topic, section_id, related_doc_ids)
            if cached is not None:
                paragraphs.append(cached)
                cache_hits += 1
                logger.info("章节 %s 缓存命中，跳过生成", section_id)
                continue

            cache_misses += 1

            # 2. 生成（Map → Reduce）
            try:
                paragraph = self._generate_section(
                    topic, section, related_doc_ids, doc_chunks
                )
                # 3. 写缓存（文档 6.3 节：每段生成完立即写缓存）
                self.cache.put(topic, section_id, related_doc_ids, paragraph)
                paragraphs.append(paragraph)
            except _LLMInterruptedError as e:
                # 文档 6.3 节：LLM 中断，记录失败章节，继续下一节
                logger.warning("章节 %s 生成中断: %s", section_id, e)
                failed_sections.append(section_id)
            except Exception as e:  # noqa: BLE001
                logger.error("章节 %s 生成异常: %s", section_id, e)
                failed_sections.append(section_id)

        completed = len(failed_sections) == 0
        retry_hint = self._build_retry_hint(failed_sections) if not completed else ""

        return GenerationResult(
            topic=topic,
            paragraphs=paragraphs,
            completed=completed,
            failed_sections=failed_sections,
            retry_hint=retry_hint,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    # ------------------------------------------------------------------
    # Map-Reduce 单章节生成
    # ------------------------------------------------------------------

    def _generate_section(
        self,
        topic: str,
        section,
        related_doc_ids: List[str],
        doc_chunks: Dict[str, List],
    ) -> Paragraph:
        """单章节 Map-Reduce 生成

        Raises:
            _LLMInterruptedError: LLM 调用最终失败（文档 6.3 节中断处理）
        """
        # --- Map 阶段：逐文献抽要点 ---
        map_summaries: List[MapSummary] = []
        for doc_id in related_doc_ids:
            chunks = doc_chunks.get(doc_id, [])
            if not chunks:
                logger.warning("文献 %s 无 chunk，跳过 Map", doc_id)
                continue
            summary = self._map_one_doc(topic, section, doc_id, chunks)
            if summary is not None:
                map_summaries.append(summary)

        if not map_summaries:
            # 无可用要点，生成空段落
            return Paragraph(
                section_id=section.section_id,
                text=f"（章节 {section.title} 暂无可用文献要点）",
                citations=[],
            )

        # --- Reduce 阶段：MMR 去重 + LLM 合并 ---
        selected = self._reduce_with_mmr(map_summaries, topic, section)
        paragraph_text = self._reduce_llm(topic, section, selected)

        # 提取段落中的引用
        citations = self._extract_citations(paragraph_text)

        return Paragraph(
            section_id=section.section_id,
            text=paragraph_text,
            citations=citations,
        )

    def _map_one_doc(
        self,
        topic: str,
        section,
        doc_id: str,
        chunks: List,
    ) -> Optional[MapSummary]:
        """Map 阶段：单文献抽要点（调 LLM）

        Raises:
            _LLMInterruptedError: LLM 最终失败
        """
        # 拼接 chunk 文本（截断控制 prompt 长度）
        chunks_text = "\n\n".join(
            f"[chunk_id: {c.chunk_id}] {c.text[:500]}"
            for c in chunks[:5]  # 每篇文献最多取前 5 个 chunk
        )
        prompt = MAP_PROMPT_TEMPLATE.format(
            topic=topic,
            section_title=section.title,
            section_description=section.description,
            doc_id=doc_id,
            chunks_text=chunks_text,
        )

        response = self._call_llm_with_retry(prompt)
        if response is None:
            raise _LLMInterruptedError(f"Map 阶段 LLM 失败: doc_id={doc_id}")

        return self._parse_map_response(doc_id, section.section_id, response)

    def _reduce_llm(
        self,
        topic: str,
        section,
        selected: List[MapSummary],
    ) -> str:
        """Reduce 阶段：LLM 合并生成段落

        Raises:
            _LLMInterruptedError: LLM 最终失败
        """
        summaries_text = "\n\n".join(
            f"[{s.doc_id}] {s.text}" for s in selected
        )
        prompt = REDUCE_PROMPT_TEMPLATE.format(
            topic=topic,
            section_title=section.title,
            section_description=section.description,
            summaries_text=summaries_text,
        )

        response = self._call_llm_with_retry(prompt)
        if response is None:
            raise _LLMInterruptedError("Reduce 阶段 LLM 失败")

        return response.strip()

    # ------------------------------------------------------------------
    # LLM 调用 + 重试（文档 6.3 节）
    # ------------------------------------------------------------------

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """调用 Qwen-Plus，指数退避重试（文档 6.3 节伪代码）

        Returns:
            LLM 响应文本；最终失败返回 None（由调用方决定是否抛 _LLMInterruptedError）
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                resp = self._client.Generation.call(
                    model=self.model,
                    prompt=prompt,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"LLM API 错误: code={resp.code}, msg={resp.message}"
                    )
                return resp.output.text
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retry:
                    wait = 2 ** attempt
                    logger.warning(
                        "综述 LLM 失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.max_retry + 1, wait, e,
                    )
                    time.sleep(wait)
        logger.error("综述 LLM 最终失败: %s", last_error)
        return None

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_map_response(
        doc_id: str,
        section_id: str,
        response_text: str,
    ) -> Optional[MapSummary]:
        """解析 Map 阶段 LLM 返回的 JSON"""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Map 响应 JSON 解析失败: %s, 原始: %s", e, text[:200])
            return None

        key_points_raw = data.get("key_points", []) if isinstance(data, dict) else []
        key_points: List[str] = []
        citations: List[Citation] = []

        for item in key_points_raw:
            if not isinstance(item, dict):
                continue
            point_text = str(item.get("text", "")).strip()
            chunk_id = str(item.get("chunk_id", "")).strip()
            if not point_text:
                continue
            key_points.append(point_text)
            if chunk_id:
                citations.append(Citation(doc_id=doc_id, chunk_id=chunk_id))

        if not key_points:
            return None

        return MapSummary(
            doc_id=doc_id,
            section_id=section_id,
            key_points=key_points,
            citations=citations,
        )

    @staticmethod
    def _extract_citations(text: str) -> List[Citation]:
        """从综述段落文本中提取 [doc_id, chunk_id] 引用"""
        # 复用 citation_verifier 的正则
        from .citation_verifier import CITATION_PATTERN
        citations: List[Citation] = []
        seen = set()
        for m in CITATION_PATTERN.finditer(text):
            doc_id = m.group(1)
            chunk_id = m.group(2)
            key = (doc_id, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(doc_id=doc_id, chunk_id=chunk_id))
        return citations

    # ------------------------------------------------------------------
    # 重试提示（文档 6.3 节）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_retry_hint(failed_sections: List[str]) -> str:
        """构造重试提示文案（文档 6.3 节）

        "段落 X-Y 因网络问题未完成，可点击重试，仅生成失败的部分"
        """
        if not failed_sections:
            return ""
        return (
            f"章节 {', '.join(failed_sections)} 因网络问题未完成，"
            f"可点击重试，仅生成失败的部分。已生成段落已保留。"
        )


# ---------------------------------------------------------------------------
# Mock 综述生成器（无 API key 时降级）
# ---------------------------------------------------------------------------


class MockReviewGenerator(_MMRReduceMixin, BaseReviewGenerator):
    """Mock 综述生成器（无 API key 时降级）

    不调 LLM，直接按"Map（每篇文献取要点）→ Reduce（MMR 去重 + 模板拼接）"
    生成段落，每段含 [doc_id, chunk_id] 引用，保证 citation_verifier 链路可联调。

    Reduce 阶段复用 _MMRReduceMixin：注入 embedder 时真正执行 MMR 去重
    （Mock 下通常是 FakeEmbedder），未注入时跳过并计数到 mmr_stats，
    保证 /review 的 dedup_stats 在 mock 模式下依然可观测、不静默。

    段落级缓存同样生效（验证缓存机制）。

    Args:
        cache: 段落级缓存（None 时新建内存缓存）
        embedder: 向量化器（用于 MMR；None 时跳过 MMR）
        mmr_lambda: MMR 多样性权衡系数（文档默认 0.7）
        api_key: 仅为兼容 create_review_generator 的参数透传（Mock 不调用任何 LLM）
    """

    def __init__(
        self,
        cache: Optional[ParagraphCache] = None,
        embedder=None,
        mmr_lambda: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.cache = cache or ParagraphCache()
        self.embedder = embedder
        self.mmr_lambda = mmr_lambda
        self.api_key = api_key or ""
        # MMR 执行统计（/review 的 dedup_stats 数据源，跳过也要计数）
        self.mmr_stats = self._new_mmr_stats()

    def generate(
        self,
        topic: str,
        sections: List,
        doc_chunks: Dict[str, List],
    ) -> GenerationResult:
        paragraphs: List[Paragraph] = []
        cache_hits = 0
        cache_misses = 0

        for section in sections:
            section_id = section.section_id
            related_doc_ids = list(section.related_doc_ids)
            if not related_doc_ids:
                related_doc_ids = list(doc_chunks.keys())

            # 缓存命中检查
            cached = self.cache.get(topic, section_id, related_doc_ids)
            if cached is not None:
                paragraphs.append(cached)
                cache_hits += 1
                continue
            cache_misses += 1

            # Map（每篇文献取第一个 chunk 的要点）→ Reduce（MMR + 模板拼接）
            paragraph = self._mock_section(topic, section, related_doc_ids, doc_chunks)
            self.cache.put(topic, section_id, related_doc_ids, paragraph)
            paragraphs.append(paragraph)

        return GenerationResult(
            topic=topic,
            paragraphs=paragraphs,
            completed=True,
            failed_sections=[],
            retry_hint="",
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def _mock_section(
        self,
        topic: str,
        section,
        related_doc_ids: List[str],
        doc_chunks: Dict[str, List],
    ) -> Paragraph:
        """单章节模板化生成：Map 要点 → MMR 去重 → Reduce 拼接段落"""
        map_summaries = self._mock_map(section, related_doc_ids, doc_chunks)
        selected = self._reduce_with_mmr(map_summaries, topic, section)
        return self._mock_reduce(topic, section, selected)

    @staticmethod
    def _mock_map(
        section,
        related_doc_ids: List[str],
        doc_chunks: Dict[str, List],
    ) -> List[MapSummary]:
        """Map 阶段：每篇文献取第一个 chunk 作为要点（带引用）"""
        summaries: List[MapSummary] = []
        for doc_id in related_doc_ids:
            chunks = doc_chunks.get(doc_id, [])
            if not chunks:
                continue
            chunk = chunks[0]
            text = (chunk.text or "").strip()
            # text 用于 MMR 向量化（保留更多上下文），key_points 保留 40 字摘要
            summaries.append(MapSummary(
                doc_id=doc_id,
                section_id=section.section_id,
                key_points=[text.replace("\n", " ")[:40]],
                citations=[Citation(doc_id=doc_id, chunk_id=chunk.chunk_id)],
                text=text[:500],
            ))
        return summaries

    @staticmethod
    def _mock_reduce(
        topic: str,
        section,
        selected: List[MapSummary],
    ) -> Paragraph:
        """Reduce 阶段：把（MMR 选出的）要点拼成模板段落，保留引用标记"""
        citations: List[Citation] = []
        sentences: List[str] = []

        # 章节引言
        sentences.append(
            f"关于{topic}的{section.title}，"
            f"{section.description}。"
        )

        # 每篇（去重后）文献一句话 + 引用
        for summary in selected:
            preview = (summary.text or "").replace("\n", " ").strip()[:40]
            for citation in summary.citations:
                citations.append(citation)
                sentences.append(
                    f"文献 [{citation.doc_id}, {citation.chunk_id}] 指出，{preview}..."
                )

        # 章节小结
        sentences.append("综合来看，该方向已有较多探索，但仍存在改进空间。")

        return Paragraph(
            section_id=section.section_id,
            text="".join(sentences),
            citations=citations,
        )


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class _LLMInterruptedError(Exception):
    """LLM 调用中断（文档 6.3 节：返回部分结果，不丢失进度）"""
    pass


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_review_generator(mode: str = "auto", **kwargs) -> BaseReviewGenerator:
    """工厂函数：按模式创建综述生成器

    Args:
        mode: "llm" / "mock" / "auto"
            - auto：优先 LLM，无 API key 降级 Mock
        **kwargs: 传给具体 generator 的参数
    Returns:
        BaseReviewGenerator 实例
    """
    mode = mode.lower()

    if mode == "mock":
        return MockReviewGenerator(**kwargs)

    if mode == "llm":
        return LLMReviewGenerator(**kwargs)

    if mode == "auto":
        generator = LLMReviewGenerator(**kwargs)
        if generator._ensure_client():
            return generator
        logger.info("auto 模式：LLM 不可用，降级为 MockReviewGenerator")
        # 降级时保留 embedder，让 Reduce 阶段 MMR 仍能真正执行（可观测不静默）
        fallback_kwargs: Dict = {}
        if kwargs.get("embedder") is not None:
            fallback_kwargs["embedder"] = kwargs["embedder"]
        if "mmr_lambda" in kwargs:
            fallback_kwargs["mmr_lambda"] = kwargs["mmr_lambda"]
        return MockReviewGenerator(**fallback_kwargs)

    raise ValueError(f"未知 review generator mode: {mode}")

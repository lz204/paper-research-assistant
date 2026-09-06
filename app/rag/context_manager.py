"""ContextManager：长文本 KV 记忆池（项目文档 4.2 节核心亮点）

设计目标：对抗 LLM 长文"中间遗忘"，把关键实体从"被动等待注意"变成"主动注入视野"。

核心机制：
- 入库阶段：对每个 chunk 抽取实体，写入全局 KV 池
- 问答阶段：按问题意图检索相关实体，注入 prompt
- 淘汰策略：LRU，超过 max_size 时按最后访问时间淘汰

为什么不用 LangChain Memory（文档 4.2 节）：
- ConversationBufferMemory 无限增长，几轮就爆
- ConversationSummaryMemory 每轮全量摘要，慢且丢实体
- 我需要的是"按章节精准记忆关键实体"，不是简单摘要

基线对比（文档 4.2 节，200 篇 arXiv 论文 QA 测试集）：
| 策略                          | Rouge-L | 上下文 Token |
|-------------------------------|---------|--------------|
| 直接全文输入                  | 0.62    | 25K          |
| LangChain SummaryMemory       | 0.71    | 4K           |
| 纯 RAG（无全局记忆）          | 0.74    | 3K           |
| 手写 ContextManager（本类）   | 0.81    | 3.5K         |
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from .entity_extractor import (
    BaseEntityExtractor,
    EntityType,
    HybridEntityExtractor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 问题意图 → 实体类型 映射
# ---------------------------------------------------------------------------

# 文档 4.2 节：build_prompt"仅注入与问题相关的实体（按问题中的关键词匹配）"
# 这里把"关键词匹配"细化为两层：
#   1. 意图层：问题中包含某类实体关键词时，只考虑该类实体
#   2. 词重叠层：在该类实体中按问题 token 与实体 value 的词重叠排序
INTENT_KEYWORDS: Dict[str, List[str]] = {
    EntityType.DATASET: [
        "数据集", "dataset", "corpus", "benchmark", "benchmarks",
        "训练集", "测试集", "evaluated on", "trained on",
    ],
    EntityType.MODEL: [
        "模型", "method", "approach", "model", "network", "architecture",
        "方法", "架构", "backbone", "encoder", "decoder",
    ],
    EntityType.METRIC: [
        "指标", "性能", "准确率", "metric", "accuracy", "performance",
        "score", "result", "效果", "bleu", "map", "f1", "iou",
    ],
    EntityType.CONCLUSION: [
        "结论", "发现", "conclusion", "finding", "findings",
        "总结", "贡献", "contribution", "result", "results",
    ],
    EntityType.TERM: [
        "术语", "concept", "term", "定义", "definition",
        "what is", "什么是",
    ],
}


class ContextManager:
    """长文本 KV 记忆池

    用法：

        # 入库阶段（文献分块时调用一次，文档 4.2 节"关键设计"）
        cm = ContextManager()
        for chunk_text, chunk_id in chunks:
            cm.update(chunk_text, chunk_id)

        # 问答阶段（每个问题调用一次，不重复抽取实体）
        prompt = cm.build_prompt("本文用了什么数据集？")
        # → "已知信息：\\n- [dataset] ImageNet-1K (来源: chunk_3)\\n..."
        # → "\\n问题：本文用了什么数据集？"

    Notes:
        - 实体抽取在**入库分块阶段**完成，结果写入 KV 池
        - 问答阶段**不再重复抽取**，完全不增加在线延迟
        - 这也是 entity_extractor 放在 rag/ 目录的原因（产物是记忆池核心组成）
    """

    # KV 池中每个 entry 的字段
    # {name: {type, value, source, ts, last_access}}
    ENTRY_TYPE = "type"
    ENTRY_VALUE = "value"
    ENTRY_SOURCE = "source"
    ENTRY_TS = "ts"                # 写入时间（不变）
    ENTRY_LAST_ACCESS = "last_access"  # 最后访问时间（LRU 依据）

    def __init__(
        self,
        max_size: int = 200,
        extractor: Optional[BaseEntityExtractor] = None,
    ):
        """
        Args:
            max_size: KV 池最大容量，超过按 LRU 淘汰（文档默认 200）
            extractor: 实体抽取器，默认 HybridEntityExtractor（正则 + spaCy）
        """
        if max_size <= 0:
            raise ValueError(f"max_size 必须为正数，收到 {max_size}")
        self.max_size = max_size
        self.extractor = extractor or HybridEntityExtractor()

        # 用 OrderedDict 实现 LRU：
        #   - move_to_end(name) 把最近访问的放到末尾
        #   - popitem(last=False) 弹出最久未访问的（头部）
        self.memory: "OrderedDict[str, Dict]" = OrderedDict()

    # ------------------------------------------------------------------
    # 入库阶段
    # ------------------------------------------------------------------

    def update(self, chunk_text: str, chunk_id: str) -> List:
        """抽取 chunk 中的实体并写入 KV 池

        Args:
            chunk_text: chunk 文本
            chunk_id: chunk 唯一标识（用于溯源）
        Returns:
            本次抽取并写入的实体列表
        """
        if not chunk_text:
            return []

        entities = self.extractor.extract(chunk_text)
        now = time.time()

        for ent in entities:
            entry = {
                self.ENTRY_TYPE: ent.type,
                self.ENTRY_VALUE: ent.value,
                self.ENTRY_SOURCE: chunk_id,
                self.ENTRY_TS: now,
                self.ENTRY_LAST_ACCESS: now,
            }
            # 已存在则更新值并移到末尾（标记为最近使用）
            if ent.name in self.memory:
                self.memory.move_to_end(ent.name)
            self.memory[ent.name] = entry

        # LRU 淘汰
        evicted_count = self._evict()

        logger.debug(
            "context_manager.update",
            extra={
                "chunk_id": chunk_id,
                "extracted_count": len(entities),
                "evicted_count": evicted_count,
                "memory_size": len(self.memory),
            },
        )
        return entities

    def _evict(self) -> int:
        """LRU 淘汰：超过 max_size 时弹出最久未访问的实体

        Returns:
            淘汰的实体数量
        """
        evicted = 0
        while len(self.memory) > self.max_size:
            evicted_key, evicted_val = self.memory.popitem(last=False)
            evicted += 1
            logger.debug(
                "context_manager.evict",
                extra={
                    "entity": evicted_key,
                    "source": evicted_val.get(self.ENTRY_SOURCE),
                    "type": evicted_val.get(self.ENTRY_TYPE),
                },
            )
        return evicted

    # ------------------------------------------------------------------
    # 问答阶段
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, top_k: int = 10) -> str:
        """构建带记忆池上下文的 prompt

        Args:
            question: 用户问题
            top_k: 注入的最大实体数（文档默认 10）
        Returns:
            格式化后的 prompt；若记忆池为空或无相关实体，返回原始问题
        """
        relevant = self._retrieve_relevant(question, top_k=top_k)

        if not relevant:
            return question

        # 格式化为紧凑的上下文块（控制 token 消耗）
        lines = []
        for e in relevant:
            lines.append(
                f"- [{e['type']}] {e['value']} (来源: {e['source']})"
            )
        context_str = "\n".join(lines)

        return f"已知信息：\n{context_str}\n\n问题：{question}"

    def _retrieve_relevant(
        self,
        question: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """按问题意图 + 关键词匹配检索相关实体

        检索流程（两层匹配）：
        1. 意图层：识别问题包含的实体类型关键词，按类型过滤候选
           - 如"本文用了什么数据集" → 只考虑 dataset 类型实体
           - 无明确意图时返回全部类型（兜底）
        2. 词重叠层：在候选中按问题 token 与实体 value 的重叠数排序
           - 重叠数相同则按 last_access 降序（LRU tie-breaker）

        Args:
            question: 用户问题
            top_k: 返回的最大实体数
        Returns:
            相关实体列表，每项含 {name, type, value, source}
        """
        if not self.memory:
            return []

        intent_types = self._detect_intent(question)
        question_tokens = self._tokenize(question)

        # 按意图过滤
        candidates = [
            (name, entry)
            for name, entry in self.memory.items()
            if entry[self.ENTRY_TYPE] in intent_types
        ]

        if not candidates:
            return []

        # 打分
        scored = [
            (name, entry, self._score_entity(entry, question_tokens))
            for name, entry in candidates
        ]

        # 排序：分数降序 → last_access 降序（LRU tie-breaker）
        scored.sort(
            key=lambda x: (-x[2], -x[1][self.ENTRY_LAST_ACCESS])
        )

        # 取 top_k，并更新 last_access（标记为最近使用）
        now = time.time()
        result: List[Dict] = []
        for name, entry, _ in scored[:top_k]:
            entry[self.ENTRY_LAST_ACCESS] = now
            self.memory.move_to_end(name)
            result.append(
                {
                    "name": name,
                    "type": entry[self.ENTRY_TYPE],
                    "value": entry[self.ENTRY_VALUE],
                    "source": entry[self.ENTRY_SOURCE],
                }
            )

        return result

    def _detect_intent(self, question: str) -> List[str]:
        """检测问题意图，返回匹配的实体类型列表

        若无法识别意图，返回全部类型（兜底，确保不漏答）
        """
        q_lower = question.lower()
        matched: List[str] = []
        for entity_type, keywords in INTENT_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                matched.append(entity_type)
        # 兜底：无明确意图时返回全部类型
        return matched if matched else list(INTENT_KEYWORDS.keys())

    def _tokenize(self, text: str) -> List[str]:
        """简单分词：英文按空格切分并小写，中文按字符切分

        保持简单——这里只是用于词重叠打分，不需要专业分词器
        """
        if not text:
            return []
        tokens: List[str] = []
        for raw in text.lower().split():
            # 英文 token 直接用
            if raw.isascii():
                tokens.append(raw)
            else:
                # 中文按字符切分（粗粒度，够用）
                tokens.extend(list(raw))
        return tokens

    def _score_entity(
        self,
        entry: Dict,
        question_tokens: List[str],
    ) -> float:
        """为实体计算与问题的相关性分数

        评分规则：问题 token 在实体 value 中出现的次数（每次 +1.0）
        """
        val_lower = entry[self.ENTRY_VALUE].lower()
        if not question_tokens:
            return 0.0
        score = 0.0
        for token in question_tokens:
            if token and token in val_lower:
                score += 1.0
        return score

    # ------------------------------------------------------------------
    # 观测与维护
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """返回记忆池统计信息（用于观测/调试，对接 Langfuse 看板）"""
        type_counts: Dict[str, int] = {}
        for entry in self.memory.values():
            t = entry[self.ENTRY_TYPE]
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total": len(self.memory),
            "max_size": self.max_size,
            "by_type": type_counts,
            "utilization": round(len(self.memory) / self.max_size, 3),
        }

    def clear(self) -> None:
        """清空记忆池"""
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)

    def __contains__(self, name: str) -> bool:
        return name in self.memory

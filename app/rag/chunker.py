"""层级分块器（项目文档第 5 章选型 + 附录）

选型理由（文档第 5 章）：
- chunk_size=256：学术段落信息密度高，512 经常跨越多个论点；256 召回率 +6%
- overlap=64：避免论点被切断时丢失上下文

层级分块含义：
- 保留章节/子章节层级信息（section / subsection）
- 优先尊重段落边界，段落过长再按句子切
- 每个 chunk 带 metadata，便于后续引用溯源（chunk_id 格式：{doc_id}#{section_idx}.{chunk_idx}）

纯 Python 实现，无外部依赖（tiktoken 可选，缺失时用字符数近似）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """分块后的文本片段

    Attributes:
        chunk_id: 唯一标识，格式 {doc_id}#{section_idx}.{chunk_idx}
        text: chunk 文本
        token_count: token 数（用于成本估算）
        doc_id: 所属文档 ID
        section: 章节标题（如 "3. Method"）
        section_idx: 章节序号（从 0 开始）
        chunk_idx: 章节内 chunk 序号（从 0 开始）
        overlap_prev: 与前一个 chunk 的重叠 token 数
        metadata: 额外元数据（页码、坐标等，由 parser 填充）
    """

    chunk_id: str
    text: str
    token_count: int
    doc_id: str
    section: str = ""
    section_idx: int = 0
    chunk_idx: int = 0
    overlap_prev: int = 0
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.token_count


# ---------------------------------------------------------------------------
# token 计数
# ---------------------------------------------------------------------------


class TokenCounter:
    """token 计数器，优先用 tiktoken，降级用字符数近似

    文档场景下 GPT/Qwen 系列的 token 与字符大致 1:4，
    降级模式足够用于分块决策（不需要精确）。
    """

    def __init__(self):
        self._encoder = None
        self._use_tiktoken = False
        try:
            import tiktoken  # type: ignore
            self._encoder = tiktoken.get_encoding("cl100k_base")
            self._use_tiktoken = True
        except Exception:  # noqa: BLE001
            logger.info("tiktoken 不可用，降级用字符数/4 近似 token 数")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._use_tiktoken and self._encoder is not None:
            return len(self._encoder.encode(text))
        # 降级：1 token ≈ 4 字符（英文）/ 1.5 字符（中文混合）
        # 取折中 3 字符/token
        return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseChunker:
    """分块器抽象基类"""

    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 层级分块器（项目默认实现）
# ---------------------------------------------------------------------------


# Markdown / LaTeX 章节标题识别
# 匹配 "# Title"、"## Title"、"1. Title"、"1.1 Title"、"\section{Title}"
SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(#{1,6})\s+(.+)"          # Markdown: # Title
    r"|"
    r"(\d+(?:\.\d+)*)\.?\s+(.+)"  # 编号: 1. Title / 1.1 Title
    r"|"
    r"\\section\{([^}]+)\}"      # LaTeX: \section{Title}
    r"|"
    r"\\subsection\{([^}]+)\}"   # LaTeX: \subsection{Title}
    r")$",
    re.MULTILINE,
)

# 段落分隔：两个以上换行
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
# 句子分隔：句号/问号/感叹号 + 空格/换行
SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


class HierarchicalChunker(BaseChunker):
    """层级分块器

    算法：
    1. 按章节标题切分文档（保留层级信息）
    2. 每个章节内按段落切分
    3. 段落累积到 chunk_size 时输出一个 chunk
    4. 单段落超过 chunk_size 时按句子切分
    5. chunk 之间保留 overlap 个 token 的重叠（取前一 chunk 末尾）

    Args:
        chunk_size: 单 chunk 最大 token 数（文档默认 256）
        overlap: 相邻 chunk 重叠 token 数（文档默认 64）
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数，收到 {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(f"overlap 必须在 [0, chunk_size) 区间，收到 {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.counter = TokenCounter()

    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        """对全文做层级分块

        Args:
            text: 全文文本（含章节标题）
            doc_id: 文档唯一标识
        Returns:
            Chunk 列表（按章节顺序、章节内顺序排列）
        """
        if not text or not text.strip():
            return []

        sections = self._split_sections(text)
        all_chunks: List[Chunk] = []
        prev_tail_text = ""  # 前一 chunk 的末尾文本，用于 overlap

        for section_idx, (section_title, section_text) in enumerate(sections):
            section_chunks = self._chunk_section(
                section_text,
                doc_id=doc_id,
                section=section_title,
                section_idx=section_idx,
                prev_tail_text=prev_tail_text,
            )
            if section_chunks:
                all_chunks.extend(section_chunks)
                # 下一章节的 overlap 取本章节最后一个 chunk 的末尾
                prev_tail_text = self._tail_text(section_chunks[-1].text)

        return all_chunks

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> List[tuple]:
        """按章节标题切分文档

        Returns:
            [(section_title, section_text), ...]
            无标题的文档返回 [("（无标题）", 全文)] 单元素列表
        """
        matches = list(SECTION_PATTERN.finditer(text))
        if not matches:
            return [("（无标题）", text)]

        sections: List[tuple] = []
        # 第一个标题之前的内容作为"前言"
        if matches[0].start() > 0:
            preface = text[: matches[0].start()].strip()
            if preface:
                sections.append(("（前言）", preface))

        for i, m in enumerate(matches):
            # 提取标题文本（哪个分组命中了取哪个）
            title = (
                m.group(2)  # Markdown
                or m.group(4)  # 编号
                or m.group(5)  # LaTeX section
                or m.group(6)  # LaTeX subsection
                or "（未命名章节）"
            ).strip()
            # 加上编号前缀（如果是编号标题）
            if m.group(3):
                title = f"{m.group(3)}. {title}"

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))

        return sections

    def _chunk_section(
        self,
        section_text: str,
        doc_id: str,
        section: str,
        section_idx: int,
        prev_tail_text: str,
    ) -> List[Chunk]:
        """对单章节文本做分块"""
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(section_text) if p.strip()]
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 0
        buffer: List[str] = []
        buffer_tokens = 0
        overlap_tokens = 0
        # 标记 buffer 中是否已追加过"本节自身内容"（区别于上一节带来的 overlap 前缀），
        # 用于避免短章节（总 token ≤ overlap）被整体丢弃
        has_own_content = False

        # 把上一章节末尾的 overlap 文本作为本章节第一个 chunk 的前缀
        if prev_tail_text:
            buffer.append(prev_tail_text)
            buffer_tokens = self.counter.count(prev_tail_text)
            overlap_tokens = buffer_tokens

        for para in paragraphs:
            para_tokens = self.counter.count(para)

            # 单段落就超过 chunk_size：按句子切
            if para_tokens > self.chunk_size:
                # 先 flush 当前 buffer
                if buffer and (buffer_tokens > self.overlap or has_own_content):
                    chunks.append(
                        self._build_chunk(
                            buffer, doc_id, section, section_idx, chunk_idx,
                            overlap_tokens,
                        )
                    )
                    chunk_idx += 1
                    buffer = [self._tail_text("\n\n".join(buffer))]
                    buffer_tokens = self.counter.count(buffer[0])
                    overlap_tokens = buffer_tokens
                    has_own_content = False

                # 按句子切分超长段落
                for sent_chunk in self._split_long_paragraph(para):
                    if buffer and buffer_tokens + self.counter.count(sent_chunk) > self.chunk_size:
                        chunks.append(
                            self._build_chunk(
                                buffer, doc_id, section, section_idx, chunk_idx,
                                overlap_tokens,
                            )
                        )
                        chunk_idx += 1
                        buffer = [self._tail_text("\n\n".join(buffer))]
                        buffer_tokens = self.counter.count(buffer[0])
                        overlap_tokens = buffer_tokens
                        has_own_content = False
                    buffer.append(sent_chunk)
                    buffer_tokens = self.counter.count("\n\n".join(buffer))
                    has_own_content = True
                continue

            # 普通段落：尝试加入 buffer
            if buffer_tokens + para_tokens > self.chunk_size and buffer:
                # buffer 已满，输出 chunk
                chunks.append(
                    self._build_chunk(
                        buffer, doc_id, section, section_idx, chunk_idx,
                        overlap_tokens,
                    )
                )
                chunk_idx += 1
                # 保留 overlap
                buffer = [self._tail_text("\n\n".join(buffer))]
                buffer_tokens = self.counter.count(buffer[0])
                overlap_tokens = buffer_tokens
                has_own_content = False

            buffer.append(para)
            buffer_tokens = self.counter.count("\n\n".join(buffer))
            has_own_content = True

        # flush 剩余（短章节即使 token ≤ overlap，只要含本节自身内容也要输出，避免内容丢失）
        if buffer and (buffer_tokens > self.overlap or has_own_content):
            chunks.append(
                self._build_chunk(
                    buffer, doc_id, section, section_idx, chunk_idx, overlap_tokens,
                )
            )

        return chunks

    def _split_long_paragraph(self, para: str) -> List[str]:
        """对超长段落按句子切分，尽量让每段接近 chunk_size"""
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(para) if s.strip()]
        if not sentences:
            return [para]

        result: List[str] = []
        buffer: List[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self.counter.count(sent)
            if buffer_tokens + sent_tokens > self.chunk_size and buffer:
                result.append(" ".join(buffer))
                buffer = [sent]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sent)
                buffer_tokens += sent_tokens

        if buffer:
            result.append(" ".join(buffer))

        return result

    def _build_chunk(
        self,
        buffer: List[str],
        doc_id: str,
        section: str,
        section_idx: int,
        chunk_idx: int,
        overlap_tokens: int,
    ) -> Chunk:
        text = "\n\n".join(buffer)
        return Chunk(
            chunk_id=f"{doc_id}#{section_idx}.{chunk_idx}",
            text=text,
            token_count=self.counter.count(text),
            doc_id=doc_id,
            section=section,
            section_idx=section_idx,
            chunk_idx=chunk_idx,
            overlap_prev=overlap_tokens,
        )

    def _tail_text(self, text: str) -> str:
        """取文本末尾 overlap 个 token 对应的字符，用于下一 chunk 的前缀"""
        if self.overlap <= 0 or not text:
            return ""
        tokens = self.counter.count(text)
        if tokens <= self.overlap:
            return text
        # 近似：按比例截取末尾字符
        char_ratio = self.overlap / tokens
        keep_chars = int(len(text) * char_ratio)
        return text[-keep_chars:] if keep_chars > 0 else ""

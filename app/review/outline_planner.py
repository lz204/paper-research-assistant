"""综述大纲规划模块（项目文档 4.3 节 + 第 3 章架构图）

文档第 3 章架构图：
    Review[综述主题] --> RAG2[多文档检索] --> MMR[MMR 去重]
    MMR --> Gen[Map-Reduce 生成] --> Verify[引用事实验证]

本模块对应 "Review → 多文档检索" 之前的步骤：先把综述主题拆成有序的章节大纲，
后续 generator 按章节分段生成，citation_verifier 按段落验证引用。

文档 4.3 节 Map-Reduce 生成要求：
- Map 阶段：逐文献抽要点
- Reduce 阶段：合并去重
所以大纲必须能挂载"每节对应哪些文献"，让 Map 阶段有明确的输入边界。

设计：
- OutlineSection dataclass：章节（含标题、描述、关键论点、相关文献）
- OutlinePlan dataclass：完整大纲（主题 + 章节列表）
- BaseOutlinePlanner ABC：支持热插拔
- LLMOutlinePlanner：Qwen-Plus 生成结构化大纲（JSON 解析 + 重试）
- MockOutlinePlanner：无 API key 时降级，按主题模板生成大纲
- create_outline_planner：工厂函数
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class OutlineSection:
    """大纲章节

    Attributes:
        section_id: 章节唯一标识（如 "s0", "s1"），用于 generator 段落级缓存 key
        title: 章节标题（如 "Transformer 在医学影像分割的应用"）
        description: 章节描述（指导 generator 生成什么内容）
        key_points: 关键论点列表（LLM 生成或模板填充）
        related_doc_ids: 该章节涉及的文献 doc_id 列表（Map 阶段的输入边界）
    """

    section_id: str
    title: str
    description: str
    key_points: List[str] = field(default_factory=list)
    related_doc_ids: List[str] = field(default_factory=list)


@dataclass
class OutlinePlan:
    """完整大纲

    Attributes:
        topic: 综述主题
        sections: 章节列表（有序）
        doc_ids: 全部文献 doc_id 列表（兜底，未被分到任何章节的文献可整体进入背景章节）
    """

    topic: str
    sections: List[OutlineSection]
    doc_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self):
        return iter(self.sections)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseOutlinePlanner(ABC):
    """大纲规划器抽象基类

    设计目的：支持 LLM / Mock / 规则等多种规划策略热插拔。
    """

    @abstractmethod
    def plan(
        self,
        topic: str,
        doc_infos: List[Dict],
        num_sections: int = 4,
    ) -> OutlinePlan:
        """规划综述大纲

        Args:
            topic: 综述主题（限 500 字，文档 6.5 节安全限制）
            doc_infos: 文献信息列表，每项至少含 doc_id，可选含 title / section 摘要
            num_sections: 期望章节数（LLM 可微调）
        Returns:
            OutlinePlan
        """
        ...


# ---------------------------------------------------------------------------
# LLM 大纲规划器（Qwen-Plus）
# ---------------------------------------------------------------------------


# 文档 6.5 节：综述主题限 500 字，防 prompt 注入和 token 爆炸
MAX_TOPIC_LEN = 500

DEFAULT_OUTLINE_PROMPT_TEMPLATE = """\
你是一位学术论文综述撰写专家。请根据给定的综述主题和文献列表，规划综述大纲。

要求：
1. 大纲包含 {num_sections} 个左右的章节，每节标题简洁（不超过 20 字）
2. 每节描述说明该节应讨论什么内容
3. key_points 列出该节应覆盖的 2-4 个核心论点
4. related_doc_ids 从给定文献列表中选择与该节最相关的文献 ID
5. 只返回 JSON，不要其他文字

综述主题：{topic}

文献列表（JSON）：
{doc_infos_json}

返回格式：
{{
  "sections": [
    {{
      "title": "章节标题",
      "description": "章节描述",
      "key_points": ["论点1", "论点2"],
      "related_doc_ids": ["doc_id_1", "doc_id_2"]
    }}
  ]
}}
"""


class LLMOutlinePlanner(BaseOutlinePlanner):
    """用 Qwen-Plus 生成综述大纲（文档 4.3 节 + 第 5 章选型）

    工作流程：
    1. 把主题 + 文献信息拼成 prompt
    2. LLM 返回 JSON 结构化大纲
    3. 解析为 OutlinePlan，section_id 自动编号

    容错（文档 6.3 节）：
    - API 超时/限流时指数退避重试 2 次
    - 最终失败降级为 MockOutlinePlanner，不阻塞主流程

    用法：
        planner = LLMOutlinePlanner(api_key="sk-xxx")
        plan = planner.plan("Transformer 在医学影像分割的应用", doc_infos)
    """

    MAX_RETRY = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self._client = None
        self._mock_mode = not bool(self.api_key)
        if self._mock_mode:
            logger.warning(
                "DASHSCOPE_API_KEY 未配置，LLMOutlinePlanner 进入 mock 模式"
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
            logger.warning("dashscope 包未安装，LLMOutlinePlanner 不可用")
            return False

    def plan(
        self,
        topic: str,
        doc_infos: List[Dict],
        num_sections: int = 4,
    ) -> OutlinePlan:
        # 文档 6.5 节：主题长度限制
        if len(topic) > MAX_TOPIC_LEN:
            logger.warning(
                "综述主题长度 %d 超过限制 %d，将被截断",
                len(topic), MAX_TOPIC_LEN,
            )
            topic = topic[:MAX_TOPIC_LEN]

        # mock 模式或 API 不可用 → 降级
        if self._mock_mode or not self._ensure_client():
            logger.info("LLMOutlinePlanner 不可用，降级为 MockOutlinePlanner")
            return MockOutlinePlanner().plan(topic, doc_infos, num_sections)

        prompt = self._build_prompt(topic, doc_infos, num_sections)
        response_text = self._call_llm_with_retry(prompt)

        if response_text is None:
            logger.warning("LLM 大纲生成失败，降级为 MockOutlinePlanner")
            return MockOutlinePlanner().plan(topic, doc_infos, num_sections)

        plan = self._parse_response(topic, response_text, doc_infos)
        if plan is None or len(plan) == 0:
            logger.warning("LLM 大纲解析失败，降级为 MockOutlinePlanner")
            return MockOutlinePlanner().plan(topic, doc_infos, num_sections)

        return plan

    def _build_prompt(
        self,
        topic: str,
        doc_infos: List[Dict],
        num_sections: int,
    ) -> str:
        """构造大纲生成 prompt"""
        # 只保留必要字段，控制 prompt 长度
        slim_infos = [
            {
                "doc_id": d.get("doc_id", f"doc_{i}"),
                "title": d.get("title", "")[:80],
            }
            for i, d in enumerate(doc_infos)
        ]
        return DEFAULT_OUTLINE_PROMPT_TEMPLATE.format(
            num_sections=num_sections,
            topic=topic,
            doc_infos_json=json.dumps(slim_infos, ensure_ascii=False, indent=2),
        )

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """调用 Qwen-Plus，指数退避重试（文档 6.3 节）"""
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
                return resp.output.text
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt
                    logger.warning(
                        "大纲生成 LLM 失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.MAX_RETRY + 1, wait, e,
                    )
                    time.sleep(wait)
        logger.error("大纲生成 LLM 最终失败: %s", last_error)
        return None

    @staticmethod
    def _parse_response(
        topic: str,
        response_text: str,
        doc_infos: List[Dict],
    ) -> Optional[OutlinePlan]:
        """解析 LLM 返回的 JSON 大纲"""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("大纲 JSON 解析失败: %s, 原始: %s", e, text[:200])
            return None

        all_doc_ids = [d.get("doc_id", f"doc_{i}") for i, d in enumerate(doc_infos)]
        sections: List[OutlineSection] = []
        raw_sections = data.get("sections", []) if isinstance(data, dict) else []

        for i, sec in enumerate(raw_sections):
            if not isinstance(sec, dict):
                continue
            related = [
                str(d) for d in sec.get("related_doc_ids", [])
                if isinstance(d, str) and d in all_doc_ids
            ]
            sections.append(
                OutlineSection(
                    section_id=f"s{i}",
                    title=str(sec.get("title", f"章节 {i + 1}")),
                    description=str(sec.get("description", "")),
                    key_points=[str(p) for p in sec.get("key_points", [])],
                    related_doc_ids=related,
                )
            )
        return OutlinePlan(topic=topic, sections=sections, doc_ids=all_doc_ids)


# ---------------------------------------------------------------------------
# Mock 大纲规划器（无 API key 时降级）
# ---------------------------------------------------------------------------


# 默认章节模板（与文档第 2 章"综述草稿生成"对应）
DEFAULT_SECTION_TEMPLATES = [
    ("研究背景", "介绍该领域的发展背景与问题动机"),
    ("核心方法", "梳理代表性方法的技术路线与差异"),
    ("实验对比", "对比各方法在公开数据集上的性能"),
    ("未来展望", "总结现状并指出潜在研究方向"),
]


class MockOutlinePlanner(BaseOutlinePlanner):
    """Mock 大纲规划器（无 API key 时降级）

    按模板生成 4 个章节，文献列表平均分配到各章节，
    保证 generator / citation_verifier 链路可联调。
    """

    def __init__(self, templates: Optional[List] = None):
        self.templates = templates or DEFAULT_SECTION_TEMPLATES

    def plan(
        self,
        topic: str,
        doc_infos: List[Dict],
        num_sections: int = 4,
    ) -> OutlinePlan:
        if len(topic) > MAX_TOPIC_LEN:
            topic = topic[:MAX_TOPIC_LEN]

        all_doc_ids = [d.get("doc_id", f"doc_{i}") for i, d in enumerate(doc_infos)]
        if num_sections <= len(self.templates):
            templates = self.templates[:num_sections]
        else:
            templates = list(self.templates) + [
                ("补充章节", "补充讨论")
            ] * (num_sections - len(self.templates))

        # 文献平均分配到各章节：前 n_docs % n_sec 个章节多分 1 篇
        n_docs = len(all_doc_ids)
        n_sec = len(templates)
        sections: List[OutlineSection] = []
        cursor = 0
        for i, (title, desc) in enumerate(templates):
            base = n_docs // n_sec if n_sec else 0
            extra = 1 if (n_sec and i < n_docs % n_sec) else 0
            count = base + extra
            related = all_doc_ids[cursor:cursor + count]
            cursor += count
            sections.append(
                OutlineSection(
                    section_id=f"s{i}",
                    title=f"{title}：{topic[:20]}" if len(topic) <= 20 else title,
                    description=desc,
                    key_points=[f"{title}下的核心论点 1", f"{title}下的核心论点 2"],
                    related_doc_ids=related,
                )
            )
        return OutlinePlan(topic=topic, sections=sections, doc_ids=all_doc_ids)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_outline_planner(mode: str = "auto", **kwargs) -> BaseOutlinePlanner:
    """工厂函数：按模式创建大纲规划器

    Args:
        mode: "llm" / "mock" / "auto"
            - auto：优先 LLM，无 API key 降级 Mock
        **kwargs: 传给具体 planner 的参数
    Returns:
        BaseOutlinePlanner 实例
    """
    mode = mode.lower()

    if mode == "mock":
        return MockOutlinePlanner(**kwargs)

    if mode == "llm":
        return LLMOutlinePlanner(**kwargs)

    if mode == "auto":
        planner = LLMOutlinePlanner(**kwargs)
        if planner._ensure_client():
            return planner
        logger.info("auto 模式：LLM 不可用，降级为 MockOutlinePlanner")
        return MockOutlinePlanner()

    raise ValueError(f"未知 outline planner mode: {mode}")

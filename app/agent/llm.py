"""Agent 的 Function Calling LLM 抽象（冻结契约 v1）。

模块组成：
- ToolCallRequest / ToolCallingResponse：工具调用请求与统一响应结构
- BaseToolCallingLLM：抽象基类（simple / LangGraph 两种编排共用）
- DashScopeToolCallingLLM：真实 Qwen-Plus function calling（失败返回 error，不抛异常）
- ScriptedMockLLM：确定性脚本化 mock，无 API key 时也能走完多步 ReAct 流程
- create_tool_calling_llm：工厂函数（auto：有 key 且可 import → DashScope，否则 mock）

安全约束：测试 / 本地默认无 DASHSCOPE_API_KEY 时只会构造 mock 客户端，
不会发起任何网络请求；真实调用仅发生在显式可用（key + dashscope）时。
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from ..llm.client import estimate_cost_yuan

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen-plus"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRequest:
    """LLM 请求执行的一次工具调用。

    Attributes:
        name: 工具名
        arguments: 已解析的参数 dict（JSON 解析失败时为 {"_raw": 原始字符串}）
        call_id: 工具调用 ID（DashScope 返回；mock 下自造）
    """

    name: str
    arguments: Dict[str, Any]
    call_id: str = ""


@dataclass
class ToolCallingResponse:
    """带 function calling 的 LLM 响应。

    Attributes:
        content: LLM 文本输出（思考 / 最终答案，可为空）
        tool_calls: 本轮请求执行的工具调用列表（空表示已给最终答案）
        model: 模型名
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        cost_yuan: 本次调用成本（元）
        error: 错误信息（空串表示成功）
    """

    content: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseToolCallingLLM(ABC):
    """带工具调用能力的 LLM 抽象基类。

    设计目的：simple 循环 / LangGraph 编排可在 DashScope 与 mock 之间热插拔，
    并保证测试全程离线。
    """

    @abstractmethod
    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolCallingResponse:
        """执行一次带 tools 的对话。

        Args:
            messages: OpenAI 格式消息列表（system/user/assistant/tool）
            tools: OpenAI tools schema 列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        Returns:
            ToolCallingResponse（任何失败都以 error 字段返回，不抛异常）
        """
        ...

    @property
    def available(self) -> bool:
        """后端是否可用（默认 True；真实客户端按 key / 依赖判断）"""
        return True


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """从 dict / 对象属性中取值（兼容 DashScope 响应的两种形态）"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage(resp: Any) -> Tuple[int, int]:
    """从 DashScope 响应中提取 (input_tokens, output_tokens)"""
    usage = _get_field(resp, "usage", None) or {}
    input_tokens = _get_field(usage, "input_tokens", 0) or 0
    output_tokens = _get_field(usage, "output_tokens", 0) or 0
    try:
        return int(input_tokens), int(output_tokens)
    except (TypeError, ValueError):
        return 0, 0


def _tool_spec_names(tools: List[Dict[str, Any]]) -> Set[str]:
    """从 OpenAI tools schema 中取出全部工具名"""
    names: Set[str] = set()
    for spec in tools or []:
        function = _get_field(spec, "function", None) or {}
        name = _get_field(function, "name", "")
        if name:
            names.add(str(name))
    return names


def _message_text(message: Any) -> str:
    """取消息 content 的字符串形态（多模态 content 列表时拼接文本）"""
    content = _get_field(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _get_field(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    """取最后一条 user 消息文本（没有则返回空串）"""
    for message in reversed(messages or []):
        if _get_field(message, "role", "") == "user":
            return _message_text(message)
    return ""


def _collect_tool_messages(messages: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """收集已执行的工具观察值：[(tool_name, parsed_result), ...]"""
    observations: List[Tuple[str, Dict[str, Any]]] = []
    for message in messages or []:
        if _get_field(message, "role", "") != "tool":
            continue
        name = str(_get_field(message, "name", "") or "")
        raw = _message_text(message)
        data: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw) if raw else {}
            if isinstance(parsed, dict):
                data = parsed
        except (json.JSONDecodeError, TypeError):
            data = {"_raw": raw}
        observations.append((name, data))
    return observations


# ---------------------------------------------------------------------------
# DashScope Function Calling 客户端
# ---------------------------------------------------------------------------


class DashScopeToolCallingLLM(BaseToolCallingLLM):
    """DashScope Qwen-Plus 的 function calling 客户端。

    真实调用 ``dashscope.Generation.call(..., tools=..., tool_choice="auto",
    result_format="message")``；所有异常都转成 ``ToolCallingResponse(error=...)``。
    """

    # 明确标记：非 mock 模式（供 AgentRunner 组装 AgentResult.mock_mode）
    mock_mode = False

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._client = None

    @property
    def available(self) -> bool:
        """有 API key 且能 import dashscope 时才可用（不发起网络请求）"""
        if not self.api_key:
            return False
        try:
            import dashscope  # noqa: F401
        except ImportError:
            logger.warning("dashscope 包未安装，DashScopeToolCallingLLM 不可用")
            return False
        return True

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolCallingResponse:
        """调用 DashScope 带 tools 的对话接口（失败返回 error 不抛异常）"""
        if not self.available:
            return ToolCallingResponse(
                model=self.model,
                error="DashScope 不可用：未配置 DASHSCOPE_API_KEY 或未安装 dashscope SDK",
            )
        try:
            import dashscope

            dashscope.api_key = self.api_key
            self._client = dashscope
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "result_format": "message",
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = dashscope.Generation.call(**kwargs)
        except Exception as exc:  # noqa: BLE001 - LLM 异常统一转 error
            logger.warning("DashScope function calling 调用失败: %s", exc)
            return ToolCallingResponse(
                model=self.model,
                error=f"DashScope 调用失败: {type(exc).__name__}: {exc}",
            )
        return self._parse_response(resp)

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(self, resp: Any) -> ToolCallingResponse:
        """解析 DashScope GenerationResponse（兼容 dict 与对象两种形态）"""
        status_code = _get_field(resp, "status_code", 200)
        if status_code != 200:
            code = _get_field(resp, "code", "")
            message = _get_field(resp, "message", "")
            return ToolCallingResponse(
                model=self.model,
                error=(
                    f"DashScope API 错误: status={status_code}, "
                    f"code={code}, msg={message}"
                ),
            )

        input_tokens, output_tokens = _extract_usage(resp)
        cost_yuan = estimate_cost_yuan(self.model, input_tokens, output_tokens)
        output = _get_field(resp, "output", None)
        choices = _get_field(output, "choices", None) or []
        message = _get_field(choices[0], "message", None) if choices else None
        if message is None:
            # 兜底：兼容 result_format="text" 的返回
            text = _get_field(output, "text", "") or ""
            return ToolCallingResponse(
                content=str(text),
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_yuan=cost_yuan,
                error="" if text else "DashScope 响应缺少 message 内容",
            )

        content = _message_text(message)
        raw_calls = _get_field(message, "tool_calls", None) or []
        tool_calls: List[ToolCallRequest] = []
        parse_errors: List[str] = []
        for index, raw_call in enumerate(raw_calls):
            function = _get_field(raw_call, "function", None) or {}
            name = str(_get_field(function, "name", "") or "")
            call_id = str(_get_field(raw_call, "id", "") or "") or f"call_{index}"
            arguments, error = self._parse_arguments(_get_field(function, "arguments", ""))
            if error:
                parse_errors.append(error)
            if not name:
                parse_errors.append(f"第 {index} 个工具调用缺少 name")
                continue
            tool_calls.append(
                ToolCallRequest(name=name, arguments=arguments, call_id=call_id)
            )
        return ToolCallingResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_yuan=cost_yuan,
            error="；".join(parse_errors),
        )

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> Tuple[Dict[str, Any], str]:
        """把工具调用 arguments 解析为 dict（失败时返回 {"_raw": 原文} 并带 error）"""
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments), ""
        if raw_arguments is None or raw_arguments == "":
            return {}, ""
        if not isinstance(raw_arguments, str):
            return {"_raw": str(raw_arguments)}, ""
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {"_raw": raw_arguments}, f"工具调用参数不是合法 JSON: {exc}"
        if isinstance(parsed, Mapping):
            return dict(parsed), ""
        return {"_raw": raw_arguments}, "工具调用参数必须是 JSON 对象"


# ---------------------------------------------------------------------------
# 脚本化 Mock LLM（确定性多步策略，绝不真实调用 API）
# ---------------------------------------------------------------------------

# 问题分类关键词（按优先级：翻译 > 综述 > 检索 > 默认）
_TRANSLATE_KEYWORDS = ("翻译", "translate")
_REVIEW_KEYWORDS = ("综述", "review", "总结多篇", "文献总结")
_SEARCH_KEYWORDS = (
    "数据集", "模型", "实验", "公式", "方法", "结论",
    "BLEU", "mAP", "准确率", "token", "参数",
    "dataset", "model", "experiment", "formula", "method", "conclusion",
    "accuracy", "parameter", "benchmark",
)

# 从问题中抽取引号内文本（翻译任务优先翻译引号内容）
_QUOTED_PATTERNS = (
    re.compile(r"「([^」]+)」"),
    re.compile(r"『([^』]+)』"),
    re.compile(r"“([^”]+)”"),
    re.compile(r"\"([^\"]+)\""),
    re.compile(r"'([^']+)'"),
)

_NO_DOC_ANSWER = "当前会话没有已解析文档，请先上传 PDF"


def _classify_question(question: str) -> str:
    """把问题分为 translate / review / search / default 四类"""
    lower = question.lower()
    if any(keyword.lower() in lower for keyword in _TRANSLATE_KEYWORDS):
        return "translate"
    if any(keyword.lower() in lower for keyword in _REVIEW_KEYWORDS):
        return "review"
    if any(keyword.lower() in lower for keyword in _SEARCH_KEYWORDS):
        return "search"
    return "default"


def _extract_quoted_text(question: str) -> str:
    """抽取问题中引号内的文本（无引号返回空串）"""
    for pattern in _QUOTED_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1).strip()
    return ""


class ScriptedMockLLM(BaseToolCallingLLM):
    """确定性脚本化 mock LLM。

    策略：按「最后一条 user 消息 + 已执行工具次数」决定下一步，保证离线可走完
    多步 ReAct 循环并给出最终答案（翻译 / 综述 / 片段检索 / 文档列表）。
    不发起任何网络请求，cost_yuan 恒为 0。
    """

    mock_mode = True

    def __init__(self, model: str = "mock-qwen-plus"):
        self.model = model

    @property
    def available(self) -> bool:
        """mock 恒可用"""
        return True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolCallingResponse:
        """按确定性脚本返回下一轮（思考 + 工具调用）或最终答案"""
        available = _tool_spec_names(tools)
        question = _last_user_message(messages)
        observations = _collect_tool_messages(messages)
        content, tool_calls = self._decide(question, observations, available)
        input_tokens = max(
            1,
            sum(len(_message_text(message)) for message in messages or []) // 4,
        )
        output_tokens = max(1, len(content) // 4)
        return ToolCallingResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_yuan=0.0,
        )

    # ------------------------------------------------------------------
    # 策略分发
    # ------------------------------------------------------------------

    def _decide(
        self,
        question: str,
        observations: List[Tuple[str, Dict[str, Any]]],
        available: Set[str],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """根据问题类别与已执行工具返回 (content, tool_calls)"""
        category = _classify_question(question)
        if category == "translate":
            return self._decide_translate(question, observations, available)
        if category == "review":
            return self._decide_review(question, observations, available)
        if category == "search":
            return self._decide_search(question, observations, available)
        return self._decide_default(question, observations, available)

    # ------------------------------------------------------------------
    # 分策略实现
    # ------------------------------------------------------------------

    def _decide_translate(
        self,
        question: str,
        observations: List[Tuple[str, Dict[str, Any]]],
        available: Set[str],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """翻译策略：首轮调 translate_text，次轮引用译文给出最终答案"""
        if "translate_text" not in available:
            return self._final("当前没有可用的翻译工具，无法完成翻译任务。")
        if "translate_text" not in self._called_names(observations):
            text = _extract_quoted_text(question) or question.strip()
            return self._call_tool(
                "translate_text",
                {"text": text, "target_lang": "zh-CN"},
                "需要调用翻译工具处理学术文本",
                observations,
            )
        data = self._latest_observation(observations, "translate_text") or {}
        translated = str(data.get("translated", "") or "")
        aligned_terms = data.get("aligned_terms") or []
        if not translated:
            return self._final("翻译工具未能返回译文，请检查待翻译文本后重试。")
        answer = f"翻译结果：{translated}"
        if aligned_terms:
            answer += "（对齐术语：" + "、".join(str(term) for term in aligned_terms[:5]) + "）"
        return self._final(answer)

    def _decide_review(
        self,
        question: str,
        observations: List[Tuple[str, Dict[str, Any]]],
        available: Set[str],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """综述策略：list_documents（取 doc_ids）→ generate_review → 最终答案"""
        if "generate_review" not in available:
            return self._final("当前没有可用的综述生成工具，无法完成综述任务。")
        called = self._called_names(observations)
        if "generate_review" in called:
            data = self._latest_observation(observations, "generate_review") or {}
            title = str(data.get("title", "") or "")
            full_text = str(data.get("full_text", "") or "")
            review_error = str(data.get("error", "") or "")
            if full_text:
                head = full_text[:200].replace("\n", " ")
                return self._final(f"《{title or '综述'}》已生成。{head}")
            if review_error:
                return self._final(f"综述生成失败：{review_error}")
            return self._final("综述生成失败：工具未返回正文，请检查文献是否已解析。")
        if "list_documents" in called:
            documents = self._latest_documents(observations)
            doc_ids = self._extract_doc_ids(documents)
            if not doc_ids:
                return self._final(_NO_DOC_ANSWER)
            return self._call_tool(
                "generate_review",
                {"topic": question, "doc_ids": doc_ids[:3], "num_sections": 3},
                "需要基于已解析文档生成综述",
                observations,
            )
        if "list_documents" in available:
            return self._call_tool(
                "list_documents",
                {},
                "先确认有哪些已解析文档可用于综述",
                observations,
            )
        # 没有 list_documents 工具：直接基于全部文献生成
        return self._call_tool(
            "generate_review",
            {"topic": question, "doc_ids": [], "num_sections": 3},
            "基于当前已解析文献生成综述",
            observations,
        )

    def _decide_search(
        self,
        question: str,
        observations: List[Tuple[str, Dict[str, Any]]],
        available: Set[str],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """检索策略：先检索片段，次轮用前 2 个片段 + [chunk_id] 引用作答"""
        if "search_chunks" not in available:
            return self._final("当前没有可用的检索工具，无法基于论文事实回答。")
        if "search_chunks" not in self._called_names(observations):
            return self._call_tool(
                "search_chunks",
                {"query": question, "top_k": 5},
                "需要先检索论文片段",
                observations,
            )
        data = self._latest_observation(observations, "search_chunks") or {}
        chunks = data.get("chunks") or []
        parts: List[str] = []
        for chunk in chunks[:2]:
            if not isinstance(chunk, Mapping):
                continue
            text = str(chunk.get("text", "") or "").replace("\n", " ")[:120]
            chunk_id = str(chunk.get("chunk_id", "") or "")
            if not text and not chunk_id:
                continue
            parts.append(f"{text}[{chunk_id}]" if chunk_id else text)
        if parts:
            return self._final("根据检索到的论文片段：" + "；".join(parts))
        return self._final(
            "检索工具执行失败或未返回相关片段，暂时无法基于论文事实回答。"
            "请确认论文已解析后重试。"
        )

    def _decide_default(
        self,
        question: str,
        observations: List[Tuple[str, Dict[str, Any]]],
        available: Set[str],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """默认策略：列出文档，再给出会话文档清单 / 无文档提示"""
        if "list_documents" not in available:
            return self._final("当前没有可用的文档列表工具，无法列出会话文档。")
        if "list_documents" not in self._called_names(observations):
            return self._call_tool(
                "list_documents",
                {},
                "先查看会话中已解析的文档",
                observations,
            )
        documents = self._latest_documents(observations)
        if not documents:
            return self._final(_NO_DOC_ANSWER)
        titles: List[str] = []
        for document in documents[:10]:
            if isinstance(document, Mapping):
                title = str(document.get("title") or document.get("doc_id") or "未命名文档")
            else:
                title = str(document)
            titles.append(title)
        return self._final(
            f"当前会话共有 {len(documents)} 篇已解析文档：" + "、".join(titles)
            + "。可以针对其中任意一篇提问，或让我生成综述。"
        )

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------

    @staticmethod
    def _called_names(observations: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """已执行工具名列表（按执行顺序）"""
        return [name for name, _ in observations]

    @staticmethod
    def _latest_observation(
        observations: List[Tuple[str, Dict[str, Any]]],
        name: str,
    ) -> Dict[str, Any]:
        """取某工具最后一次的观察值（没有返回 {}）"""
        for tool_name, data in reversed(observations):
            if tool_name == name:
                return data
        return {}

    def _latest_documents(self, observations: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        """取 list_documents 最后一次返回的文档列表"""
        data = self._latest_observation(observations, "list_documents")
        documents = data.get("documents") if isinstance(data, Mapping) else None
        return list(documents) if isinstance(documents, list) else []

    @staticmethod
    def _extract_doc_ids(documents: List[Any]) -> List[str]:
        """从文档列表里提取 doc_id（去重保序）"""
        doc_ids: List[str] = []
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            doc_id = str(document.get("doc_id") or document.get("id") or "").strip()
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        return doc_ids

    def _call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        thought: str,
        observations: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[str, List[ToolCallRequest]]:
        """构造一轮工具调用响应"""
        call_id = f"mock_call_{len(observations) + 1}_{name}"
        return thought, [ToolCallRequest(name=name, arguments=arguments, call_id=call_id)]

    @staticmethod
    def _final(content: str) -> Tuple[str, List[ToolCallRequest]]:
        """构造最终答案响应（无工具调用）"""
        return content, []


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_tool_calling_llm(mode: str = "auto") -> BaseToolCallingLLM:
    """按模式创建带工具调用能力的 LLM。

    Args:
        mode: "auto" / "dashscope" / "mock"
            - auto：有 DASHSCOPE_API_KEY 且 dashscope 可 import → DashScope，否则 mock
            - dashscope：强制 DashScope（不可用时仍返回客户端，调用时报 error）
            - mock：脚本化 mock
    Returns:
        BaseToolCallingLLM 实例
    """
    normalized = (mode or "auto").lower()
    if normalized == "mock":
        return ScriptedMockLLM()
    if normalized == "dashscope":
        return DashScopeToolCallingLLM()
    if normalized == "auto":
        llm = DashScopeToolCallingLLM()
        if llm.available:
            return llm
        logger.info("auto 模式：DashScope 不可用，使用 ScriptedMockLLM")
        return ScriptedMockLLM()
    raise ValueError(f"未知 tool-calling LLM 模式: {mode}")

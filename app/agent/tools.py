"""Agent 工具注册表（冻结契约 v1）。

提供 4 个默认工具（工具名固定，前后端 / MCP server 已按此对接）：
1. ``list_documents``  —— 列出当前会话已解析文档
2. ``search_chunks``   —— 在论文 chunk 中做语义/关键词检索
3. ``translate_text``  —— 学术文本翻译（带术语对齐）
4. ``generate_review`` —— 多文献综述生成

设计原则：
- 工具通过回调注入（``create_default_tools``），Agent 核心不直接依赖 RAG / 翻译 /
  综述实现，便于单测注入假数据、API 层注入真实管道；
- ``ToolRegistry.call`` 对未知工具 / 参数校验失败 / 回调抛异常统一转成
  ``ToolExecution.error``，绝不向上抛异常（LLM 可继续基于错误信息决策）；
- ``default_tool_specs`` 返回静态 OpenAI tools 格式，供 MCP server 等无回调场景使用。
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .types import ToolExecution, _json_safe

# 工具结果里综述全文的最大保留字符数（避免 prompt 爆炸）
REVIEW_FULL_TEXT_MAX_CHARS = 1500


# ---------------------------------------------------------------------------
# 静态工具 schema（OpenAI tools 格式）
# ---------------------------------------------------------------------------

_DEFAULT_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "列出当前会话中已解析的论文文档（返回 doc_id、标题、页数等元信息）。"
                "当用户询问“有哪些论文/文档”、需要确认是否有可检索文档，"
                "或生成综述前需要拿到文档列表时调用。无需参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "在已解析论文的正文片段（chunk）中检索与问题相关的内容。"
                "当问题涉及数据集、模型、实验、公式、方法、结论、指标（BLEU/mAP/准确率）、"
                "token、参数等论文事实时，必须先调用本工具获取片段，禁止凭空编造。"
                "返回每个片段的 chunk_id / doc_id / text / section / score。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词或完整问题（中文/英文均可）",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "限定检索的文档 ID；为空串表示在所有已解析文档中检索",
                        "default": "",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数量上限，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": (
                "把英文学术文本翻译成目标语言（默认简体中文），并做专业术语对齐。"
                "当用户要求翻译论文段落、摘要或术语时调用；返回 translated / "
                "aligned_terms / target_lang / mock_mode。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待翻译原文（建议单段，避免过长）",
                    },
                    "target_lang": {
                        "type": "string",
                        "description": "目标语言代码，默认 zh-CN（简体中文）",
                        "default": "zh-CN",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_review",
            "description": (
                "针对给定主题、基于指定文献生成结构化中文综述。当用户要求“写综述/总结多篇文献/"
                "对比多篇论文”时调用。返回 title / full_text / references / citation_stats；"
                "full_text 过长时截断并置 full_text_truncated=true。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "综述主题（中文，尽量具体）",
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参与综述的文档 ID 列表（可为空，表示使用全部已解析文档）",
                        "default": [],
                    },
                    "num_sections": {
                        "type": "integer",
                        "description": "期望章节数，默认 3",
                        "default": 3,
                    },
                },
                "required": ["topic"],
            },
        },
    },
]


def default_tool_specs() -> List[Dict[str, Any]]:
    """返回 4 个默认工具的静态 OpenAI tools schema（深拷贝，调用方可安全修改）。

    Returns:
        每个元素形如 {"type": "function", "function": {"name", "description", "parameters"}}
    """
    return copy.deepcopy(_DEFAULT_SPECS)


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

_TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> str:
    """按 JSON Schema 校验参数。

    Args:
        schema: 工具参数 JSON Schema（type=object）
        arguments: LLM 给出的参数字典
    Returns:
        错误信息（空串表示校验通过）
    """
    if not isinstance(arguments, dict):
        return f"参数必须是 JSON 对象，实际为 {type(arguments).__name__}"
    properties: Dict[str, Any] = schema.get("properties", {}) or {}
    required: List[str] = schema.get("required", []) or []
    for key in required:
        if key not in arguments:
            return f"缺少必填参数: {key}"
    for key, value in arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            # 容忍 LLM 传入 schema 之外的多余参数（不阻塞工具执行）
            continue
        expected = prop.get("type")
        checker = _TYPE_CHECKS.get(expected) if isinstance(expected, str) else None
        if checker is not None and not checker(value):
            return f"参数 {key} 类型错误：期望 {expected}，实际 {type(value).__name__}"
    return ""


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------


@dataclass
class _Tool:
    """内部工具定义（名字 + schema + 执行回调）"""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    """Agent 工具注册表。

    职责：
    - ``specs()`` 导出 OpenAI tools 格式给 LLM / MCP；
    - ``call(name, arguments)`` 统一执行并归一为 ``ToolExecution``（异常不向上抛）。

    用法：
        registry = create_default_tools(...)
        specs = registry.specs()
        execution = registry.call("search_chunks", {"query": "BLEU", "top_k": 3})
        execution.result["chunks"]  # 归一化后的 chunk 列表
    """

    def __init__(self, tools: Optional[List[_Tool]] = None):
        self._tools: Dict[str, _Tool] = {}
        for tool in tools or []:
            self.register(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=tool.handler,
            )

    # ------------------------------------------------------------------
    # 注册 / 导出
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """注册（或覆盖）一个工具"""
        self._tools[name] = _Tool(
            name=name,
            description=description,
            parameters=copy.deepcopy(parameters),
            handler=handler,
        )

    @property
    def names(self) -> List[str]:
        """已注册工具名列表（按注册顺序）"""
        return list(self._tools.keys())

    def specs(self) -> List[Dict[str, Any]]:
        """导出 OpenAI tools 格式的 schema 列表（深拷贝，避免外部修改内部状态）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": copy.deepcopy(tool.parameters),
                },
            }
            for tool in self._tools.values()
        ]

    def has(self, name: str) -> bool:
        """工具是否已注册"""
        return name in self._tools

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolExecution:
        """执行一次工具调用（所有异常都转成 ToolExecution.error，不向上抛）。

        Args:
            name: 工具名
            arguments: LLM 给出的参数字典
        Returns:
            ToolExecution（ok=False 时 error 记录失败原因）
        """
        exec_arguments: Dict[str, Any] = arguments if isinstance(arguments, dict) else {}
        execution = ToolExecution(name=name, arguments=exec_arguments)
        t0 = time.perf_counter()
        try:
            tool = self._tools.get(name)
            if tool is None:
                available = "、".join(self._tools.keys()) or "（空）"
                execution.error = f"未知工具: {name}（可用工具: {available}）"
                return execution
            error = _validate_arguments(tool.parameters, arguments)
            if error:
                execution.error = f"参数校验失败: {error}"
                return execution
            raw = tool.handler(dict(arguments))
            if isinstance(raw, Mapping):
                result: Dict[str, Any] = dict(raw)
            else:
                result = {"result": raw}
            execution.result = _json_safe(result)
        except Exception as exc:  # noqa: BLE001 - 工具异常统一降级为 error
            execution.error = f"工具执行异常: {type(exc).__name__}: {exc}"
        finally:
            execution.latency_ms = int((time.perf_counter() - t0) * 1000)
        return execution


# ---------------------------------------------------------------------------
# 内部：结果归一化工具
# ---------------------------------------------------------------------------


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """从 dict 或对象属性中取值（兼容 dict / dataclass / 普通对象）"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """安全转 float（失败时返回 default）"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> Dict[str, Any]:
    """把回调返回的文档信息统一转成 dict"""
    if isinstance(value, Mapping):
        return _json_safe(dict(value))
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(value)
    if hasattr(value, "__dict__"):
        return _json_safe(
            {k: v for k, v in vars(value).items() if not str(k).startswith("_")}
        )
    return {"value": _json_safe(value)}


def _normalize_chunk(item: Any) -> Dict[str, Any]:
    """把 chunk 回调返回项归一化为 {chunk_id, doc_id, text, section, score}。

    兼容三种形态：
    - dict：{"chunk_id": ..., "text": ..., "score": ...}
    - RetrievalResult：带 ``.chunk`` 与 ``.score``
    - Chunk 等带同名属性的对象
    """
    score: Any = 0.0
    if isinstance(item, Mapping):
        chunk: Any = item
        score = item.get("score", 0.0)
    elif hasattr(item, "chunk") and not hasattr(item, "chunk_id"):
        # RetrievalResult（app.rag.retriever）
        chunk = item.chunk
        score = getattr(item, "score", 0.0)
    else:
        chunk = item
        score = getattr(item, "score", 0.0)
    return {
        "chunk_id": str(_get_field(chunk, "chunk_id", "") or ""),
        "doc_id": str(_get_field(chunk, "doc_id", "") or ""),
        "text": str(_get_field(chunk, "text", "") or ""),
        "section": str(_get_field(chunk, "section", "") or ""),
        "score": _as_float(score),
    }


# ---------------------------------------------------------------------------
# 4 个默认工具的处理函数
# ---------------------------------------------------------------------------


def _handle_list_documents(
    arguments: Dict[str, Any],
    list_documents: Callable[[], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """list_documents：调用回调并包装为 {"documents": [...]}"""
    documents = list_documents() or []
    return {"documents": [_as_dict(doc) for doc in documents]}


def _handle_search_chunks(
    arguments: Dict[str, Any],
    search_chunks: Callable[[str, str, int], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """search_chunks：调用回调并把 chunk 统一包装为 {"chunks": [...]}"""
    query = str(arguments["query"])
    doc_id = str(arguments.get("doc_id", "") or "")
    top_k = int(arguments.get("top_k", 5))
    results = search_chunks(query, doc_id, top_k) or []
    return {"chunks": [_normalize_chunk(item) for item in results]}


def _handle_translate_text(
    arguments: Dict[str, Any],
    translate_text: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """translate_text：调用回调并统一结果字段"""
    text = str(arguments["text"])
    target_lang = str(arguments.get("target_lang", "zh-CN") or "zh-CN")
    raw = translate_text(text, target_lang)
    if not isinstance(raw, Mapping):
        # 兼容 TranslationUnit 等对象
        raw = {
            "translated": getattr(raw, "target_text", "") or "",
            "aligned_terms": list(getattr(raw, "aligned_terms", []) or []),
            "target_lang": getattr(raw, "target_lang", target_lang),
            "mock_mode": bool(getattr(raw, "from_mock", False)),
        }
    aligned_terms = [str(term) for term in (raw.get("aligned_terms") or [])]
    return {
        "translated": str(raw.get("translated", "") or ""),
        "aligned_terms": aligned_terms,
        "target_lang": str(raw.get("target_lang", target_lang) or target_lang),
        "mock_mode": bool(raw.get("mock_mode", False)),
    }


def _handle_generate_review(
    arguments: Dict[str, Any],
    generate_review: Callable[[str, List[str], int], Dict[str, Any]],
) -> Dict[str, Any]:
    """generate_review：调用回调，full_text 截断到 1500 字以内避免 prompt 爆炸"""
    topic = str(arguments["topic"])
    doc_ids = [str(doc_id) for doc_id in (arguments.get("doc_ids") or [])]
    num_sections = int(arguments.get("num_sections", 3))
    raw = generate_review(topic, doc_ids, num_sections)
    if not isinstance(raw, Mapping):
        # 兼容 ReviewDocument 等对象
        raw = {
            "title": getattr(raw, "title", "") or "",
            "full_text": getattr(raw, "full_text", "") or "",
            "references": getattr(raw, "references", []) or [],
            "citation_stats": getattr(raw, "citation_stats", {}) or {},
            "mock_mode": bool(getattr(raw, "mock_mode", False)),
            "error": getattr(raw, "error", "") or "",
        }
    full_text = str(raw.get("full_text", "") or "")
    truncated = len(full_text) > REVIEW_FULL_TEXT_MAX_CHARS
    return {
        "title": str(raw.get("title", "") or ""),
        "full_text": full_text[:REVIEW_FULL_TEXT_MAX_CHARS],
        "full_text_truncated": truncated,
        "references": _json_safe(raw.get("references") or []),
        "citation_stats": _json_safe(dict(raw.get("citation_stats") or {})),
        "mock_mode": bool(raw.get("mock_mode", False)),
        "error": str(raw.get("error", "") or ""),
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_default_tools(
    list_documents: Callable[[], List[Dict[str, Any]]],
    search_chunks: Callable[[str, str, int], List[Dict[str, Any]]],
    translate_text: Callable[[str, str], Dict[str, Any]],
    generate_review: Callable[[str, List[str], int], Dict[str, Any]],
) -> ToolRegistry:
    """创建包含 4 个默认工具的 ToolRegistry。

    Args:
        list_documents: 无参回调，返回文档信息列表
        search_chunks: (query, doc_id, top_k) 位置参数回调，返回 chunk 列表
        translate_text: (text, target_lang) 位置参数回调，返回翻译结果 dict
        generate_review: (topic, doc_ids, num_sections) 位置参数回调，返回综述结果 dict
    Returns:
        ToolRegistry（specs/names/call 均可用）
    """
    callbacks = {
        "list_documents": list_documents,
        "search_chunks": search_chunks,
        "translate_text": translate_text,
        "generate_review": generate_review,
    }
    for name, callback in callbacks.items():
        if not callable(callback):
            raise TypeError(f"{name} 回调必须是可调用对象，实际为 {type(callback).__name__}")

    handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        "list_documents": lambda args: _handle_list_documents(args, list_documents),
        "search_chunks": lambda args: _handle_search_chunks(args, search_chunks),
        "translate_text": lambda args: _handle_translate_text(args, translate_text),
        "generate_review": lambda args: _handle_generate_review(args, generate_review),
    }
    tools: List[_Tool] = []
    for spec in _DEFAULT_SPECS:
        function = spec["function"]
        name = function["name"]
        tools.append(
            _Tool(
                name=name,
                description=function["description"],
                parameters=function["parameters"],
                handler=handlers[name],
            )
        )
    return ToolRegistry(tools)

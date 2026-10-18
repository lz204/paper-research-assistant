"""Agent 核心数据结构（冻结契约 v1）。

本模块只包含纯数据类与 JSON 安全化工具，不依赖 LangGraph / DashScope / API 层，
便于 simple 循环、LangGraph 编排、app.api 与前端共享同一份结果结构。

字段命名严格遵循冻结契约 v1：前后端 / MCP server 已按 ``AgentResult.to_dict()``
的 key 对接，禁止随意增删字段。

主要结构：
- ToolExecution：单次工具调用的执行记录（结果 / 错误 / 耗时）
- AgentStep：Agent 的一轮决策（LLM 本轮思考 + 本轮工具调用）
- AgentResult：一次 Agent 问答的完整结果（答案 / 步骤 / 引用 / 成本 / 元信息）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# 扁平工具调用记录里 result_preview 的最大字符数（避免 API 响应过大）
RESULT_PREVIEW_MAX_CHARS = 300
# 引用文本预览的最大字符数（契约：text_preview 前 200 字）
CITATION_PREVIEW_MAX_CHARS = 200


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """把任意对象递归转换成为 JSON-safe 的 plain dict / list / 标量。

    Args:
        value: 任意 Python 对象（可能来自工具回调：dict / dataclass / Chunk 等）
        _depth: 递归深度（防御性上限，避免异常对象自引用导致无限递归）
    Returns:
        仅含 dict / list / str / int / float / bool / None 的结构
    """
    if _depth > 8:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth + 1) for v in value]
    # dataclass 实例（Chunk / RetrievalResult 等）
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _json_safe(getattr(value, key), _depth + 1)
            for key in value.__dataclass_fields__
        }
    # 普通对象：转其 __dict__
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_safe(v, _depth + 1)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    # Mapping-like（自定义映射类型兜底）
    if hasattr(value, "items"):
        try:
            return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
        except Exception:  # noqa: BLE001 - 兜底：不可迭代的映射转字符串
            return str(value)
    return str(value)


@dataclass
class ToolExecution:
    """单次工具调用记录。

    Attributes:
        name: 工具名（list_documents / search_chunks / translate_text / generate_review）
        arguments: 实际传给工具的参数字典（已做 doc_id 补全等归一化）
        result: 工具返回的结构化结果（失败时为空 dict）
        error: 失败原因（空串表示成功）
        latency_ms: 工具执行耗时（毫秒）
    """

    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: int = 0
    # 对应的 LLM tool_call id（追加 role=tool 消息时必须回填，否则模型无法
    # 把工具结果关联到本次调用，会重复调用同一工具；不进 to_dict 契约）
    call_id: str = ""

    @property
    def ok(self) -> bool:
        """工具调用是否成功（无 error 即成功）"""
        return not self.error

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON-safe 的 plain dict"""
        return {
            "name": self.name,
            "arguments": _json_safe(self.arguments),
            "result": _json_safe(self.result),
            "error": self.error,
            "latency_ms": self.latency_ms,
            "ok": self.ok,
        }


@dataclass
class AgentStep:
    """Agent 的一轮决策记录。

    Attributes:
        index: 轮次序号（从 1 开始）
        thought: 本轮 LLM 返回的 content（推理/思考文本，可为空）
        tool_calls: 本轮执行的工具调用列表（最终答案轮为空）
        latency_ms: 本轮 LLM 调用耗时（毫秒）
        input_tokens: 本轮输入 token 数
        output_tokens: 本轮输出 token 数
        cost_yuan: 本轮成本（元）
    """

    index: int
    thought: str = ""
    tool_calls: List[ToolExecution] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON-safe 的 plain dict"""
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_yuan": self.cost_yuan,
        }


@dataclass
class AgentResult:
    """一次 Agent 问答的完整结果（契约 v1，前后端 / MCP 共用）。

    Attributes:
        answer: 最终答案（中文；失败时可能为空）
        steps: 每轮决策记录
        tool_calls: 扁平工具调用记录
            {step, name, arguments, result_preview, latency_ms, ok}
            （result_preview 为 JSON 字符串预览，便于前端直接展示）
        citations: 引用列表 {chunk_id, doc_id, text_preview}
        trace_id: 本次运行追踪 ID（16 位 hex）
        model: LLM 模型名
        input_tokens: 累计输入 token
        output_tokens: 累计输出 token
        cost_yuan: 累计成本（元）
        mock_mode: 是否为 mock 模式（未真实调用 DashScope）
        framework: 编排方式，"langgraph" 或 "simple"
        latency_ms: 端到端耗时（毫秒）
        error: 错误信息（空串表示无错误；工具失败不写入此字段）
    """

    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    mock_mode: bool = True
    framework: str = "simple"
    latency_ms: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON-safe 的 plain dict（键与冻结契约 v1 完全一致）"""
        return {
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "tool_calls": _json_safe(self.tool_calls),
            "citations": _json_safe(self.citations),
            "trace_id": self.trace_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_yuan": self.cost_yuan,
            "mock_mode": self.mock_mode,
            "framework": self.framework,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

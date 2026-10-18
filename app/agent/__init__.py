"""论文研读助手 Agent 核心包（冻结契约 v1）。

对外统一导出数据结构、工具注册表、LLM 抽象与编排器，供 app.api / app.frontend /
MCP server 复用：

- 数据结构：AgentResult / AgentStep / ToolExecution
- 工具注册：ToolRegistry / create_default_tools / default_tool_specs
  （4 个工具：list_documents / search_chunks / translate_text / generate_review）
- LLM 抽象：BaseToolCallingLLM / ToolCallingResponse / ToolCallRequest /
  DashScopeToolCallingLLM / ScriptedMockLLM / create_tool_calling_llm
- 编排：AgentRunner / create_agent_runner（simple ReAct 循环，auto 优先 LangGraph）

设计约束：
- 本包只依赖 app.llm（成本估算）等核心模块，**不 import app.api / app.frontend**，
  避免 API ↔ Agent 循环依赖；
- LangGraph 编排位于 ``app.agent.langgraph_runner``，由 AgentRunner 延迟 import
  （缺包或构图异常时自动回退 simple 并把 framework 置 "simple"），因此本包 import
  不会因 langgraph 缺失而失败；
- 测试 / 无 API key 时 ``create_tool_calling_llm("auto")`` 只会返回 ScriptedMockLLM，
  不会发起真实 DashScope 请求。
"""

from .llm import (
    BaseToolCallingLLM,
    DashScopeToolCallingLLM,
    ScriptedMockLLM,
    ToolCallRequest,
    ToolCallingResponse,
    create_tool_calling_llm,
)
from .runner import AgentRunner, create_agent_runner
from .tools import ToolRegistry, create_default_tools, default_tool_specs
from .types import AgentResult, AgentStep, ToolExecution

__all__ = [
    "AgentRunner",
    "create_agent_runner",
    "ToolRegistry",
    "ToolExecution",
    "AgentStep",
    "AgentResult",
    "create_default_tools",
    "default_tool_specs",
    "BaseToolCallingLLM",
    "ToolCallingResponse",
    "ToolCallRequest",
    "DashScopeToolCallingLLM",
    "ScriptedMockLLM",
    "create_tool_calling_llm",
]

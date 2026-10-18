"""LangGraph 编排实现（冻结契约 v1）。

用 ``langgraph.graph.StateGraph`` 实现 agent ↔ tools 条件图：
    START → agent ──(无 tool_calls / error / 已有答案)──→ END
                 └──(有 tool_calls)──→ tools ──(达到最大步数)──→ finish → END
                                              └──(还有步数)──→ agent

与 simple 循环共享 ``AgentRunner`` 的
``_call_llm / _execute_tool_calls / _assistant_tool_call_message /
_append_tool_messages / _max_steps_answer / _finalize``，保证两种编排的
``AgentResult`` 结构与指标完全一致。

任何 LangGraph 导入 / 构图 / 运行异常都会由 ``AgentRunner.run`` 捕获并自动回退
simple 循环（framework 置 "simple"），因此本模块不直接对外暴露给 API 层。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .runner import AgentRunner
from .types import AgentResult, AgentStep, ToolExecution

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """LangGraph 状态定义（契约要求至少含 messages / steps / tool_calls / final_answer / error）。"""

    messages: List[Dict[str, Any]]
    steps: List[AgentStep]
    tool_calls: List[Dict[str, Any]]  # 扁平工具调用记录（与 AgentResult.tool_calls 对应）
    citations: Dict[str, Dict[str, Any]]
    final_answer: str
    error: str
    step_no: int
    pending: List[Any]  # 本轮待执行的 ToolCallRequest 列表
    last_executions: List[ToolExecution]
    executions: List[ToolExecution]  # 全量工具执行记录（收敛作答用）
    question: str
    doc_id: str
    nudged: bool  # 是否已注入过"请立即收敛作答"提醒


def _build_graph(runner: AgentRunner) -> Any:
    """构建并编译 agent ↔ tools 条件图（编译结果缓存在 runner 上复用）。

    Args:
        runner: AgentRunner（提供 LLM / 工具注册表 / 共享逻辑）
    Returns:
        已 compile 的 langgraph 图（可 ``.invoke(state, config)``）
    """
    cached = getattr(runner, "_langgraph_compiled", None)
    if cached is not None:
        return cached

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """LLM 决策节点：产出思考、工具调用请求或最终答案"""
        messages = list(state.get("messages", []))
        steps = list(state.get("steps", []))
        step_no = int(state.get("step_no", 0)) + 1
        t0 = time.time()
        response = runner._call_llm(messages)
        step = AgentStep(
            index=step_no,
            thought=str(response.content or ""),
            latency_ms=int((time.time() - t0) * 1000),
            input_tokens=int(response.input_tokens or 0),
            output_tokens=int(response.output_tokens or 0),
            cost_yuan=float(response.cost_yuan or 0.0),
        )
        steps.append(step)
        update: Dict[str, Any] = {
            "steps": steps,
            "step_no": step_no,
            "pending": list(response.tool_calls or []),
        }
        if response.error:
            update["error"] = response.error
            return update
        if not response.tool_calls:
            update["final_answer"] = str(response.content or "").strip()
            return update
        # 有工具调用：追加 assistant 消息（工具观察消息由 tools 节点追加）
        messages.append(runner._assistant_tool_call_message(response))
        update["messages"] = messages
        return update

    def tools_node(state: AgentState) -> Dict[str, Any]:
        """工具执行节点：执行上一步 pending 工具调用并记录结果 / 引用"""
        messages = list(state.get("messages", []))
        steps = list(state.get("steps", []))
        pending = state.get("pending") or []
        step_index = int(state.get("step_no", 0)) or len(steps)
        executions, new_flat, new_citations = runner._execute_tool_calls(
            step_index,
            pending,
            str(state.get("doc_id", "") or ""),
        )
        if steps:
            steps[-1].tool_calls = executions
        runner._append_tool_messages(messages, executions)
        flat = list(state.get("tool_calls", [])) + new_flat
        citations = dict(state.get("citations", {}))
        citations.update(new_citations)
        all_executions = list(state.get("executions", [])) + executions
        nudged = bool(state.get("nudged", False))
        step_no = int(state.get("step_no", 0))
        nudge_at = 2  # 完成 2 步工具调用后注入一次收敛提醒
        if not nudged and step_no >= nudge_at and step_no < runner.max_steps:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "（系统提醒）已获得足够的工具结果，请立即基于现有信息给出最终答案，"
                        "不要再调用工具。"
                    ),
                }
            )
            nudged = True
        return {
            "messages": messages,
            "steps": steps,
            "tool_calls": flat,
            "citations": citations,
            "pending": [],
            "last_executions": executions,
            "executions": all_executions,
            "nudged": nudged,
        }

    def finish_node(state: AgentState) -> Dict[str, Any]:
        """达到最大步数节点：强制无工具收敛作答（失败回退兜底摘要）"""
        steps = list(state.get("steps", []))
        answer, final_response = runner._force_final_answer(
            str(state.get("question", "") or ""),
            list(state.get("executions", [])),
        )
        if final_response is not None and steps:
            # 收敛调用的 token/成本归并到最后一个步骤（保持 steps 数不变）
            steps[-1].input_tokens += int(final_response.input_tokens or 0)
            steps[-1].output_tokens += int(final_response.output_tokens or 0)
            steps[-1].cost_yuan += float(final_response.cost_yuan or 0.0)
        return {
            "final_answer": answer,
            "pending": [],
            "steps": steps,
        }

    def route_after_agent(state: AgentState) -> str:
        """LLM 节点之后：有错误/答案结束，否则有工具调用去 tools"""
        if state.get("error") or state.get("final_answer"):
            return "end"
        return "tools" if state.get("pending") else "end"

    def route_after_tools(state: AgentState) -> str:
        """工具节点之后：达到最大步数去 finish，否则回到 LLM"""
        if int(state.get("step_no", 0)) >= runner.max_steps:
            return "finish"
        return "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "finish": "finish"},
    )
    builder.add_edge("finish", END)
    graph = builder.compile()
    runner._langgraph_compiled = graph
    return graph


def run_agent_langgraph(
    runner: AgentRunner,
    question: str,
    *,
    doc_id: str = "",
    user_id: str = "",
    history: List[Dict[str, Any]] | None = None,
) -> AgentResult:
    """用 LangGraph 状态图执行一次 Agent 问答。

    Args:
        runner: AgentRunner（提供 LLM / 工具 / 共享逻辑）
        question: 用户问题
        doc_id: 当前文档 ID（作为 search_chunks 默认过滤）
        user_id: 用户 ID（保留，不进入 prompt）
        history: 会话历史
    Returns:
        AgentResult（framework="langgraph"）
    """
    del user_id  # 仅保留签名兼容，当前不进 prompt
    start_time = time.time()
    trace_id = uuid.uuid4().hex[:16]
    graph = _build_graph(runner)
    initial_state: AgentState = {
        "messages": runner._initial_messages(question, history),
        "steps": [],
        "tool_calls": [],
        "citations": {},
        "final_answer": "",
        "error": "",
        "step_no": 0,
        "pending": [],
        "last_executions": [],
        "executions": [],
        "question": question,
        "doc_id": str(doc_id or ""),
    }
    config = {"recursion_limit": max(25, 2 * runner.max_steps + 10)}
    final_state = graph.invoke(initial_state, config)
    return runner._finalize(
        trace_id=trace_id,
        start_time=start_time,
        steps=list(final_state.get("steps", [])),
        flat=list(final_state.get("tool_calls", [])),
        citations=dict(final_state.get("citations", {})),
        answer=str(final_state.get("final_answer", "") or ""),
        error=str(final_state.get("error", "") or ""),
        framework="langgraph",
    )

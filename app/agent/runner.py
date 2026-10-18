"""Agent 编排核心（冻结契约 v1）：simple ReAct 循环 + 工厂函数。

模块组成：
- DEFAULT_SYSTEM_PROMPT：中文 system prompt（强调先调工具、禁止编造、附 [chunk_id] 引用）
- AgentRunner：简单 for 循环编排（LLM → 有 tool_calls 则执行工具 → 回到 LLM）
- create_agent_runner：工厂函数（auto 优先 LangGraph，可 import 才用；失败回退 simple）

LangGraph 编排在 ``app.agent.langgraph_runner`` 中实现，与 simple 循环共享同一套
``_call_llm / _execute_tool_calls / _append_tool_messages / _finalize`` 逻辑，
保证两种 framework 的结果结构完全一致。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .llm import (
    BaseToolCallingLLM,
    DashScopeToolCallingLLM,
    ToolCallingResponse,
    create_tool_calling_llm,
)
from .tools import ToolRegistry
from .types import (
    CITATION_PREVIEW_MAX_CHARS,
    RESULT_PREVIEW_MAX_CHARS,
    AgentResult,
    AgentStep,
    ToolExecution,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是「论文研读 Agent」，可以调用工具完成论文问答、片段检索、翻译与综述生成。\n"
    "必须遵守：\n"
    "1. 必须先调用工具获取论文事实依据，再给出最终回答；禁止编造数据、结论或引用。\n"
    "2. 可用工具：list_documents（列出已解析文档）、search_chunks（检索论文片段）、"
    "translate_text（翻译）、generate_review（生成综述）。\n"
    "3. 最终答案使用中文；引用论文片段时在结论后附 [chunk_id] 引用。\n"
    "4. 最多 {max_steps} 步工具调用；信息已足够时直接给出最终答案，不要再调用工具。"
)

# 匹配答案中的 [chunk_id] / [doc_id, chunk_id] 引用标记
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


# ---------------------------------------------------------------------------
# 公共小工具（simple 与 langgraph 编排共用）
# ---------------------------------------------------------------------------


def _preview_result(result: Mapping[str, Any]) -> str:
    """把工具结果转成截断的 JSON 字符串预览（供扁平 tool_calls 记录）"""
    try:
        text = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(result)
    return text[:RESULT_PREVIEW_MAX_CHARS]


def _make_flat_entry(
    step_index: int,
    execution: ToolExecution,
) -> Dict[str, Any]:
    """构造扁平 tool_calls 记录：{step, name, arguments, result_preview, latency_ms, ok}"""
    return {
        "step": step_index,
        "name": execution.name,
        "arguments": execution.arguments,
        "result_preview": _preview_result(execution.result),
        "latency_ms": execution.latency_ms,
        "ok": execution.ok,
    }


def _collect_search_citations(
    executions: List[ToolExecution],
    citations: Dict[str, Dict[str, Any]],
) -> None:
    """从 search_chunks 结果中收集引用 {chunk_id, doc_id, text_preview}（按 chunk_id 去重）"""
    for execution in executions:
        if execution.name != "search_chunks" or not execution.ok:
            continue
        chunks = execution.result.get("chunks") or []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            chunk_id = str(chunk.get("chunk_id", "") or "").strip()
            if not chunk_id or chunk_id in citations:
                continue
            text = str(chunk.get("text", "") or "")
            citations[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": str(chunk.get("doc_id", "") or ""),
                "text_preview": text[:CITATION_PREVIEW_MAX_CHARS],
            }


def _merge_answer_citations(
    answer: str,
    citations: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """把最终答案中出现的 [chunk_id] 合并进引用列表（去重）"""
    for match in _CITATION_PATTERN.finditer(answer or ""):
        for token in re.split(r"[,，\s]+", match.group(1)):
            token = token.strip()
            if not token or "#" not in token or len(token) > 100:
                continue
            if token in citations:
                continue
            citations[token] = {
                "chunk_id": token,
                "doc_id": token.split("#", 1)[0],
                "text_preview": "",
            }
    return citations


# ---------------------------------------------------------------------------
# AgentRunner（simple 循环）
# ---------------------------------------------------------------------------


class AgentRunner:
    """Agent 编排器（simple ReAct 循环）。

    循环逻辑：
        for step in 1..max_steps:
            resp = LLM(messages, tools)
            if resp.error → 记录错误并结束
            if 无 tool_calls → 取 content 为最终答案并结束
            否则逐个执行工具并追加 tool 观察消息，继续下一轮
        超过 max_steps 仍有 tool_calls → 用「已达到最大步数 + 最后工具结果摘要」兜底

    Args:
        llm: 带工具调用能力的 LLM（DashScope 真实客户端或 ScriptedMockLLM）
        tools: ToolRegistry（默认空注册表）
        framework: "auto" / "simple" / "langgraph"；auto 优先 LangGraph，失败回退 simple
        system_prompt: 自定义 system prompt（空串使用默认中文 prompt）
        max_steps: 最大决策轮数（≥1）
    """

    def __init__(
        self,
        llm: Optional[BaseToolCallingLLM] = None,
        tools: Optional[ToolRegistry] = None,
        framework: str = "auto",
        system_prompt: str = "",
        max_steps: int = 6,
    ):
        self.llm: BaseToolCallingLLM = llm or create_tool_calling_llm("auto")
        self.tools: ToolRegistry = tools if tools is not None else ToolRegistry()
        requested = (framework or "auto").lower()
        if requested not in ("auto", "simple", "langgraph"):
            logger.warning("未知 framework=%s，按 auto 处理", framework)
            requested = "auto"
        self.framework = requested
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(
            max_steps=self.max_steps
        )
        self.model = str(getattr(self.llm, "model", "") or "")
        # mock_mode：未显式声明时，只要不是 DashScope 真实客户端都视为 mock
        self.mock_mode = bool(
            getattr(
                self.llm,
                "mock_mode",
                not isinstance(self.llm, DashScopeToolCallingLLM),
            )
        )

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        doc_id: str = "",
        user_id: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """执行一次 Agent 问答。

        Args:
            question: 用户问题（中文/英文）
            doc_id: 当前文档 ID；作为 search_chunks 未指定 doc_id 时的默认过滤
            user_id: 用户 ID（当前仅保留给调用方审计，不进入 prompt）
            history: 会话历史 [{"role": "user"/"assistant", "content": str}, ...]
        Returns:
            AgentResult（任何 LLM / 工具异常都转为 error 字段，不抛出）
        """
        question_text = str(question or "").strip()
        if not question_text:
            return AgentResult(
                trace_id=uuid.uuid4().hex[:16],
                model=self.model,
                mock_mode=self.mock_mode,
                error="问题不能为空",
            )
        if self._use_langgraph():
            try:
                from .langgraph_runner import run_agent_langgraph

                return run_agent_langgraph(
                    self,
                    question_text,
                    doc_id=doc_id,
                    user_id=user_id,
                    history=history,
                )
            except Exception as exc:  # noqa: BLE001 - LangGraph 任何异常都回退 simple
                logger.warning("LangGraph 编排失败，回退 simple 循环: %s", exc)
                result = self._run_simple(
                    question_text,
                    doc_id=doc_id,
                    user_id=user_id,
                    history=history,
                )
                result.framework = "simple"
                return result
        return self._run_simple(
            question_text,
            doc_id=doc_id,
            user_id=user_id,
            history=history,
        )

    # ------------------------------------------------------------------
    # framework 选择
    # ------------------------------------------------------------------

    def _use_langgraph(self) -> bool:
        """是否需要尝试 LangGraph 编排（can import 且 framework 允许）"""
        if self.framework == "simple":
            return False
        try:
            import langgraph.graph  # noqa: F401
        except Exception:  # noqa: BLE001 - 缺包时降级
            if self.framework == "langgraph":
                logger.warning("langgraph 不可用，回退 simple 循环")
            return False
        return True

    # ------------------------------------------------------------------
    # simple 循环
    # ------------------------------------------------------------------

    def _run_simple(
        self,
        question: str,
        *,
        doc_id: str = "",
        user_id: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """简单 for 循环编排（LangGraph 不可用 / 显式 simple 时使用）"""
        del user_id  # 仅保留签名兼容，当前不进 prompt
        start_time = time.time()
        trace_id = uuid.uuid4().hex[:16]
        messages = self._initial_messages(question, history)
        steps: List[AgentStep] = []
        flat: List[Dict[str, Any]] = []
        citations: Dict[str, Dict[str, Any]] = {}
        answer = ""
        error = ""
        last_executions: List[ToolExecution] = []
        exhausted = False

        for step_index in range(1, self.max_steps + 1):
            t0 = time.time()
            response = self._call_llm(messages)
            step = AgentStep(
                index=step_index,
                thought=str(response.content or ""),
                latency_ms=int((time.time() - t0) * 1000),
                input_tokens=int(response.input_tokens or 0),
                output_tokens=int(response.output_tokens or 0),
                cost_yuan=float(response.cost_yuan or 0.0),
            )
            steps.append(step)

            if response.error:
                error = response.error
                break
            if not response.tool_calls:
                answer = str(response.content or "").strip()
                break

            executions, new_flat, new_citations = self._execute_tool_calls(
                step_index, response.tool_calls, doc_id
            )
            step.tool_calls = executions
            last_executions = executions
            flat.extend(new_flat)
            citations.update(new_citations)
            messages.append(self._assistant_tool_call_message(response))
            self._append_tool_messages(messages, executions)
        else:
            # for 循环自然结束（达到 max_steps 仍有工具调用）：兜底答案
            exhausted = True

        if exhausted:
            answer = self._max_steps_answer(last_executions)

        return self._finalize(
            trace_id=trace_id,
            start_time=start_time,
            steps=steps,
            flat=flat,
            citations=citations,
            answer=answer,
            error=error,
            framework="simple",
        )

    # ------------------------------------------------------------------
    # 共享逻辑（LangGraph 编排也调用这些方法）
    # ------------------------------------------------------------------

    def _initial_messages(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """构造初始 messages：system + 合法 history + user"""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for item in history or []:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", "") or "")
            content = item.get("content", "")
            if role not in ("user", "assistant"):
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _call_llm(self, messages: List[Dict[str, Any]]) -> ToolCallingResponse:
        """调用 LLM（任何异常统一转成 error 响应，不向上抛）"""
        try:
            return self.llm.chat_with_tools(
                messages,
                self.tools.specs(),
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001 - LLM 异常降级为 error 字段
            logger.exception("LLM 调用异常")
            return ToolCallingResponse(
                model=self.model,
                error=f"LLM 调用异常: {type(exc).__name__}: {exc}",
            )

    def _execute_tool_calls(
        self,
        step_index: int,
        tool_calls: List[Any],
        doc_id: str = "",
    ) -> tuple:
        """逐个执行工具调用。

        Args:
            step_index: 当前轮次（写入扁平记录 step 字段）
            tool_calls: LLM 返回的 ToolCallRequest 列表
            doc_id: search_chunks 未显式指定 doc_id 时补全的默认文档过滤
        Returns:
            (executions, flat_entries, citations)
        """
        executions: List[ToolExecution] = []
        flat: List[Dict[str, Any]] = []
        citations: Dict[str, Dict[str, Any]] = {}
        for call in tool_calls or []:
            arguments: Dict[str, Any] = {}
            raw_arguments = getattr(call, "arguments", {})
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            name = str(getattr(call, "name", "") or "")
            if name == "search_chunks" and doc_id and not arguments.get("doc_id"):
                # doc_id 作为默认文档过滤（LLM 未显式指定时）
                arguments["doc_id"] = doc_id
            execution = self.tools.call(name, arguments)
            executions.append(execution)
            flat.append(_make_flat_entry(step_index, execution))
        _collect_search_citations(executions, citations)
        return executions, flat, citations

    def _assistant_tool_call_message(
        self,
        response: ToolCallingResponse,
    ) -> Dict[str, Any]:
        """构造 assistant 工具调用消息（OpenAI tools 格式）"""
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": str(response.content or ""),
        }
        tool_calls = []
        for index, call in enumerate(response.tool_calls or []):
            try:
                arguments_text = json.dumps(call.arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments_text = "{}"
            tool_calls.append(
                {
                    "id": call.call_id or f"call_{index}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments_text},
                }
            )
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _append_tool_messages(
        self,
        messages: List[Dict[str, Any]],
        executions: List[ToolExecution],
    ) -> None:
        """把工具结果以 role=tool 消息追加到 messages（content 截断到 3000 字符）"""
        for execution in executions:
            try:
                content = json.dumps(execution.result, ensure_ascii=False)
            except (TypeError, ValueError):
                content = "{}"
            messages.append(
                {
                    "role": "tool",
                    "name": execution.name,
                    "content": content[:3000],
                }
            )

    def _max_steps_answer(self, executions: List[ToolExecution]) -> str:
        """超过最大步数时的兜底答案：提示 + 最后一次工具结果摘要"""
        answer = "已达到最大步数，以下为已收集信息："
        if executions:
            last = executions[-1]
            summary = _preview_result(last.result) if last.ok else last.error
            if summary:
                answer += summary
        return answer

    def _finalize(
        self,
        *,
        trace_id: str,
        start_time: float,
        steps: List[AgentStep],
        flat: List[Dict[str, Any]],
        citations: Dict[str, Dict[str, Any]],
        answer: str,
        error: str,
        framework: str,
    ) -> AgentResult:
        """组装 AgentResult（合并引用、累计 token / cost、空答案兜底 error）"""
        answer_text = str(answer or "").strip()
        merged_citations = _merge_answer_citations(answer_text, citations)
        if not answer_text and not error:
            error = "Agent 未产生回答"
        return AgentResult(
            answer=answer_text,
            steps=steps,
            tool_calls=flat,
            citations=list(merged_citations.values()),
            trace_id=trace_id,
            model=self.model,
            input_tokens=sum(step.input_tokens for step in steps),
            output_tokens=sum(step.output_tokens for step in steps),
            cost_yuan=sum(step.cost_yuan for step in steps),
            mock_mode=self.mock_mode,
            framework=framework,
            latency_ms=int((time.time() - start_time) * 1000),
            error=error,
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_agent_runner(
    llm: Optional[BaseToolCallingLLM] = None,
    tools: Optional[ToolRegistry] = None,
    framework: str = "auto",
    max_steps: int = 6,
) -> AgentRunner:
    """创建 AgentRunner。

    Args:
        llm: LLM 客户端（None 时 create_tool_calling_llm("auto")）
        tools: 工具注册表（None 时为空注册表；通常传 create_default_tools(...)）
        framework: "auto" / "simple" / "langgraph"；auto 时优先 langgraph（可 import 才用）
        max_steps: 最大决策轮数
    Returns:
        AgentRunner
    """
    resolved = (framework or "auto").lower()
    if resolved == "auto":
        try:
            import langgraph.graph  # noqa: F401

            resolved = "langgraph"
        except Exception:  # noqa: BLE001 - 缺包时用 simple
            logger.info("langgraph 不可用，auto 模式使用 simple 循环")
            resolved = "simple"
    return AgentRunner(
        llm=llm,
        tools=tools,
        framework=resolved,
        max_steps=max_steps,
    )

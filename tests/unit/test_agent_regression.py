"""Agent 核心 pytest 回归测试（全部离线，不真实调用 DashScope API）。

覆盖冻结契约 v1：
- 工具注册表：4 个工具 schema / 成功调用归一化 / 未知工具 / 参数异常 / 回调异常；
- simple ReAct 循环：数据集问答、翻译、综述三类任务（多步 + 引用 + 工具记录）；
- LangGraph 条件图：framework=="langgraph" 真跑，结果结构与 simple 一致；
- 边界：max_steps 兜底、LLM 报错透传、空问题、无工具可用、框架回退；
- AgentResult.to_dict() JSON-safe、答案中 [chunk_id] 引用合并、history 注入。

运行：
    pytest tests/unit/test_agent_regression.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from app.agent import (
    AgentResult,
    AgentRunner,
    BaseToolCallingLLM,
    ScriptedMockLLM,
    ToolCallingResponse,
    ToolRegistry,
    create_agent_runner,
    create_default_tools,
    create_tool_calling_llm,
    default_tool_specs,
)

# ---------------------------------------------------------------------------
# 假后端（文档 / 检索 / 翻译 / 综述回调，全部离线）
# ---------------------------------------------------------------------------

FAKE_DOCS: List[Dict[str, Any]] = [
    {"doc_id": "paper_1", "title": "Attention Is All You Need", "num_pages": 15},
    {
        "doc_id": "paper_2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "num_pages": 16,
    },
]

FAKE_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "paper_1#2.0",
        "doc_id": "paper_1",
        "text": "Experiments show 85.6% top-5 accuracy on ImageNet-1K dataset.",
        "section": "3. Results",
        "score": 0.91,
    },
    {
        "chunk_id": "paper_2#2.1",
        "doc_id": "paper_2",
        "text": "BERT achieves BLEU=27.3 on WMT 2014 machine translation.",
        "section": "4. Experiments",
        "score": 0.84,
    },
]

FAKE_REVIEW_TEXT = "本文综述了 Transformer 与 BERT 的主要进展。" * 100  # >1500 字，验证截断


class FakeBackend:
    """假工具后端：模拟 app.api 注入的检索 / 翻译 / 综述管道。"""

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        fail_search: bool = False,
    ):
        self.documents = list(FAKE_DOCS if documents is None else documents)
        self.chunks = list(FAKE_CHUNKS if chunks is None else chunks)
        self.fail_search = fail_search

    def list_documents(self) -> List[Dict[str, Any]]:
        return [dict(document) for document in self.documents]

    def search_chunks(self, query: str, doc_id: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        if self.fail_search:
            raise RuntimeError("模拟检索服务不可用")
        results = [dict(chunk) for chunk in self.chunks]
        if doc_id:
            results = [chunk for chunk in results if chunk.get("doc_id") == doc_id]
        return results[: int(top_k)]

    def translate_text(self, text: str, target_lang: str = "zh-CN") -> Dict[str, Any]:
        return {
            "translated": f"【中文译文】{text[:80]}",
            "aligned_terms": ["自注意力机制"],
            "target_lang": target_lang,
            "mock_mode": True,
        }

    def generate_review(
        self,
        topic: str,
        doc_ids: List[str],
        num_sections: int = 3,
    ) -> Dict[str, Any]:
        return {
            "title": f"综述：{topic}",
            "full_text": FAKE_REVIEW_TEXT,
            "references": [{"index": 1, "doc_id": "paper_1", "chunk_id": "paper_1#2.0"}],
            "citation_stats": {"total": 1, "entailment": 1},
            "mock_mode": True,
            "error": "",
        }

    def build_tools(self) -> ToolRegistry:
        return create_default_tools(
            list_documents=self.list_documents,
            search_chunks=self.search_chunks,
            translate_text=self.translate_text,
            generate_review=self.generate_review,
        )


class FailingLLM(BaseToolCallingLLM):
    """始终返回 error 的 LLM（验证 LLM 报错不抛异常、error 透传）"""

    model = "failing-llm"
    mock_mode = True

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolCallingResponse:
        return ToolCallingResponse(model=self.model, error="模拟 LLM 服务故障")


class RecordingLLM(BaseToolCallingLLM):
    """固定答案 + 记录 messages 的 LLM（验证 history 注入与引用合并）"""

    model = "recording-llm"
    mock_mode = True

    def __init__(self, answer: str = "固定答案 [paper_1#2.0]"):
        self.answer = answer
        self.calls: List[List[Dict[str, Any]]] = []

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolCallingResponse:
        self.calls.append([dict(message) for message in messages])
        return ToolCallingResponse(
            content=self.answer,
            model=self.model,
            input_tokens=3,
            output_tokens=2,
            cost_yuan=0.0,
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> FakeBackend:
    """假工具后端"""
    return FakeBackend()


@pytest.fixture()
def simple_runner(backend: FakeBackend) -> AgentRunner:
    """simple 循环 runner（mock LLM + 假工具）"""
    return AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="simple",
    )


# ---------------------------------------------------------------------------
# 1. 工具注册表 / 执行
# ---------------------------------------------------------------------------


def test_default_tool_specs_contract() -> None:
    """4 个默认工具 schema 齐全且为 OpenAI tools 格式（且返回深拷贝）"""
    specs = default_tool_specs()
    names = [spec["function"]["name"] for spec in specs]
    assert names == [
        "list_documents",
        "search_chunks",
        "translate_text",
        "generate_review",
    ]
    for spec in specs:
        assert spec["type"] == "function"
        parameters = spec["function"]["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(spec["function"]["description"], str)
        assert spec["function"]["description"]
    # search_chunks 的契约参数名
    search_schema = next(
        spec["function"]["parameters"]
        for spec in specs
        if spec["function"]["name"] == "search_chunks"
    )
    assert set(search_schema["properties"]) == {"query", "doc_id", "top_k"}
    assert search_schema["required"] == ["query"]
    # 深拷贝：外部修改不污染下一次调用
    specs[0]["function"]["name"] = "tampered"
    assert default_tool_specs()[0]["function"]["name"] == "list_documents"


def test_tool_execution_success_and_normalization(backend: FakeBackend) -> None:
    """工具成功调用：search/translate/review 结果字段按契约归一化"""
    tools = backend.build_tools()
    assert tools.names == [
        "list_documents", "search_chunks", "translate_text", "generate_review",
    ]
    assert [spec["function"]["name"] for spec in tools.specs()] == tools.names

    listed = tools.call("list_documents", {})
    assert listed.ok and listed.error == ""
    assert len(listed.result["documents"]) == 2

    searched = tools.call("search_chunks", {"query": "数据集", "doc_id": "paper_1", "top_k": 3})
    assert searched.ok
    assert set(searched.result["chunks"][0]) == {
        "chunk_id", "doc_id", "text", "section", "score",
    }
    assert all(chunk["doc_id"] == "paper_1" for chunk in searched.result["chunks"])

    translated = tools.call("translate_text", {"text": "Self-attention is the core."})
    assert translated.ok
    assert translated.result["target_lang"] == "zh-CN"
    assert translated.result["translated"]
    assert isinstance(translated.result["aligned_terms"], list)

    reviewed = tools.call("generate_review", {"topic": "Transformer 综述"})
    assert reviewed.ok
    assert reviewed.result["full_text_truncated"] is True
    assert len(reviewed.result["full_text"]) <= 1500
    assert reviewed.result["title"].startswith("综述")


def test_tool_registry_errors_do_not_raise(backend: FakeBackend) -> None:
    """未知工具 / 参数异常 / 回调抛异常统一转为 ToolExecution.error"""
    tools = backend.build_tools()

    unknown = tools.call("not_exists", {})
    assert not unknown.ok and "未知工具" in unknown.error

    missing = tools.call("search_chunks", {"top_k": 3})
    assert not missing.ok and "缺少必填参数" in missing.error

    bad_type = tools.call("search_chunks", {"query": "q", "top_k": "abc"})
    assert not bad_type.ok and "类型错误" in bad_type.error

    failing = FakeBackend(fail_search=True).build_tools().call("search_chunks", {"query": "BLEU"})
    assert not failing.ok and "模拟检索服务不可用" in failing.error
    assert failing.result == {}


# ---------------------------------------------------------------------------
# 2. simple ReAct 循环
# ---------------------------------------------------------------------------


def test_simple_runner_dataset_question(simple_runner: AgentRunner) -> None:
    """数据集问答：search_chunks → 最终答案 + [chunk_id] 引用"""
    result = simple_runner.run("这篇论文用了什么数据集和模型？")
    assert result.framework == "simple"
    assert result.error == ""
    assert len(result.steps) >= 2
    assert [call["name"] for call in result.tool_calls] == ["search_chunks"]
    assert result.answer and "paper_1#2.0" in result.answer
    assert [c["chunk_id"] for c in result.citations] == ["paper_1#2.0", "paper_2#2.1"]
    assert result.citations[0]["text_preview"].startswith("Experiments show 85.6%")
    assert result.mock_mode is True
    assert len(result.trace_id) == 16
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert result.cost_yuan == 0.0  # mock 不计费


def test_simple_runner_translate_task(simple_runner: AgentRunner) -> None:
    """翻译任务：translate_text（引号内文本）→ 最终答案引用译文"""
    result = simple_runner.run('请把 "Self-attention is the core of Transformer." 翻译成中文')
    assert result.error == ""
    assert len(result.steps) >= 2
    assert [call["name"] for call in result.tool_calls] == ["translate_text"]
    assert "Self-attention" in result.tool_calls[0]["arguments"]["text"]
    assert result.tool_calls[0]["arguments"]["target_lang"] == "zh-CN"
    assert "中文译文" in result.answer


def test_simple_runner_review_task(simple_runner: AgentRunner) -> None:
    """综述任务：list_documents → generate_review → 最终答案"""
    result = simple_runner.run("请基于已上传论文生成一篇 Transformer 综述")
    assert result.error == ""
    assert [call["name"] for call in result.tool_calls] == ["list_documents", "generate_review"]
    assert len(result.steps) == 3
    review_arguments = result.tool_calls[-1]["arguments"]
    assert review_arguments["doc_ids"] == ["paper_1", "paper_2"]
    assert review_arguments["num_sections"] == 3
    assert "综述" in result.answer
    # AgentStep 里也保留完整工具执行记录
    assert result.steps[1].tool_calls[0].name == "generate_review"
    assert result.steps[1].tool_calls[0].ok is True


def test_flat_tool_calls_preview_contract(simple_runner: AgentRunner) -> None:
    """扁平 tool_calls 字段与 result_preview 长度符合契约"""
    result = simple_runner.run("这篇论文用了什么数据集？")
    assert result.tool_calls
    for call in result.tool_calls:
        assert {"step", "name", "arguments", "result_preview", "latency_ms", "ok"} <= set(call)
        assert isinstance(call["result_preview"], str)
        assert len(call["result_preview"]) <= 300
        assert call["ok"] is True


# ---------------------------------------------------------------------------
# 3. LangGraph 循环
# ---------------------------------------------------------------------------


def test_langgraph_runner_dataset_question(backend: FakeBackend) -> None:
    """LangGraph 条件图真跑：framework=="langgraph"，结果与 simple 一致"""
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="langgraph",
    )
    result = runner.run("这篇论文用了什么数据集和模型？")
    assert result.framework == "langgraph"
    assert result.error == ""
    assert len(result.steps) >= 2
    assert result.answer
    assert [c["chunk_id"] for c in result.citations] == ["paper_1#2.0", "paper_2#2.1"]


def test_langgraph_runner_review_multi_step(backend: FakeBackend) -> None:
    """LangGraph 下综述任务同样 list_documents → generate_review → 答案"""
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="langgraph",
    )
    result = runner.run("请基于已上传论文生成一篇 Transformer 综述")
    assert result.framework == "langgraph"
    assert [call["name"] for call in result.tool_calls] == ["list_documents", "generate_review"]
    assert len(result.steps) == 3
    assert "综述" in result.answer


# ---------------------------------------------------------------------------
# 4. 边界与容错
# ---------------------------------------------------------------------------


def test_max_steps_fallback(backend: FakeBackend) -> None:
    """max_steps=1 时工具调用后直接兜底，不抛异常、error 保持空"""
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="simple",
        max_steps=1,
    )
    result = runner.run("这篇论文用了什么数据集和模型？")
    assert len(result.steps) == 1
    assert "已达到最大步数" in result.answer
    assert result.error == ""


def test_llm_error_propagated(backend: FakeBackend) -> None:
    """LLM 报错：error 透传 + 保留已产生步骤，不抛异常"""
    runner = AgentRunner(
        llm=FailingLLM(),
        tools=backend.build_tools(),
        framework="simple",
    )
    result = runner.run("这篇论文用了什么数据集？")
    assert "模拟 LLM 服务故障" in result.error
    assert len(result.steps) == 1
    assert result.answer == ""


def test_empty_question_returns_error() -> None:
    """空问题：返回明确 error，不调用 LLM / 工具"""
    runner = AgentRunner(llm=ScriptedMockLLM(), tools=ToolRegistry(), framework="simple")
    result = runner.run("   ")
    assert "问题不能为空" in result.error
    assert result.steps == []
    assert result.tool_calls == []


def test_no_tools_available() -> None:
    """无工具可用：仍给出非空兜底答案，不抛异常"""
    runner = AgentRunner(llm=ScriptedMockLLM(), tools=ToolRegistry(), framework="simple")
    result = runner.run("这篇论文用了什么数据集？")
    assert result.answer
    assert result.tool_calls == []
    assert result.error == ""


def test_framework_falls_back_to_simple(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend,
) -> None:
    """LangGraph 运行异常时自动回退 simple 并把 framework 置 "simple" """

    def _boom(*args: Any, **kwargs: Any) -> AgentResult:
        raise RuntimeError("模拟 langgraph 运行异常")

    monkeypatch.setattr("app.agent.langgraph_runner.run_agent_langgraph", _boom)
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="langgraph",
    )
    result = runner.run("这篇论文用了什么数据集？")
    assert result.framework == "simple"
    assert result.answer
    assert result.error == ""


# ---------------------------------------------------------------------------
# 5. 序列化 / 引用合并 / history / 工厂
# ---------------------------------------------------------------------------


def test_agent_result_to_dict_json_safe(simple_runner: AgentRunner) -> None:
    """to_dict 顶层 key 与契约一致、嵌套结构正确、可 json.dumps"""
    result = simple_runner.run("这篇论文用了什么数据集？")
    payload = result.to_dict()
    assert set(payload) == {
        "answer", "steps", "tool_calls", "citations", "trace_id", "model",
        "input_tokens", "output_tokens", "cost_yuan", "mock_mode",
        "framework", "latency_ms", "error",
    }
    json.dumps(payload, ensure_ascii=False)  # 不抛异常即 JSON-safe
    # AgentStep 的 ToolExecution 序列化字段
    first_execution = payload["steps"][0]["tool_calls"][0]
    assert {"name", "arguments", "result", "error", "latency_ms", "ok"} <= set(first_execution)
    assert first_execution["ok"] is True
    assert all({"chunk_id", "doc_id", "text_preview"} <= set(c) for c in payload["citations"])
    assert payload["tool_calls"][0]["step"] == 1
    assert isinstance(payload["latency_ms"], int)
    assert isinstance(payload["cost_yuan"], float)


def test_citations_merged_from_answer(backend: FakeBackend) -> None:
    """最终答案里出现的 [chunk_id]（非检索结果）也应合并进 citations"""
    llm = RecordingLLM(answer="根据论文 [paper_1#2.0] 与 [paper_9#9.0]，结论成立。")
    runner = AgentRunner(llm=llm, tools=backend.build_tools(), framework="simple")
    result = runner.run("论文结论是什么？")
    citation_ids = {citation["chunk_id"] for citation in result.citations}
    assert {"paper_1#2.0", "paper_9#9.0"} <= citation_ids
    assert result.answer and result.error == ""


def test_empty_answer_sets_agent_error(backend: FakeBackend) -> None:
    """LLM 无工具调用且答案为空 → error='Agent 未产生回答'（契约兜底）"""
    llm = RecordingLLM(answer="")
    runner = AgentRunner(llm=llm, tools=backend.build_tools(), framework="simple")
    result = runner.run("论文结论是什么？")
    assert result.answer == ""
    assert result.error == "Agent 未产生回答"


def test_dashscope_response_parsing_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """DashScope function calling 响应解析（离线 fake 响应，不发起网络请求）"""
    from types import SimpleNamespace

    from app.agent import DashScopeToolCallingLLM

    # 无 key 时 available=False、chat 直接返回 error（不触发网络）
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    unavailable = DashScopeToolCallingLLM(api_key="")
    assert unavailable.available is False
    response = unavailable.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert "DASHSCOPE_API_KEY" in response.error

    # 有 key 且 dashscope 已安装时 available=True（仅属性检查，不发起请求）
    llm = DashScopeToolCallingLLM(api_key="sk-fake-offline")
    assert llm.available is True

    # 1) dict 形态响应：content + tool_calls（arguments JSON 字符串）
    dict_response = {
        "status_code": 200,
        "usage": {"input_tokens": 120, "output_tokens": 30},
        "output": {
            "choices": [
                {
                    "message": {
                        "content": "需要检索论文片段",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search_chunks",
                                    "arguments": '{"query": "BLEU", "top_k": 3}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    }
    parsed = llm._parse_response(dict_response)
    assert parsed.error == ""
    assert parsed.content == "需要检索论文片段"
    assert parsed.tool_calls[0].name == "search_chunks"
    assert parsed.tool_calls[0].arguments == {"query": "BLEU", "top_k": 3}
    assert parsed.tool_calls[0].call_id == "call_1"
    assert parsed.input_tokens == 120 and parsed.output_tokens == 30
    assert parsed.cost_yuan > 0  # qwen-plus 价目估算

    # 2) 对象形态响应 + 非法参数 JSON → arguments={"_raw": ...} 且保留 error
    object_response = SimpleNamespace(
        status_code=200,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        output=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(
                                    name="translate_text",
                                    arguments="{not valid json",
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
    )
    parsed = llm._parse_response(object_response)
    assert parsed.tool_calls[0].arguments == {"_raw": "{not valid json"}
    assert "不是合法 JSON" in parsed.error

    # 3) API 错误响应：error 透传，不抛异常
    error_response = {"status_code": 400, "code": "InvalidParameter", "message": "bad request"}
    parsed = llm._parse_response(error_response)
    assert parsed.error and "400" in parsed.error


def test_history_injected_into_messages(backend: FakeBackend) -> None:
    """history 合法消息按序注入；非法角色/空内容被忽略"""
    llm = RecordingLLM(answer="好的。")
    runner = AgentRunner(llm=llm, tools=backend.build_tools(), framework="simple")
    result = runner.run(
        "当前问题",
        history=[
            {"role": "user", "content": "上一轮问题"},
            {"role": "assistant", "content": "上一轮回答"},
            {"role": "system", "content": "非法角色应忽略"},
            {"role": "user", "content": ""},
        ],
    )
    assert result.error == ""
    messages = llm.calls[0]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "user",
    ]
    assert messages[1]["content"] == "上一轮问题"
    assert messages[-1]["content"] == "当前问题"


def test_create_agent_runner_auto_offline(backend: FakeBackend) -> None:
    """create_agent_runner(auto)：可 import langgraph 时优先 langgraph；llm=None 无 key 用 mock"""
    runner = create_agent_runner(llm=None, tools=backend.build_tools())
    assert runner.framework in ("langgraph", "simple")
    assert isinstance(runner.llm, ScriptedMockLLM)  # 无 key 自动 mock，不发起真实请求
    assert create_tool_calling_llm("auto").available is True
    result = runner.run("这篇论文用了什么数据集？")
    assert result.framework == runner.framework
    assert result.error == ""
    assert result.answer

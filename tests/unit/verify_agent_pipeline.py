"""Agent 核心链路独立验证脚本（离线 mock，不真实调用 DashScope API）。

验证内容（对应冻结契约 v1）：
1. 工具注册表：4 个工具 spec 齐全；未知工具 / 参数异常 / 回调抛异常均返回
   ToolExecution.error，不向上抛；结果字段归一化正确；
2. simple ReAct 循环：假文档 / 假检索 / 假翻译 / 假综述回调下，跑通
   「数据集问答」「翻译任务」「综述任务」，断言多步、答案、工具调用记录、引用抽取；
3. LangGraph 循环：同一任务用 framework="langgraph" 真跑，断言 framework=="langgraph"；
4. 边界：无工具可用 / LLM 报错 / max_steps=1 均不抛异常；AgentResult.to_dict() 可 json.dumps。

运行方式（在项目根目录 paper-research-assistant/ 下）：
    python tests/unit/verify_agent_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 项目根目录加入 sys.path：让脚本在任意目录执行时都能 import app.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在 import app 之前清空环境变量：强制 mock，避免本地 .env 的真实 key 触发真实 API
for _k in ("DASHSCOPE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
    os.environ.pop(_k, None)
    os.environ[_k] = ""

from app.agent import (  # noqa: E402
    AgentRunner,
    BaseToolCallingLLM,
    ScriptedMockLLM,
    ToolCallingResponse,
    create_agent_runner,
    create_default_tools,
    create_tool_calling_llm,
    default_tool_specs,
)
from app.agent.tools import ToolRegistry  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """记录一条断言结果并打印（不抛异常，便于一次看全部结果）"""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  [FAIL] {name}" + (f"  —  {detail}" if detail else ""))


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
    """一组假的工具回调（模拟 app.api 注入的真实管道，便于离线验证）。"""

    def __init__(
        self,
        documents: List[Dict[str, Any]] | None = None,
        chunks: List[Dict[str, Any]] | None = None,
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

    def build_tools(self):
        """创建绑定本后端的 ToolRegistry"""
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


# ---------------------------------------------------------------------------
# 1. 工具注册表
# ---------------------------------------------------------------------------


def test_tool_registry() -> None:
    """测试 1：工具注册表（specs / 归一化 / 未知工具 / 参数异常 / 回调异常）"""
    print("\n" + "=" * 70)
    print("测试 1：工具注册表（4 个工具 + 异常降级）")
    print("=" * 70)

    specs = default_tool_specs()
    names = [spec["function"]["name"] for spec in specs]
    expected_names = ["list_documents", "search_chunks", "translate_text", "generate_review"]
    check("default_tool_specs 返回 4 个工具", len(specs) == 4, f"names={names}")
    check("工具名与契约一致", names == expected_names, f"names={names}")
    check(
        "spec 为 OpenAI tools 格式",
        all(
            spec.get("type") == "function"
            and isinstance(spec.get("function", {}).get("parameters"), dict)
            and spec["function"]["parameters"].get("type") == "object"
            for spec in specs
        ),
        "存在不符合 OpenAI tools 格式的 spec",
    )

    backend = FakeBackend()
    tools = backend.build_tools()
    check("ToolRegistry.names 为 4 个工具", tools.names == expected_names, str(tools.names))
    check("registry.specs() 与 default_tool_specs() 名称一致",
          [s["function"]["name"] for s in tools.specs()] == expected_names)

    # list_documents 成功
    execution = tools.call("list_documents", {})
    check("list_documents 成功", execution.ok and len(execution.result["documents"]) == 2,
          f"result={execution.result}")
    check("list_documents 记录耗时", execution.latency_ms >= 0)

    # search_chunks 归一化
    execution = tools.call("search_chunks", {"query": "数据集", "top_k": 2})
    chunks = execution.result.get("chunks") if execution.ok else []
    check("search_chunks 成功", execution.ok, execution.error)
    check(
        "chunk 归一化字段齐全",
        bool(chunks)
        and set(chunks[0]) == {"chunk_id", "doc_id", "text", "section", "score"},
        f"chunks={chunks}",
    )
    check("search_chunks 非法 top_k 类型返回 error",
          not tools.call("search_chunks", {"query": "q", "top_k": "abc"}).ok)
    check("search_chunks 缺少必填参数返回 error",
          "缺少必填参数" in tools.call("search_chunks", {"top_k": 3}).error)

    # translate_text 归一化
    execution = tools.call("translate_text", {"text": "Self-attention is the core."})
    data = execution.result
    check(
        "translate_text 字段归一化",
        execution.ok
        and data.get("translated")
        and data.get("target_lang") == "zh-CN"
        and isinstance(data.get("aligned_terms"), list),
        f"result={data}",
    )

    # generate_review：full_text 截断 + full_text_truncated
    execution = tools.call(
        "generate_review",
        {"topic": "Transformer 综述", "doc_ids": ["paper_1", "paper_2"], "num_sections": 3},
    )
    data = execution.result
    check("generate_review 成功", execution.ok, execution.error)
    check("generate_review.full_text 截断到 1500 字以内",
          len(data.get("full_text", "")) <= 1500, f"len={len(data.get('full_text', ''))}")
    check("generate_review 标记 full_text_truncated", data.get("full_text_truncated") is True)

    # 未知工具 / 回调异常 → error 不抛
    unknown = tools.call("not_exists", {})
    check("未知工具返回 error 不抛异常",
          not unknown.ok and "未知工具" in unknown.error, unknown.error)
    failing_backend = FakeBackend(fail_search=True)
    failing_tools = failing_backend.build_tools()
    failing = failing_tools.call("search_chunks", {"query": "BLEU"})
    check("回调抛异常返回 error 不抛异常",
          not failing.ok and "模拟检索服务不可用" in failing.error, failing.error)
    check("异常工具结果为空 dict", failing.result == {}, str(failing.result))


# ---------------------------------------------------------------------------
# 2. simple ReAct 循环
# ---------------------------------------------------------------------------


def test_simple_loop() -> None:
    """测试 2：simple 循环跑通数据集问答 / 翻译 / 综述（多步 + 引用）"""
    print("\n" + "=" * 70)
    print("测试 2：simple ReAct 循环（数据集 / 翻译 / 综述）")
    print("=" * 70)

    backend = FakeBackend()
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="simple",
    )

    # 2a. 数据集问答（检索类）
    result = runner.run("这篇论文用了什么数据集和模型？")
    tool_names = [call["name"] for call in result.tool_calls]
    check("simple 数据集问答：framework=simple", result.framework == "simple", result.framework)
    check("simple 数据集问答：多步（steps>=2）", len(result.steps) >= 2, f"steps={len(result.steps)}")
    check("simple 数据集问答：调用 search_chunks", tool_names == ["search_chunks"], str(tool_names))
    check("simple 数据集问答：答案非空", bool(result.answer), result.answer)
    check("simple 数据集问答：答案含 chunk 引用", "paper_1#2.0" in result.answer, result.answer)
    check("simple 数据集问答：error 为空", result.error == "", result.error)
    citation_ids = [citation["chunk_id"] for citation in result.citations]
    check(
        "simple 数据集问答：引用抽取正确",
        citation_ids == ["paper_1#2.0", "paper_2#2.1"],
        str(citation_ids),
    )
    check(
        "simple 数据集问答：引用预览为 chunk 原文前 200 字",
        result.citations
        and result.citations[0]["doc_id"] == "paper_1"
        and result.citations[0]["text_preview"].startswith("Experiments show 85.6%"),
        str(result.citations[:1]),
    )
    check("simple 数据集问答：mock_mode=True", result.mock_mode is True)
    check("simple 数据集问答：model 非空", bool(result.model), result.model)
    check("simple 数据集问答：trace_id 16 位", len(result.trace_id) == 16, result.trace_id)

    # 2b. 翻译任务
    result = runner.run('请把 "Self-attention is the core of Transformer." 翻译成中文')
    tool_names = [call["name"] for call in result.tool_calls]
    check("simple 翻译任务：调用 translate_text", tool_names == ["translate_text"], str(tool_names))
    check("simple 翻译任务：多步（steps>=2）", len(result.steps) >= 2, f"steps={len(result.steps)}")
    check("simple 翻译任务：答案引用译文",
          "中文译文" in result.answer and "自注意力机制" in result.answer, result.answer)
    check("simple 翻译任务：工具入参为引号内文本",
          result.tool_calls and "Self-attention" in result.tool_calls[0]["arguments"]["text"],
          str(result.tool_calls[:1]))

    # 2c. 综述任务
    result = runner.run("请基于已上传论文生成一篇 Transformer 综述")
    tool_names = [call["name"] for call in result.tool_calls]
    check("simple 综述任务：先 list_documents 再 generate_review",
          tool_names == ["list_documents", "generate_review"], str(tool_names))
    check("simple 综述任务：多步（steps>=2）", len(result.steps) >= 2, f"steps={len(result.steps)}")
    check("simple 综述任务：答案含综述标题", "综述" in result.answer, result.answer[:120])
    check("simple 综述任务：工具调用记录落在 AgentStep 中",
          all(step.tool_calls or index == len(result.steps) - 1
              for index, step in enumerate(result.steps)),
          str([len(step.tool_calls) for step in result.steps]))
    check("simple 综述任务：flat tool_calls 有 result_preview",
          all(isinstance(call.get("result_preview"), str) for call in result.tool_calls))
    review_call = result.tool_calls[-1]
    check("simple 综述任务：doc_ids 来自 list_documents 结果",
          review_call["arguments"].get("doc_ids") == ["paper_1", "paper_2"],
          str(review_call["arguments"]))

    # 2d. doc_id 默认过滤（runner.run(doc_id=...) 补全 search_chunks 参数）
    result = runner.run("论文用了什么数据集？", doc_id="paper_2")
    check("doc_id 作为 search_chunks 默认过滤",
          result.tool_calls
          and result.tool_calls[0]["arguments"].get("doc_id") == "paper_2",
          str(result.tool_calls[:1]))
    check("doc_id 过滤后只引用该文档 chunk",
          [citation["chunk_id"] for citation in result.citations] == ["paper_2#2.1"],
          str([citation["chunk_id"] for citation in result.citations]))


# ---------------------------------------------------------------------------
# 3. LangGraph 循环
# ---------------------------------------------------------------------------


def test_langgraph_loop() -> None:
    """测试 3：LangGraph 条件图真跑（framework=='langgraph' + 多步 + 引用）"""
    print("\n" + "=" * 70)
    print("测试 3：LangGraph 循环（agent ↔ tools 条件图）")
    print("=" * 70)

    backend = FakeBackend()
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=backend.build_tools(),
        framework="langgraph",
    )
    result = runner.run("这篇论文用了什么数据集和模型？")
    check("LangGraph：framework=='langgraph'", result.framework == "langgraph", result.framework)
    check("LangGraph：多步（steps>=2）", len(result.steps) >= 2, f"steps={len(result.steps)}")
    check("LangGraph：答案非空", bool(result.answer), result.answer)
    check("LangGraph：error 为空", result.error == "", result.error)
    check("LangGraph：调用 search_chunks",
          [call["name"] for call in result.tool_calls] == ["search_chunks"],
          str([call["name"] for call in result.tool_calls]))
    check("LangGraph：引用抽取与 simple 一致",
          [citation["chunk_id"] for citation in result.citations] == ["paper_1#2.0", "paper_2#2.1"],
          str([citation["chunk_id"] for citation in result.citations]))

    # 综述任务在 LangGraph 下同样多步（list_documents → generate_review → 答案）
    result = runner.run("请基于已上传论文生成一篇 Transformer 综述")
    check("LangGraph：综述任务 3 步",
          [call["name"] for call in result.tool_calls] == ["list_documents", "generate_review"]
          and len(result.steps) == 3,
          f"steps={len(result.steps)}, tools={[c['name'] for c in result.tool_calls]}")

    # 工厂 auto 模式：langgraph 可用时应优先选择
    factory_runner = create_agent_runner(llm=ScriptedMockLLM(), tools=backend.build_tools())
    check("create_agent_runner(auto)：langgraph 可用时优先 langgraph",
          factory_runner.framework == "langgraph", factory_runner.framework)
    check("create_agent_runner(auto)：运行结果 framework=langgraph",
          factory_runner.run("数据集是什么？").framework == "langgraph")


# ---------------------------------------------------------------------------
# 4. 边界场景
# ---------------------------------------------------------------------------


def test_edge_cases() -> None:
    """测试 4：无工具可用 / LLM 报错 / max_steps=1 / to_dict JSON"""
    print("\n" + "=" * 70)
    print("测试 4：边界场景（无工具 / LLM 报错 / max_steps / JSON 序列化）")
    print("=" * 70)

    # 4a. 无工具可用：不抛异常，给出兜底答案
    runner = AgentRunner(llm=ScriptedMockLLM(), tools=ToolRegistry(), framework="simple")
    result = runner.run("这篇论文用了什么数据集？")
    check("无工具可用：不抛异常且答案非空", bool(result.answer), result.answer)
    check("无工具可用：无工具调用", result.tool_calls == [], str(result.tool_calls))
    check("无工具可用：error 为空", result.error == "", result.error)

    # 4b. LLM 报错：error 透传，不抛异常
    runner = AgentRunner(llm=FailingLLM(), tools=FakeBackend().build_tools(), framework="simple")
    result = runner.run("数据集是什么？")
    check("LLM 报错：error 透传", "模拟 LLM 服务故障" in result.error, result.error)
    check("LLM 报错：保留已产生步骤", len(result.steps) == 1, f"steps={len(result.steps)}")
    check("LLM 报错：不抛异常", isinstance(result.answer, str))

    # 4c. max_steps=1：不抛异常，兜底答案包含最后工具结果
    runner = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=FakeBackend().build_tools(),
        framework="simple",
        max_steps=1,
    )
    result = runner.run("这篇论文用了什么数据集和模型？")
    check("max_steps=1：steps 恰为 1", len(result.steps) == 1, f"steps={len(result.steps)}")
    check("max_steps=1：答案非空且提示已达最大步数",
          "已达到最大步数" in result.answer, result.answer[:120])
    check("max_steps=1：error 为空", result.error == "", result.error)

    # 4d. 空问题：不抛异常，返回明确 error
    result = runner.run("   ")
    check("空问题：返回 error 不抛异常", "问题不能为空" in result.error, result.error)

    # 4e. to_dict JSON-safe
    result = AgentRunner(
        llm=ScriptedMockLLM(),
        tools=FakeBackend().build_tools(),
        framework="simple",
    ).run("数据集是什么？")
    payload = result.to_dict()
    expected_keys = {
        "answer", "steps", "tool_calls", "citations", "trace_id", "model",
        "input_tokens", "output_tokens", "cost_yuan", "mock_mode",
        "framework", "latency_ms", "error",
    }
    check("to_dict 顶层 key 与契约一致", set(payload) == expected_keys, str(sorted(payload)))
    try:
        dumped = json.dumps(payload, ensure_ascii=False)
        json_ok = True
    except TypeError as exc:  # pragma: no cover - 正常不会触发
        dumped = ""
        json_ok = False
        print(f"  json.dumps 失败: {exc}")
    check("to_dict 结果可 json.dumps", json_ok and bool(dumped))
    check(
        "to_dict 嵌套结构正确",
        isinstance(payload["steps"], list)
        and payload["steps"]
        and isinstance(payload["steps"][0]["tool_calls"], list)
        and all(
            {"name", "arguments", "result", "error", "latency_ms", "ok"}
            <= set(call)
            for call in payload["steps"][0]["tool_calls"]
        ),
        "steps/tool_calls 结构不符合契约",
    )
    check(
        "flat tool_calls 字段齐全",
        all(
            {"step", "name", "arguments", "result_preview", "latency_ms", "ok"} <= set(call)
            for call in payload["tool_calls"]
        ),
        str(payload["tool_calls"]),
    )
    check(
        "citations 字段齐全",
        all({"chunk_id", "doc_id", "text_preview"} <= set(citation)
            for citation in payload["citations"]),
        str(payload["citations"]),
    )

    # 4f. 工厂函数默认 mock（无 key）
    llm = create_tool_calling_llm("auto")
    check("create_tool_calling_llm('auto') 无 key 时为 ScriptedMockLLM",
          isinstance(llm, ScriptedMockLLM), type(llm).__name__)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> int:
    """执行全部验证并返回进程退出码（0=全过）"""
    print("=" * 70)
    print("Agent 核心链路集成验证（冻结契约 v1，离线 mock）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)

    t0 = time.time()
    test_tool_registry()
    test_simple_loop()
    test_langgraph_loop()
    test_edge_cases()
    elapsed_ms = (time.time() - t0) * 1000

    print("\n" + "=" * 70)
    print(f"通过 {PASS} / 失败 {FAIL}  |  总耗时 {elapsed_ms:.1f}ms")
    if FAILURES:
        print("失败断言：")
        for name in FAILURES:
            print(f"  - {name}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

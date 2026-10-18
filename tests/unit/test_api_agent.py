"""后端 Agent API + VLM 可选解析链路 pytest 测试（全部 mock_mode=True）。

覆盖 task-5 冻结契约：
- GET /health components 含 agent；
- GET /agent/tools 返回 4 个工具（OpenAI tools schema 名称正确）；
- mock 模式上传（use_vlm=False）→ POST /agent 完成工具调用并返回答案；
- use_vlm=True mock 上传 → 走 VLM 解析链路（vlm_used/formulas/parse_mode）；
- POST /agent/tools/call 调 search_chunks 返回 chunks、未知工具返回 error；
- 超长 question（>2000）被拒、max_steps=0/99 被 clamp 且不崩；
- Agent 运行成本写入 /cost/summary 的 agent.run 维度。

全部用例不触发任何真实 DashScope 调用（conftest 已清空 API key，
VLM 链路在 mock 模式下强制 QwenVlmClient mock）。

运行：
    pytest tests/unit/test_api_agent.py -v
"""

from __future__ import annotations

import base64

import pytest

from app.api import (
    MAX_AGENT_QUESTION_LEN,
    MAX_AGENT_STEPS,
    MAX_VLM_PAGES,
    AgentBody,
    AgentResponse,
    UploadResponse,
    create_app,
)

# 各用例使用不同字节内容，避免 doc_id（MD5）在不同用例间串号
FAKE_PDF = b"%PDF-1.4 fake pdf for agent/vlm tests"


@pytest.fixture()
def client():
    """mock 模式 FastAPI TestClient（fastapi 未安装时跳过）"""
    app = create_app(mock_mode=True)
    if app is None:  # pragma: no cover - 环境缺 fastapi
        pytest.skip("fastapi 未安装，跳过 API 测试")
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def _upload(
    client,
    filename: str,
    raw: bytes,
    *,
    use_vlm: bool = False,
) -> dict:
    """上传 base64 编码的假 PDF，返回响应 JSON"""
    payload = {
        "filename": filename,
        "file_b64": base64.b64encode(raw).decode("ascii"),
        "user_id": "u_agent_test",
    }
    if use_vlm:
        payload["use_vlm"] = True
    resp = client.post("/upload", json=payload)
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# 0. 模块级 schema / 常量（契约默认值必须稳定）
# ---------------------------------------------------------------------------


def test_agent_vlm_schema_and_limits() -> None:
    """AgentBody / AgentResponse / UploadResponse 默认值 + 常量"""
    body = AgentBody(question="q")
    assert body.doc_id == ""
    assert body.user_id == ""
    assert body.max_steps == 6
    assert body.framework == "auto"
    assert body.history == []

    resp = AgentResponse()
    assert resp.answer == ""
    assert resp.tool_calls == []
    assert resp.framework == "auto"
    assert resp.error == ""

    upload = UploadResponse()
    # 新增字段默认值：保证既有调用方零改动
    assert upload.vlm_used is False
    assert upload.formulas == []
    assert upload.parse_mode == ""

    assert MAX_AGENT_QUESTION_LEN == 2000
    assert MAX_AGENT_STEPS == 8
    assert MAX_VLM_PAGES == 10


# ---------------------------------------------------------------------------
# 1. GET /health 增加 agent 组件
# ---------------------------------------------------------------------------


def test_health_has_agent_component(client) -> None:
    """/health.components 含 agent 且 mock 模式标记为 mock"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "agent" in data["components"]
    assert data["components"]["agent"] == "mock"
    # 既有组件不能丢（兼容性）
    for key in ("parser", "rag", "review", "llm", "translation"):
        assert key in data["components"]


# ---------------------------------------------------------------------------
# 2. GET /agent/tools（4 个工具 + schema 名称）
# ---------------------------------------------------------------------------


def test_agent_tools_specs(client) -> None:
    """GET /agent/tools 返回 4 个工具，OpenAI tools 格式名称正确"""
    resp = client.get("/agent/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    tools = data["tools"]
    assert len(tools) == 4

    names = {tool["function"]["name"] for tool in tools}
    assert names == {
        "list_documents",
        "search_chunks",
        "translate_text",
        "generate_review",
    }
    for tool in tools:
        assert tool["type"] == "function"
        function = tool["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"

    params = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    assert params["search_chunks"]["required"] == ["query"]
    assert params["translate_text"]["required"] == ["text"]
    assert params["generate_review"]["required"] == ["topic"]
    assert params["list_documents"]["required"] == []


# ---------------------------------------------------------------------------
# 3. mock 上传（文本层）→ POST /agent 工具调用问答
# ---------------------------------------------------------------------------


def test_agent_ask_mock_mode(client) -> None:
    """文本层上传后 /agent 问数据集：answer 非空 + 工具调用 + mock 标记"""
    upload = _upload(client, "agent_paper.pdf", FAKE_PDF, use_vlm=False)
    assert upload["error"] == ""
    # use_vlm=False 默认行为不变（不触发 VLM 链路）
    assert upload["vlm_used"] is False
    doc_id = upload["doc_id"]

    resp = client.post("/agent", json={
        "question": "本文的数据集是什么？",
        "doc_id": doc_id,
        "user_id": "u_agent_test",
        "max_steps": 6,
        "framework": "auto",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "", f"Agent 运行失败: {data['error']}"
    assert data["answer"], "Agent 答案不能为空"
    assert data["mock_mode"] is True
    assert data["framework"] in ("simple", "langgraph")
    assert data["tool_calls"], "Agent 应至少调用一次工具"
    called = {call["name"] for call in data["tool_calls"]}
    assert called & {"search_chunks", "list_documents"}, f"实际调用: {called}"
    # 契约字段齐全（前端 / MCP 直接透传 to_dict()）
    for key in (
        "answer", "steps", "tool_calls", "citations", "trace_id", "model",
        "input_tokens", "output_tokens", "cost_yuan", "mock_mode", "framework",
        "latency_ms", "error",
    ):
        assert key in data, f"AgentResult.to_dict() 缺字段 {key}"
    assert isinstance(data["steps"], list)
    assert isinstance(data["tool_calls"], list)
    # 文档已被索引，list_documents 结果条数应与 /documents 一致
    docs = client.get("/documents").json()["docs"]
    assert any(item["doc_id"] == doc_id for item in docs)


def test_agent_cost_recorded(client) -> None:
    """Agent 运行写入成本记录（step="agent.run"，user_id/paper_id 维度）"""
    upload = _upload(client, "agent_cost.pdf", FAKE_PDF + b" cost")
    resp = client.post("/agent", json={
        "question": "请用 list_documents 看看有哪些论文",
        "doc_id": upload["doc_id"],
        "user_id": "u_agent_cost",
    })
    assert resp.status_code == 200
    assert resp.json()["error"] == ""

    state = client.app.state.app_state
    records = state._ensure_cost_tracker().get_records(step="agent.run")
    assert len(records) == 1
    assert records[0].user_id == "u_agent_cost"
    assert records[0].paper_id == upload["doc_id"]
    assert records[0].from_cache is False


def test_agent_question_validation(client) -> None:
    """空 question / 超长 question（>2000）被拒，不返回 500"""
    empty = client.post("/agent", json={"question": "   "})
    assert empty.status_code == 200
    assert empty.json()["error"] != ""

    too_long = client.post("/agent", json={"question": "x" * (MAX_AGENT_QUESTION_LEN + 1)})
    assert too_long.status_code == 200
    data = too_long.json()
    assert data["error"] != ""
    assert str(MAX_AGENT_QUESTION_LEN) in data["error"]
    assert data["answer"] == ""


def test_agent_max_steps_clamped(client) -> None:
    """max_steps=0 / 99 被 clamp 到 [1, 8]，请求不崩"""
    for max_steps in (0, 99):
        resp = client.post("/agent", json={
            "question": "本文的实验结论是什么？",
            "max_steps": max_steps,
            "framework": "simple",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "", f"max_steps={max_steps} 运行失败: {data['error']}"
        assert data["answer"]
        assert data["framework"] in ("simple", "langgraph")


def test_agent_invalid_doc_id_does_not_crash(client) -> None:
    """doc_id 不存在时工具返回空结果，Agent 正常回答且不返回 500"""
    resp = client.post("/agent", json={
        "question": "这篇论文的数据集是什么？",
        "doc_id": "doc_not_indexed",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "", f"Agent 运行失败: {data['error']}"
    assert data["answer"]


# ---------------------------------------------------------------------------
# 4. POST /upload use_vlm=True（mock VLM 演示链路）
# ---------------------------------------------------------------------------


def test_upload_use_vlm_mock_demo(client) -> None:
    """use_vlm=True mock 上传：vlm_used=True + formulas 预览 + chunks + parse_mode=vlm"""
    data = _upload(client, "vlm_paper.pdf", FAKE_PDF + b" vlm", use_vlm=True)
    assert data["error"] == ""
    assert data["vlm_used"] is True
    assert data["parse_mode"] == "vlm"
    assert data["num_pages"] > 0
    assert data["num_chunks"] > 0
    assert len(data["chunks"]) == data["num_chunks"]
    assert data["chunks"][0]["chunk_id"]

    formulas = data["formulas"]
    assert formulas, "VLM 链路应返回公式预览"
    assert len(formulas) <= 20
    for formula in formulas:
        for key in ("latex", "confidence", "page_no", "paragraph_idx", "need_review"):
            assert key in formula, f"公式预览缺字段 {key}"
        assert isinstance(formula["latex"], str) and formula["latex"]
        assert 0.0 <= float(formula["confidence"]) <= 1.0
        assert isinstance(formula["page_no"], int)
        assert isinstance(formula["need_review"], bool)
    # 含低置信公式 → need_review=True；含高置信公式 → formula_resolved=True
    assert any(f["confidence"] >= 0.7 for f in formulas)
    assert data["formula_resolved"] is True
    assert data["need_review"] is True

    # VLM 上传的文档同样进入进程内索引，可被 /agent 检索
    docs = client.get("/documents").json()["docs"]
    assert any(item["doc_id"] == data["doc_id"] for item in docs)


def test_upload_use_vlm_real_pdf_chain(client) -> None:
    """真实 PDF + use_vlm=True（mock VLM）：pdf_to_image → VLM → FusionEngine 链路可跑通

    用 PyMuPDF 现场生成一页 PDF，验证 fake PDF 走不到的完整解析编排；
    fitz 未安装时跳过（链路的 mock 演示路径已由上一用例覆盖）。
    """
    fitz = pytest.importorskip("fitz", reason="PyMuPDF 未安装，跳过真实 PDF VLM 链路")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Attention Is All You Need. ImageNet top-5 85.6%.")
    page.insert_text((72, 140), "We propose a Transformer encoder with self-attention.")
    raw = doc.tobytes()
    doc.close()

    data = _upload(client, "real_vlm.pdf", raw, use_vlm=True)
    assert data["error"] == ""
    assert data["vlm_used"] is True
    assert data["parse_mode"] == "vlm"
    assert data["num_pages"] == 1
    assert data["num_chunks"] > 0
    assert data["formulas"], "真实 PDF 链路应产出 VLM 公式预览"
    assert data["chunks"][0]["text"]


def test_upload_default_use_vlm_unchanged(client) -> None:
    """不传 use_vlm（默认 False）时既有上传行为完全不变"""
    data = _upload(client, "plain_paper.pdf", FAKE_PDF + b" plain")
    assert data["error"] == ""
    assert data["vlm_used"] is False
    assert data["formulas"] == []
    assert data["num_chunks"] > 0
    assert data["formula_resolved"] is True


# ---------------------------------------------------------------------------
# 5. POST /agent/tools/call（MCP server 适配）
# ---------------------------------------------------------------------------


def test_agent_tools_call_search_chunks(client) -> None:
    """/agent/tools/call 调 search_chunks 返回 chunks（含 score/text 字段）"""
    upload = _upload(client, "tool_paper.pdf", FAKE_PDF + b" tools")
    resp = client.post("/agent/tools/call", json={
        "name": "search_chunks",
        "arguments": {
            "query": "ImageNet 数据集准确率",
            "doc_id": upload["doc_id"],
            "top_k": 3,
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "search_chunks"
    assert data["error"] == ""
    assert isinstance(data["latency_ms"], int)
    chunks = data["result"]["chunks"]
    assert chunks, "search_chunks 应返回检索片段"
    assert len(chunks) <= 3
    first = chunks[0]
    for key in ("chunk_id", "doc_id", "text", "section", "score"):
        assert key in first
    assert first["chunk_id"]
    assert first["doc_id"] == upload["doc_id"]


def test_agent_tools_call_unknown_tool(client) -> None:
    """未知工具名返回 error 非空（不抛 500）"""
    resp = client.post("/agent/tools/call", json={
        "name": "run_shell_command",
        "arguments": {"cmd": "rm -rf /"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "run_shell_command"
    assert data["error"] != ""
    # 未知工具不执行：结果为 None 或空 dict（ToolExecution.result 默认空 dict）
    assert data["result"] in (None, {})
    assert data["latency_ms"] >= 0


def test_agent_tools_call_translate_text(client) -> None:
    """/agent/tools/call 调 translate_text 返回译文 + 术语对齐"""
    resp = client.post("/agent/tools/call", json={
        "name": "translate_text",
        "arguments": {"text": "Self-attention is the core of Transformer."},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    result = data["result"]
    assert result["translated"]
    assert result["target_lang"] == "zh-CN"
    assert isinstance(result["aligned_terms"], list)
    assert result["mock_mode"] is True


def test_agent_tools_call_generate_review(client) -> None:
    """/agent/tools/call 调 generate_review 返回综述摘要（mock 模式强制 mock pipeline）"""
    upload = _upload(client, "review_tool_paper.pdf", FAKE_PDF + b" review tool")
    resp = client.post("/agent/tools/call", json={
        "name": "generate_review",
        "arguments": {
            "topic": "Transformer 在视觉与语言任务中的应用",
            "doc_ids": [upload["doc_id"]],
            "num_sections": 2,
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    result = data["result"]
    assert result["title"]
    assert result["full_text"]
    assert result["mock_mode"] is True
    assert isinstance(result["citation_stats"], dict)


def test_agent_tools_call_missing_required_argument(client) -> None:
    """缺少必填参数时 error 非空（参数校验失败不执行工具）"""
    resp = client.post("/agent/tools/call", json={
        "name": "translate_text",
        "arguments": {},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] != ""

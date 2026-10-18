"""MCP stdio server 单元测试（不启动子进程、不需要后端）。

覆盖：initialize / notification / tools/list / tools/call 成功与错误 /
未知方法 / JSON 解析错误 / OpenAI→MCP schema 转换。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_mcp_module():
    path = PROJECT_ROOT / "scripts" / "agent_mcp_server.py"
    spec = importlib.util.spec_from_file_location("agent_mcp_server_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mcp = _load_mcp_module()


def test_initialize_returns_protocol_version() -> None:
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "paper-agent-mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_notification_returns_none() -> None:
    assert mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping() -> None:
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 9, "method": "ping"})
    assert resp["result"] == {}


def test_tools_list_uses_local_specs(monkeypatch) -> None:
    spec = [
        {
            "type": "function",
            "function": {
                "name": "search_chunks",
                "description": "检索论文片段",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    monkeypatch.setattr(mcp, "load_tool_specs", lambda: [mcp.openai_spec_to_mcp(s) for s in spec])
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = resp["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "search_chunks"
    assert tools[0]["inputSchema"]["type"] == "object"


def test_tools_call_success(monkeypatch) -> None:
    def fake_call(api_base, name, arguments, timeout=60.0):
        assert name == "search_chunks"
        assert arguments["query"] == "数据集"
        return {"name": name, "result": {"chunks": [{"chunk_id": "c1"}]}, "error": "", "latency_ms": 3}

    monkeypatch.setattr(mcp, "call_backend_tool", fake_call)
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_chunks", "arguments": {"query": "数据集"}},
        },
        api_base="http://example.invalid",
    )
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["result"]["chunks"][0]["chunk_id"] == "c1"


def test_tools_call_backend_error(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp, "call_backend_tool", lambda *a, **k: {"error": "后端不可达"},
    )
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "search_chunks", "arguments": {"query": "x"}},
        }
    )
    assert resp["result"]["isError"] is True


def test_tools_call_invalid_params() -> None:
    resp = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"arguments": {}}}
    )
    assert resp["error"]["code"] == -32602


def test_unknown_method() -> None:
    resp = mcp.handle_message({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    assert resp["error"]["code"] == -32601


def test_parse_error_and_blank_line() -> None:
    assert mcp.handle_line("   ") is None
    resp = mcp.handle_line("{not json")
    assert resp["error"]["code"] == -32700


def test_openai_spec_to_mcp() -> None:
    spec = {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "翻译",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
    }
    mcp_spec = mcp.openai_spec_to_mcp(spec)
    assert mcp_spec["name"] == "translate_text"
    assert mcp_spec["inputSchema"]["properties"]["text"]["type"] == "string"
    assert "parameters" not in mcp_spec

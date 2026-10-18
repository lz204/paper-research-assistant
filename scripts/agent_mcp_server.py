"""最小 MCP（Model Context Protocol）stdio server：把本项目的 Agent 工具暴露给任意 MCP 客户端。

为什么手写而不用官方 ``mcp`` 包：
    官方 mcp 包依赖 ``starlette>=0.27``（新版会解析到 1.x），而本项目锁定
    ``fastapi==0.110.0`` / ``starlette==0.36.3``（见 requirements.lock）。
    安装 mcp 会把 starlette 升级到不兼容版本，直接导致 FastAPI TestClient 崩溃。
    这里按 MCP 2024-11-05 stdio 传输规范手写 JSON-RPC 2.0 帧处理，零新增依赖。

协议要点（stdio transport）：
    - stdin/stdout 按行分隔的 JSON，每行一条 JSON-RPC 2.0 消息（UTF-8）
    - 支持方法：initialize / notifications/initialized / ping / tools/list / tools/call
    - tools/list 读取本地 app.agent.default_tool_specs()（无需后端在线）
    - tools/call 通过 HTTP 转发到 FastAPI 的 POST /agent/tools/call（后端持有文档索引）

用法：
    python scripts/agent_mcp_server.py --api-base http://127.0.0.1:8000

Claude Desktop 等 MCP 客户端配置示例：
    {
      "mcpServers": {
        "paper-agent": {
          "command": "python",
          "args": ["scripts/agent_mcp_server.py",
                   "--api-base", "http://127.0.0.1:8000"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "paper-agent-mcp"
SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# 工具 schema：OpenAI tools 格式 → MCP tools 格式
# ---------------------------------------------------------------------------


def openai_spec_to_mcp(spec: Dict[str, Any]) -> Dict[str, Any]:
    """把 OpenAI function-calling 的 tool spec 转成 MCP tools/list 的条目格式。"""
    fn = spec.get("function") or {}
    return {
        "name": str(fn.get("name") or ""),
        "description": str(fn.get("description") or ""),
        "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def load_tool_specs() -> List[Dict[str, Any]]:
    """读取本地工具 schema（app.agent 尚未就绪时返回空列表，不抛异常）。"""
    try:
        from app.agent import default_tool_specs  # type: ignore

        return [openai_spec_to_mcp(s) for s in default_tool_specs()]
    except Exception:  # noqa: BLE001 - MCP server 不应因工具模块缺失而崩溃
        return []


# ---------------------------------------------------------------------------
# 后端调用：tools/call → FastAPI POST /agent/tools/call
# ---------------------------------------------------------------------------


def call_backend_tool(
    api_base: str,
    name: str,
    arguments: Dict[str, Any],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """调用后端工具执行接口；任何异常都转成 ``{"error": ...}`` 返回。"""
    try:
        import httpx  # type: ignore
    except ImportError:  # pragma: no cover - httpx 在 requirements 中必有
        return {"error": "httpx 未安装，无法调用后端"}

    url = api_base.rstrip("/") + "/agent/tools/call"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json={"name": name, "arguments": arguments})
            if resp.status_code != 200:
                return {"error": f"后端返回 HTTP {resp.status_code}: {resp.text[:200]}"}
            data = resp.json()
            if not isinstance(data, dict):
                return {"error": f"后端响应格式非法: {type(data).__name__}"}
            return data
    except Exception as e:  # noqa: BLE001 - 网络错误不抛给 MCP 客户端
        return {"error": f"调用后端失败（{api_base}）: {e}"}


# ---------------------------------------------------------------------------
# JSON-RPC 处理
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(
    msg: Dict[str, Any],
    *,
    api_base: str = "http://127.0.0.1:8000",
    timeout: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """处理单条 JSON-RPC 消息；notification（无 id）返回 None（不需要响应）。"""
    if not isinstance(msg, dict):
        return _error(None, -32600, "Invalid Request: 消息必须是 JSON 对象")

    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # JSON-RPC 通知（带 method 无 id）：MCP initialized / cancelled 等，不回包
    if msg_id is None and isinstance(method, str):
        return None

    if method == "initialize":
        return _result(
            msg_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "ping":
        return _result(msg_id, {})

    if method == "tools/list":
        return _result(msg_id, {"tools": load_tool_specs()})

    if method == "tools/call":
        if not isinstance(params, dict):
            return _error(msg_id, -32602, "Invalid params: params 必须是对象")
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not name:
            return _error(msg_id, -32602, "Invalid params: 缺少 name")
        if not isinstance(arguments, dict):
            return _error(msg_id, -32602, "Invalid params: arguments 必须是对象")

        tool_result = call_backend_tool(api_base, name, arguments, timeout=timeout)
        is_error = bool(tool_result.get("error"))
        # MCP tools/call 返回 content 数组；把结构化结果序列化为文本
        content = [
            {
                "type": "text",
                "text": json.dumps(tool_result, ensure_ascii=False, default=str)[:8000],
            }
        ]
        return _result(msg_id, {"content": content, "isError": is_error})

    return _error(msg_id, -32601, f"Method not found: {method}")


def handle_line(
    line: str,
    *,
    api_base: str = "http://127.0.0.1:8000",
    timeout: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """处理一行输入（空行返回 None；JSON 解析失败返回 -32700 Parse error）。"""
    text = (line or "").strip()
    if not text:
        return None
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as e:
        return _error(None, -32700, f"Parse error: {e}")
    return handle_message(msg, api_base=api_base, timeout=timeout)


def main(argv: Optional[List[str]] = None) -> int:
    """stdio 主循环：逐行读取 JSON-RPC 消息，响应写 stdout。"""
    parser = argparse.ArgumentParser(description="论文研读 Agent MCP stdio server")
    parser.add_argument(
        "--api-base", default="http://127.0.0.1:8000",
        help="FastAPI 后端地址（tools/call 转发目标）",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="后端请求超时（秒）")
    args = parser.parse_args(argv)

    for raw_line in sys.stdin:
        response = handle_line(raw_line, api_base=args.api_base, timeout=args.timeout)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

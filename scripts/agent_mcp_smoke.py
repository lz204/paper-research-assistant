"""MCP stdio server 冒烟脚本：起子进程 → initialize → tools/list → （可选）tools/call。

用法：
    # 只验证协议帧与 tools/list（不需要后端）
    python scripts/agent_mcp_smoke.py

    # 连同 tools/call 一起验证（需要先启动 FastAPI：.\\scripts\\dev.ps1 api）
    python scripts/agent_mcp_smoke.py --call search_chunks --query "本文用了什么数据集"

返回码：0=协议帧全部正确；1=有断言失败；2=无法启动 server。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER = PROJECT_ROOT / "scripts" / "agent_mcp_server.py"


def _build_payload(call_tool: str, query: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-mcp-smoke", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    if call_tool:
        messages.append(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": call_tool,
                    "arguments": {"query": query, "top_k": 3},
                },
            }
        )
    return messages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MCP stdio server 冒烟")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--call", default="", help="可选：额外调用某工具（如 search_chunks）")
    parser.add_argument("--query", default="本文用了什么数据集")
    args = parser.parse_args(argv)

    if not SERVER.exists():
        print(f"[FAIL] 找不到 MCP server: {SERVER}")
        return 2

    payload = "\n".join(json.dumps(m, ensure_ascii=False) for m in _build_payload(args.call, args.query)) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, str(SERVER), "--api-base", args.api_base],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 无法启动 MCP server: {e}")
        return 2

    if proc.returncode != 0:
        print(f"[FAIL] server 退出码 {proc.returncode}，stderr:\n{proc.stderr[-2000:]}")
        return 2

    responses: Dict[int, Dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"[FAIL] server 输出不是合法 JSON: {line[:200]}")
            return 1
        if isinstance(msg, dict) and "id" in msg:
            responses[int(msg["id"])] = msg

    failed = 0

    init = responses.get(1, {})
    if init.get("result", {}).get("protocolVersion") != "2024-11-05":
        print(f"[FAIL] initialize 响应异常: {init}")
        failed += 1
    else:
        print(f"[PASS] initialize: protocolVersion=2024-11-05, "
              f"server={init['result']['serverInfo']['name']}")

    tools_resp = responses.get(2, {})
    tools = (tools_resp.get("result") or {}).get("tools") or []
    if not tools:
        print(f"[FAIL] tools/list 为空（app.agent 是否已实现？）: {tools_resp}")
        failed += 1
    else:
        names = [t.get("name") for t in tools]
        print(f"[PASS] tools/list: {len(tools)} 个工具 {names}")

    if args.call:
        call_resp = responses.get(3, {})
        result = call_resp.get("result") or {}
        if "content" not in result:
            print(f"[FAIL] tools/call 响应异常: {call_resp}")
            failed += 1
        elif result.get("isError"):
            text = (result["content"][0].get("text") or "")[:300]
            print(f"[WARN] tools/call 返回 isError（后端是否已启动？）: {text}")
        else:
            text = (result["content"][0].get("text") or "")[:300]
            print(f"[PASS] tools/call: {text}")

    print("[OK] MCP 冒烟完成" if failed == 0 else f"[FAIL] {failed} 项失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

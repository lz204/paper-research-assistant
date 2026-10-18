"""FastAPI 后端 HTTP 客户端（API-first 架构的唯一前端出口）。

项目文档第 3 / 10 章明确「FastAPI 单体 + Streamlit 前端」的 API-first 架构：
前端的 PDF 解析、RAG 问答、综述生成、多语种翻译、成本看板等业务能力，
都应通过 HTTP 调用后端执行，而不是在 Streamlit 进程内直接 import 业务模块。
本模块把「调用后端」这件事收敛到一处，Streamlit 页面只依赖 ApiClient。

设计要点：
- base_url 默认取环境变量 ``API_BASE_URL``（docker-compose 中为 ``http://app:8000``），
  未配置时回退本地 ``http://127.0.0.1:8000``；
- 连接超时 3 秒（后端未启动时快速失败，避免页面卡住），
  读超时使用构造参数 timeout（默认 180 秒，覆盖综述生成 / 真实 LLM 长耗时）；
- 全链路降级：网络错误、超时、非 JSON 响应、HTTP 4xx/5xx 一律转换成
  ``{"error": "..."}`` 字典返回，绝不向 Streamlit 页面抛异常；
  页面据此展示 st.error，并引导用户启动后端或切换到「本地进程模式」；
- ``available`` 属性带约 3 秒缓存，避免每次页面交互都请求一次 /health。

后端契约 v1（由 Lead 冻结）：
    GET  /health        → {"status": ..., "version", "mock_mode", "components"}
    POST /upload        → {"doc_id", "paper_id", "title", "num_pages", "num_chunks",
                           "formula_resolved", "need_review", "chunks", "error"}
    POST /ask           → {"answer", "citations", "trace_id", "from_cache",
                           "latency_ms", "model", "input_tokens", "output_tokens",
                           "cost_yuan", "mock_mode", "memory_used", "error"}
    POST /review        → {"title", "full_text", "full_html", "outline", "paragraphs",
                           "citation_stats", "references", "verified", "docx_path",
                           "trace_id", "latency_ms", "docx_b64", "docx_mode",
                           "docx_name", "docx_size", "mock_mode", "error"}
    POST /translate     → {"translated", "source_text", "aligned_terms",
                           "target_lang", "mock_mode", "latency_ms", "error"}
    POST /agent         → {"answer", "steps", "tool_calls", "citations", "trace_id",
                           "model", "input_tokens", "output_tokens", "cost_yuan",
                           "mock_mode", "framework", "latency_ms", "error"}
    GET  /agent/tools   → {"tools": [OpenAI tools 格式...], "error"}
    GET  /documents     → {"docs": [{"doc_id", "title", "num_chunks"}], "error"}
    GET  /cost/summary  → {"total_cost", "daily", "by_model", "by_user",
                           "by_paper", "cache", "error"}

用法::

    from app.frontend.api_client import ApiClient

    client = ApiClient()
    if client.available:
        resp = client.ask("doc_xxx", "什么是 Transformer？")
        if resp.get("error"):
            st.error(resp["error"])
        else:
            st.write(resp["answer"])
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# 后端地址默认值：本地开发端口（docker-compose 用 API_BASE_URL 覆盖）
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# 连接超时（秒）：后端未启动时快速失败
CONNECT_TIMEOUT = 3.0
# available 属性缓存时长（秒）
HEALTH_CACHE_TTL = 3.0
# 错误信息截断长度，避免 st.error 里塞入超长响应正文
MAX_ERROR_TEXT = 300


class ApiClient:
    """FastAPI 后端 HTTP 客户端（API-first 前端唯一出口）。

    Attributes:
        base_url: 后端根地址（已去掉尾部斜杠）
        timeout: 读超时（秒）
    """

    def __init__(self, base_url: str = "", timeout: float = 180.0):
        """初始化客户端。

        Args:
            base_url: 后端根地址；为空时取环境变量 API_BASE_URL，
                仍未配置则回退 DEFAULT_BASE_URL。尾部斜杠会被去掉。
            timeout: 读超时（秒），连接超时固定 3 秒。
        """
        resolved = (
            base_url or os.getenv("API_BASE_URL", "") or DEFAULT_BASE_URL
        ).strip()
        self.base_url = resolved.rstrip("/")
        self.timeout = float(timeout)
        # available 属性缓存：None 表示尚未探测
        self._health_cached: Optional[bool] = None
        self._health_checked_at: float = 0.0

    # ------------------------------------------------------------------
    # 健康检查 / 可用性
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """GET /health 健康检查（失败不抛异常）。

        Returns:
            成功：后端返回的 JSON 字典（保证含 ``ok=True``）。
            失败：``{"ok": False, "error": "..."}``。
        """
        data = self._request("GET", "/health")
        if data.get("error"):
            return {"ok": False, "error": data["error"]}
        data.setdefault("ok", True)
        return data

    @property
    def available(self) -> bool:
        """后端是否可达（带约 3 秒缓存，避免每次渲染都请求 /health）。"""
        now = time.monotonic()
        if (
            self._health_cached is not None
            and (now - self._health_checked_at) < HEALTH_CACHE_TTL
        ):
            return self._health_cached
        health = self.health()
        ok = bool(health.get("ok")) and health.get("status") == "ok"
        self._health_cached = ok
        self._health_checked_at = now
        return ok

    def clear_cache(self) -> None:
        """清空 available 缓存（切换后端 / 手动重试时调用）。"""
        self._health_cached = None
        self._health_checked_at = 0.0

    # ------------------------------------------------------------------
    # 业务接口
    # ------------------------------------------------------------------

    def upload_pdf(
        self,
        filename: str,
        raw: bytes,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """POST /upload：上传 PDF（内部做 base64 编码）。

        Args:
            filename: PDF 文件名
            raw: PDF 原始字节
            user_id: 用户 ID（用于后端成本按用户分摊）

        Returns:
            上传响应字典；失败时 ``{"error": "..."}``。
        """
        if not isinstance(raw, (bytes, bytearray)):
            return {"error": "upload_pdf 的 raw 参数必须是 bytes"}
        try:
            file_b64 = base64.b64encode(bytes(raw)).decode("ascii")
        except Exception as e:  # noqa: BLE001 - 兜底：编码失败也不能让页面崩
            logger.exception("PDF base64 编码失败")
            return {"error": f"PDF base64 编码失败: {e}"}
        return self._request(
            "POST",
            "/upload",
            {
                "filename": filename or "upload.pdf",
                "file_b64": file_b64,
                "user_id": user_id or "",
            },
        )

    def ask(
        self,
        paper_id: str,
        question: str,
        user_id: str = "",
        use_memory: bool = True,
        top_k: int = 5,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """POST /ask：论文问答（RAG 检索 + 记忆池 + 引用溯源）。

        Args:
            paper_id: 论文 / 文档 ID
            question: 用户提问（后端限制 2000 字，文档 6.5 节）
            user_id: 用户 ID
            use_memory: 是否启用会话记忆池
            top_k: RAG 检索条数
            history: 最近若干轮问答，结构 [{"q": ..., "a": ...}]

        Returns:
            问答响应字典；失败时 ``{"error": "..."}``。
        """
        return self._request(
            "POST",
            "/ask",
            {
                "paper_id": paper_id or "",
                "question": question or "",
                "user_id": user_id or "",
                "use_memory": bool(use_memory),
                "top_k": int(top_k),
                "history": list(history or []),
            },
        )

    def review(
        self,
        topic: str,
        doc_ids: List[str],
        user_id: str = "",
        num_sections: int = 4,
        verify_citations: bool = True,
        export_docx: bool = True,
    ) -> Dict[str, Any]:
        """POST /review：跨文献综述生成（含引用验证与 Word 导出）。

        Args:
            topic: 综述主题（后端限制 500 字，文档 6.5 节）
            doc_ids: 参与综述的文档 ID 列表（至少 2 篇）
            user_id: 用户 ID
            num_sections: 期望章节数
            verify_citations: 是否做引用事实验证（三色 NLI）
            export_docx: 是否让后端导出 Word / Markdown

        Returns:
            综述响应字典；失败时 ``{"error": "..."}``。
        """
        return self._request(
            "POST",
            "/review",
            {
                "topic": topic or "",
                "doc_ids": list(doc_ids or []),
                "user_id": user_id or "",
                "num_sections": int(num_sections),
                "verify_citations": bool(verify_citations),
                "export_docx": bool(export_docx),
            },
        )

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        user_id: str = "",
    ) -> Dict[str, Any]:
        """POST /translate：多语种文献翻译（术语统一校准）。

        Args:
            text: 待翻译文本（页面限制 5000 字）
            source_lang: 源语言代码（en / zh / ja / de / fr / ru）
            target_lang: 目标语言代码（默认 zh-CN）
            user_id: 用户 ID

        Returns:
            翻译响应字典；失败时 ``{"error": "..."}``。
        """
        return self._request(
            "POST",
            "/translate",
            {
                "text": text or "",
                "source_lang": source_lang or "en",
                "target_lang": target_lang or "zh-CN",
                "user_id": user_id or "",
            },
        )

    def agent(
        self,
        question: str,
        doc_id: str = "",
        user_id: str = "",
        max_steps: int = 6,
        framework: str = "auto",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """POST /agent：LLM 自主调用工具完成研读任务（ReAct / LangGraph 编排）。

        后端会自主规划「列文档 / 检索 / 翻译 / 综述」等工具调用，返回完整轨迹；
        读超时沿用客户端 timeout（默认 180 秒，覆盖多步工具调用 + 真实 LLM 长耗时）。

        Args:
            question: 用户任务 / 问题（后端限制 2000 字，文档 6.5 节）
            doc_id: 目标文档 ID；空字符串表示「全部文档」（不限定文档）
            user_id: 用户 ID（供后端按用户分摊成本）
            max_steps: Agent 最大步数（页面限制 1-8，默认 6）
            framework: 编排框架（auto / simple / langgraph）
            history: 最近若干轮问答，结构 [{"q": ..., "a": ...}]

        Returns:
            含 answer / steps / tool_calls / citations / framework 等字段的字典；
            失败时 ``{"error": "..."}``。
        """
        try:
            steps = int(max_steps)
        except (TypeError, ValueError):
            steps = 6
        steps = max(1, min(8, steps))
        return self._request(
            "POST",
            "/agent",
            {
                "question": question or "",
                "doc_id": doc_id or "",
                "user_id": user_id or "",
                "max_steps": steps,
                "framework": framework or "auto",
                "history": list(history or []),
            },
        )

    def agent_tools(self) -> Dict[str, Any]:
        """GET /agent/tools：列出 Agent 可调用的工具（OpenAI tools 格式）。

        Returns:
            ``{"tools": [{"type": "function", "function": {...}}, ...], "error": ""}``；
            不可达 / 后端未实现时 ``{"error": "..."}``。
        """
        return self._request("GET", "/agent/tools")

    def docs(self) -> Dict[str, Any]:
        """GET /documents：列出后端已解析的文档。

        注：业务端点使用 ``/documents``，``/docs`` 留给 FastAPI 自带的 Swagger UI。

        Returns:
            ``{"docs": [{"doc_id", "title", "num_chunks"}], ...}``；
            失败时 ``{"error": "..."}``。
        """
        return self._request("GET", "/documents")

    def cost_summary(self) -> Dict[str, Any]:
        """GET /cost/summary：成本看板聚合数据（文档 6.1 节）。

        Returns:
            ``{"total_cost", "daily", "by_model", "by_user", "by_paper",
            "cache", ...}``；失败时 ``{"error": "..."}``。
        """
        return self._request("GET", "/cost/summary")

    # ------------------------------------------------------------------
    # 内部：HTTP 请求与响应解析（所有异常在此收口，不向上抛）
    # ------------------------------------------------------------------

    def _timeout(self) -> httpx.Timeout:
        """构造 httpx 超时配置：连接 3 秒，读 / 写 / 连接池用 self.timeout。"""
        return httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行一次 HTTP 请求。

        任何异常（连接失败 / 超时 / 非 JSON / HTTP 错误）都转成
        ``{"error": "..."}`` 返回，绝不让前端页面崩溃。

        Args:
            method: HTTP 方法（GET / POST）
            path: 以 / 开头的接口路径
            payload: JSON 请求体（GET 传 None）

        Returns:
            解析后的 JSON 字典；失败时 ``{"error": "..."}``。
        """
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self._timeout()) as client:
                resp = client.request(method, url, json=payload)
        except httpx.ConnectError as e:
            return {
                "error": (
                    f"无法连接后端 {self.base_url}（{_short(e)}）；"
                    "请先启动 FastAPI 后端"
                )
            }
        except httpx.TimeoutException:
            return {
                "error": (
                    f"请求后端超时（{path}，读超时 {self.timeout:.0f}s）；"
                    "可稍后重试或切换到本地进程模式"
                )
            }
        except httpx.HTTPError as e:
            return {"error": f"HTTP 请求失败（{path}）：{_short(e)}"}
        except Exception as e:  # noqa: BLE001 - 兜底：绝不让前端崩溃
            logger.exception("API 请求异常: %s %s", method, url)
            return {"error": f"请求 {path} 异常：{_short(e)}"}
        return _parse_response(resp, path)


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------


def _parse_response(resp: httpx.Response, path: str) -> Dict[str, Any]:
    """把 httpx 响应解析成字典；HTTP 错误 / JSON 异常 → {"error": ...}。"""
    text = resp.text or ""
    try:
        data = resp.json()
    except ValueError:
        snippet = text.strip()[:MAX_ERROR_TEXT] or "(空响应)"
        return {
            "error": (
                f"后端 {path} 返回非 JSON（HTTP {resp.status_code}）：{snippet}"
            )
        }
    if not isinstance(data, dict):
        return {
            "error": (
                f"后端 {path} 返回了非对象 JSON：{type(data).__name__}"
            )
        }
    if resp.status_code >= 400:
        # FastAPI 校验错误是 {"detail": ...}，业务错误是 {"error": ...}
        detail = data.get("error") or data.get("detail") or f"HTTP {resp.status_code}"
        return {
            "error": f"后端 {path} 返回 HTTP {resp.status_code}：{_short(detail)}"
        }
    return data


def _short(obj: Any, limit: int = MAX_ERROR_TEXT) -> str:
    """把异常 / 响应内容压缩成单行短文本，避免页面里出现超长堆栈。"""
    text = str(obj).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


__all__ = ["ApiClient", "DEFAULT_BASE_URL"]

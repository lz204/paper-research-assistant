"""FastAPI 路由模块（项目文档第 3 / 10 章 + 6.5 节安全）

文档第 3 章架构图：
    User[用户提问] --> Q{问题路由}
    Q -->|全局性| Hier[层级摘要]
    Q -->|细节性| RAG[RAG 检索]
    Review[综述主题] --> RAG2[多文档检索]
    ... --> LLM[Qwen-Plus API] --> Out[回答+引用]

文档第 10 章技术栈：后端 = FastAPI + Celery（单体 + 异步任务）。
文档第 3 章："不上微服务。FastAPI 单体 + Celery 异步任务，单机跑得动"。

文档 6.5 节安全：
- API key 走 .env + 环境变量
- 用户提问限 2000 字，综述主题限 500 字
- 输出做基础 XSS 过滤

容错：fastapi 未安装时模块仍可 import（路由定义降级），
保证无 fastapi 依赖也能从 app.api 导入 schemas 数据类，
便于本地联调和单元测试。

设计：
- 请求/响应 schema dataclass：UploadRequest / AskRequest / ReviewRequest / TranslateRequest
- create_app：创建 FastAPI 应用（fastapi 可用时）
- 路由：POST /upload（可选 use_vlm 走 VLM 解析链路）、POST /ask、POST /review、
  POST /translate、POST /agent（Agent 工具调用）、GET /agent/tools、
  POST /agent/tools/call（供 MCP 适配器）、GET /health、GET /documents、
  GET /cost/summary
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查时导入，避免启动时加载重组件
    from ..rag.context_manager import ContextManager


# ---------------------------------------------------------------------------
# 文档 6.5 节安全限制
# ---------------------------------------------------------------------------


MAX_QUESTION_LEN = 2000     # 用户提问限 2000 字
MAX_REVIEW_TOPIC_LEN = 500  # 综述主题限 500 字
MAX_PDF_PAGES = 200         # 单 PDF 页数限制
MAX_PDF_SIZE_MB = 50        # 单 PDF 文件大小限制
MAX_TRANSLATE_LEN = 5000    # 单次翻译文本限 5000 字符

# VLM 可选解析链路：最多送前 10 页给 Qwen-VL-Max（文档 6.2 节成本治理：
# 单页 3-5 秒 + 按页计费，默认文本层解析，只有显式 use_vlm=True 才走 VLM）
MAX_VLM_PAGES = 10
# 公式置信度阈值（文档 4.1 节：低于 0.7 标 need_review，前端黄色高亮）
VLM_FORMULA_REVIEW_THRESHOLD = 0.7
# Agent 问答限制（文档 6.5 节安全：提问限 2000 字；max_steps 上限 8 防死循环）
MAX_AGENT_QUESTION_LEN = 2000
MAX_AGENT_STEPS = 8


# ---------------------------------------------------------------------------
# 请求/响应 schema（fastapi 不可用时可独立使用）
# ---------------------------------------------------------------------------


@dataclass
class UploadRequest:
    """PDF 上传请求（文档第 2 章"PDF 公式无损解析"）

    Attributes:
        filename: PDF 文件名
        file_b64: PDF 文件 base64 编码内容
        user_id: 用户 ID（可选，用于成本按用户分摊）
        use_vlm: 是否启用可选 VLM 解析链路（默认 False，仅文本层，避免 API 成本）
    """

    filename: str = ""
    file_b64: str = ""
    user_id: str = ""
    use_vlm: bool = False


@dataclass
class UploadResponse:
    """PDF 上传响应

    vlm_used / formulas / parse_mode 为 VLM 可选解析链路扩展字段（默认值保证向后兼容）：
    - vlm_used: 本次是否走了 VLM 解析链路
    - formulas: 公式预览（latex / confidence / page_no / paragraph_idx / need_review）
    - parse_mode: 解析模式，"text"（文本层）/ "vlm" / "text_fallback"（VLM 失败降级）
    """

    doc_id: str = ""
    paper_id: str = ""
    num_pages: int = 0
    num_chunks: int = 0
    formula_resolved: bool = True
    need_review: bool = False
    error: str = ""
    title: str = ""
    chunks: List[Dict] = field(default_factory=list)
    vlm_used: bool = False
    formulas: List[Dict] = field(default_factory=list)
    parse_mode: str = ""


@dataclass
class TranslateRequest:
    """多语种文献翻译请求（文档第 2 章"多语种文献翻译"）

    Attributes:
        text: 待翻译文本（限 5000 字符）
        source_lang: 源语言代码（默认 en）
        target_lang: 目标语言代码（默认 zh-CN）
        user_id: 用户 ID（可选，便于成本/调用统计）
    """

    text: str = ""
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    user_id: str = ""


@dataclass
class TranslateResponse:
    """多语种文献翻译响应（术语已按专业词库校准）"""

    translated: str = ""
    source_text: str = ""
    aligned_terms: List[str] = field(default_factory=list)
    target_lang: str = "zh-CN"
    mock_mode: bool = False
    latency_ms: int = 0
    error: str = ""


@dataclass
class AskRequest:
    """论文问答请求（文档第 2 章"基于记忆池的长文问答"）

    Attributes:
        paper_id: 论文 ID
        question: 用户提问（限 2000 字，文档 6.5 节）
        user_id: 用户 ID
        use_memory: 是否启用记忆池（默认 True）
        top_k: RAG 检索 top_k
    """

    paper_id: str = ""
    question: str = ""
    user_id: str = ""
    use_memory: bool = True
    top_k: int = 5
    history: List[Dict] = field(default_factory=list)


@dataclass
class AskResponse:
    """论文问答响应

    model/input_tokens/output_tokens/cost_yuan 用于前端展示本次调用成本；
    memory_used 标注是否注入了记忆池/会话历史（文档第 2 章"基于记忆池的长文问答"）；
    memory_entities 为本次实际注入记忆池（ContextManager）的实体条目数，
    mock 与真实 RAG 分支都会真实统计（P0-2 接线后可观测）；
    memory_items 为历史字段名兼容别名（与 memory_entities 同值，保留避免破坏旧调用方）。
    """

    answer: str = ""
    citations: List[Dict] = field(default_factory=list)
    trace_id: str = ""
    from_cache: bool = False
    latency_ms: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    mock_mode: bool = False
    memory_used: bool = False
    memory_entities: int = 0
    memory_items: int = 0  # 兼容别名：等于 memory_entities
    error: str = ""


@dataclass
class ReviewRequest:
    """综述生成请求（文档第 2 章"综述草稿生成"）

    Attributes:
        topic: 综述主题（限 500 字，文档 6.5 节）
        doc_ids: 文献 doc_id 列表
        user_id: 用户 ID
        num_sections: 期望章节数
        verify_citations: 是否做引用验证
        export_docx: 是否导出 Word
    """

    topic: str = ""
    doc_ids: List[str] = field(default_factory=list)
    user_id: str = ""
    num_sections: int = 4
    verify_citations: bool = True
    export_docx: bool = False


@dataclass
class ReviewResponse:
    """综述生成响应

    P0-3/P0-4 接线后可观测字段（均有默认值，保证既有调用方零改动）：
    - generator_mode: 本次实际使用的生成器模式（"mock" / "llm"），来自真实
      generator 实例类型而非入参，便于确认 mock 路径与 LLM 路径的差异；
    - mmr_enabled: 本次请求是否具备 MMR 去重条件（embedder 已传入且生成器
      具备 MMR 路径；LLM/Mock 生成器均已接入）。它表示"能力已启用"，不代表
      本次一定执行了去重——实际是否执行看 dedup_stats.calls / mmr_applied；
    - mmr_applied: 本次综述生成是否真正执行过 MMR 去重（= dedup_stats.calls > 0）；
    - dedup_stats: MMR 统计 {mmr_applied, calls, input_summaries,
      selected_summaries, skipped, reduce_top_k, mmr_lambda}，跳过也会计数；
    - verification_skipped: 是否因无可用 chunks/检索器等原因跳过引用事实验证；
    - verification_skip_reason: 跳过原因（未跳过时为空串）；
    - verification_error: 历史字段名兼容别名（= verification_skip_reason）。
    """

    title: str = ""
    full_text: str = ""
    full_html: str = ""
    outline: List[Dict] = field(default_factory=list)
    paragraphs: List[Dict] = field(default_factory=list)
    citation_stats: Dict[str, int] = field(default_factory=dict)
    references: List[Dict] = field(default_factory=list)
    verified: bool = False
    generator_mode: str = ""
    mmr_enabled: bool = False
    mmr_applied: bool = False
    dedup_stats: Dict = field(default_factory=dict)
    verification_skipped: bool = False
    verification_skip_reason: str = ""
    verification_error: str = ""  # 兼容别名：等于 verification_skip_reason
    docx_path: str = ""
    docx_b64: str = ""
    docx_mode: str = ""
    docx_name: str = ""
    docx_size: int = 0
    trace_id: str = ""
    latency_ms: int = 0
    mock_mode: bool = False
    error: str = ""


@dataclass
class HealthResponse:
    """健康检查响应"""

    status: str = "ok"
    version: str = "0.1.0"
    mock_mode: bool = True
    components: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Agent 问答响应（字段与 app.agent.AgentResult.to_dict() 契约对齐）

    /agent 端点直接返回 result.to_dict() 的 plain dict；
    本 dataclass 供文档 / 前端 schema 对齐使用。
    """

    answer: str = ""
    steps: List[Dict] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    trace_id: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    mock_mode: bool = False
    framework: str = "auto"
    latency_ms: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Pydantic 请求模型（FastAPI 路由用）
#
# ⚠️ 必须定义在模块级、不能在 create_app 闭包内：
# 本文件顶部 `from __future__ import annotations` 使路由函数注解变成字符串，
# FastAPI 按模块命名空间解析（函数 __globals__），闭包内的类无法被解析，
# 会导致 body 参数被误判为 query（422）或 openapi 报 ForwardRef 未定义。
#
# 用 try 包裹：fastapi 未安装时本模块仍可 import（与 schema dataclass 容错约定一致）。
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel as _BaseModel  # type: ignore

    class UploadBody(_BaseModel):
        filename: str = ""
        file_b64: str = ""
        user_id: str = ""
        use_vlm: bool = False

    class AskBody(_BaseModel):
        paper_id: str = ""
        question: str = ""
        user_id: str = ""
        use_memory: bool = True
        top_k: int = 5
        history: List[Dict] = []

    class ReviewBody(_BaseModel):
        topic: str = ""
        doc_ids: List[str] = []
        user_id: str = ""
        num_sections: int = 4
        verify_citations: bool = True
        export_docx: bool = False

    class TranslateBody(_BaseModel):
        text: str = ""
        source_lang: str = "en"
        target_lang: str = "zh-CN"
        user_id: str = ""

    class AgentBody(_BaseModel):
        """Agent 问答请求（POST /agent）"""

        question: str = ""
        doc_id: str = ""
        user_id: str = ""
        max_steps: int = 6
        framework: str = "auto"          # auto | simple | langgraph
        history: List[Dict] = []

    class AgentToolCallBody(_BaseModel):
        """Agent 工具直接调用请求（POST /agent/tools/call，供 MCP 适配器使用）"""

        name: str = ""
        arguments: Dict = {}

except ImportError:  # pragma: no cover - fastapi/pydantic 未安装时的降级占位
    # fastapi 不可用时 create_app 直接返回 None，路由不会用到这些模型，
    # 这里仅保证模块级 import 不报错。
    class UploadBody:  # type: ignore[no-redef]
        pass

    class AskBody:  # type: ignore[no-redef]
        pass

    class ReviewBody:  # type: ignore[no-redef]
        pass

    class TranslateBody:  # type: ignore[no-redef]
        pass

    class AgentBody:  # type: ignore[no-redef]
        pass

    class AgentToolCallBody:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Agent 响应辅助（模块级，便于在 create_app 闭包内复用）
# ---------------------------------------------------------------------------


# AgentResult.to_dict() 的稳定字段（契约冻结）；出错时也返回同构 dict，
# 保证前端无需分支判断响应结构。
AGENT_RESULT_KEYS = (
    "answer", "steps", "tool_calls", "citations", "trace_id", "model",
    "input_tokens", "output_tokens", "cost_yuan", "mock_mode", "framework",
    "latency_ms", "error",
)


def _empty_agent_result(
    error: str,
    framework: str = "auto",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    """构造与 AgentResult.to_dict() 同构的错误响应（不抛 500）

    Args:
        error: 错误信息
        framework: 请求的编排框架（auto/simple/langgraph）
        mock_mode: 当前服务是否 mock 模式
    Returns:
        与 AgentResult.to_dict() 字段一致的 dict
    """
    result: Dict[str, Any] = {key: None for key in AGENT_RESULT_KEYS}
    result.update({
        "answer": "",
        "steps": [],
        "tool_calls": [],
        "citations": [],
        "trace_id": "",
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_yuan": 0.0,
        "mock_mode": bool(mock_mode),
        "framework": framework or "auto",
        "latency_ms": 0,
        "error": error,
    })
    return result


def _jsonable(value: Any) -> Any:
    """把工具返回值转成 JSON 可序列化对象（失败时降级为字符串）"""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        from fastapi.encoders import jsonable_encoder

        return jsonable_encoder(value)
    except Exception:  # noqa: BLE001
        return str(value)


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app(*, mock_mode: Optional[bool] = None):
    """创建 FastAPI 应用实例

    Args:
        mock_mode: 是否强制 mock 模式；None 时根据 DASHSCOPE_API_KEY 环境变量自动判断
    Returns:
        FastAPI 应用（fastapi 不可用时返回 None）
    """
    # --factory 方式启动时 uvicorn 调用 create_app() 不传参，
    # 这里读取环境变量决定 mock 模式，避免 Docker 部署永远处于 mock 模式
    if mock_mode is None:
        mock_mode = not bool(os.getenv("DASHSCOPE_API_KEY", ""))

    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
    except ImportError:
        logger.warning("fastapi 未安装，create_app 返回 None（路由不可用）")
        return None

    app = FastAPI(
        title="多语种文献翻译与论文研读助手",
        description=(
            "API-first 架构：PDF 解析 + RAG 问答 + 综述生成 + 引用溯源。"
            "文档第 3 章架构图。"
        ),
        version="0.1.0",
    )

    # 注入依赖组件（mock 模式下全部走 Mock）
    state = _AppState(mock_mode=mock_mode)
    app.state.app_state = state

    # ------------------------------------------------------------------
    # 路由：健康检查
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return HealthResponse(
            status="ok",
            version="0.1.0",
            mock_mode=state.mock_mode,
            components={
                "parser": "ok" if state.mock_mode else "ok",
                "rag": "ok",
                "review": "ok",
                "llm": "mock" if state.mock_mode else "dashscope",
                "translation": "mock" if state.mock_mode else "dashscope",
                "agent": "mock" if state.mock_mode else "dashscope",
            },
        )

    # ------------------------------------------------------------------
    # 路由：PDF 上传（文档第 2 章 + 4.1 节）
    # ------------------------------------------------------------------

    @app.post("/upload", response_model=None)
    def upload(body: UploadBody):
        import base64
        import hashlib

        # 文档 6.5 节：单 PDF 文件 < 50MB
        try:
            raw = base64.b64decode(body.file_b64) if body.file_b64 else b""
        except Exception as e:  # noqa: BLE001
            return UploadResponse(error=f"file_b64 解码失败: {e}")
        if len(raw) > MAX_PDF_SIZE_MB * 1024 * 1024:
            return UploadResponse(
                error=f"PDF 文件超过 {MAX_PDF_SIZE_MB}MB 限制（文档 6.5 节）",
            )

        # 计算 doc_id（基于文件 MD5，文档 4.1 节解析结果缓存 key）
        doc_id = "doc_" + hashlib.md5(raw).hexdigest()[:12] if raw else "doc_empty"

        # mock 模式且未启用 VLM：用固定 mock 论文文本走真实分块 + 进程内索引，
        # 保证前端能展示页数/分块/文档索引（不再返回 0 页 0 chunk）。
        if state.mock_mode and not body.use_vlm:
            return state.mock_upload(doc_id, body.filename)

        # 真实模式 / 或显式 use_vlm=True：调 app.parser 流水线
        try:
            if body.use_vlm:
                # 可选 VLM 解析链路（pdf_to_image → QwenVlmClient → FusionEngine）
                return state.run_parser_pipeline_vlm(body.filename, raw, body.user_id)
            result = state.run_parser_pipeline(body.filename, raw, body.user_id)
            return result
        except Exception as e:  # noqa: BLE001
            logger.error("PDF 解析失败: %s", e)
            return UploadResponse(error=str(e))

    # ------------------------------------------------------------------
    # 路由：论文问答（文档第 2 章 + 4.2 节）
    # ------------------------------------------------------------------

    @app.post("/ask", response_model=None)
    def ask(body: AskBody):
        # 文档 6.5 节：用户提问限 2000 字
        if len(body.question) > MAX_QUESTION_LEN:
            return AskResponse(
                error=f"提问长度 {len(body.question)} 超过限制 {MAX_QUESTION_LEN}（文档 6.5 节）",
            )

        if not body.question.strip():
            return AskResponse(error="question 不能为空")

        # mock 模式：返回模板化回答（P0-2：同样从该 doc 的记忆池注入实体）
        if state.mock_mode:
            return state.mock_ask(
                body.paper_id, body.question, body.user_id,
                use_memory=body.use_memory,
                history=body.history,
            )

        # 真实模式：调 RAG + LLM（history 用于记忆池注入）
        try:
            result = state.run_ask(
                paper_id=body.paper_id,
                question=body.question,
                user_id=body.user_id,
                use_memory=body.use_memory,
                top_k=body.top_k,
                history=body.history,
            )
            return result
        except Exception as e:  # noqa: BLE001
            logger.error("问答失败: %s", e)
            return AskResponse(error=str(e))

    # ------------------------------------------------------------------
    # 路由：综述生成（文档第 2 章 + 4.3 节）
    # ------------------------------------------------------------------

    @app.post("/review", response_model=None)
    def review(body: ReviewBody):
        # 文档 6.5 节：综述主题限 500 字
        if len(body.topic) > MAX_REVIEW_TOPIC_LEN:
            return ReviewResponse(
                error=f"综述主题长度 {len(body.topic)} 超过限制 {MAX_REVIEW_TOPIC_LEN}（文档 6.5 节）",
            )

        if not body.topic.strip():
            return ReviewResponse(error="topic 不能为空")
        if not body.doc_ids:
            # 无 doc 时显式标记引用验证跳过（而不是静默返回 False），便于前端解释
            skip_reason = "未提供 doc_ids（无可用文献），引用事实验证被跳过"
            return ReviewResponse(
                error="doc_ids 不能为空",
                verification_skipped=True,
                verification_skip_reason=skip_reason,
                verification_error=skip_reason,
            )

        # mock 模式 / 真实模式 都走 ReviewPipeline（pipeline 内部自动降级）
        try:
            result = state.run_review(
                topic=body.topic,
                doc_ids=body.doc_ids,
                user_id=body.user_id,
                num_sections=body.num_sections,
                verify_citations=body.verify_citations,
                export_docx=body.export_docx,
            )
            return result
        except Exception as e:  # noqa: BLE001
            logger.error("综述生成失败: %s", e)
            return ReviewResponse(error=str(e))

    # ------------------------------------------------------------------
    # 路由：多语种文献翻译（文档第 2 章"多语种文献翻译"）
    # ------------------------------------------------------------------

    @app.post("/translate", response_model=None)
    def translate(body: TranslateBody):
        # 空文本 / 超长文本直接拒绝（术语预处理的输入边界）
        if not body.text.strip():
            return TranslateResponse(error="text 不能为空")
        if len(body.text) > MAX_TRANSLATE_LEN:
            return TranslateResponse(
                error=f"文本长度 {len(body.text)} 超过限制 {MAX_TRANSLATE_LEN}",
            )

        try:
            unit = state._ensure_translator().translate_text(
                body.text, body.source_lang, body.target_lang,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("翻译失败: %s", e)
            return TranslateResponse(error=str(e))

        return TranslateResponse(
            translated=unit.target_text,
            source_text=unit.source_text,
            aligned_terms=list(unit.aligned_terms),
            target_lang=unit.target_lang,
            mock_mode=state.mock_mode or unit.from_mock,
            latency_ms=unit.latency_ms,
        )

    # ------------------------------------------------------------------
    # 路由：文档索引列表（前端需展示已上传/已解析文献）
    # 注意：路径用 /documents 而非 /docs —— FastAPI 默认用 /docs 提供
    # Swagger UI，业务端点若占用会遮蔽框架自带的接口文档页。
    # ------------------------------------------------------------------

    @app.get("/documents")
    def list_docs():
        return {"docs": state.list_docs(), "error": ""}

    # ------------------------------------------------------------------
    # 路由：Agent 助手（工具调用 + 多步推理）
    #
    # 契约：app.agent 提供 create_agent_runner / create_default_tools /
    # default_tool_specs / AgentResult.to_dict()。这里全部延迟 import，
    # 保证 app/agent 尚未就绪时本模块仍可 import（并行开发容错）。
    # ------------------------------------------------------------------

    @app.post("/agent", response_model=None)
    def agent(body: AgentBody):
        question = (body.question or "").strip()
        # 文档 6.5 节安全：提问限 2000 字
        if not question:
            return _empty_agent_result(
                "question 不能为空", body.framework, state.mock_mode,
            )
        if len(question) > MAX_AGENT_QUESTION_LEN:
            return _empty_agent_result(
                f"提问长度 {len(question)} 超过限制 {MAX_AGENT_QUESTION_LEN}（文档 6.5 节）",
                body.framework, state.mock_mode,
            )

        # max_steps clamp 到 [1, MAX_AGENT_STEPS]，防死循环
        try:
            max_steps = int(body.max_steps)
        except (TypeError, ValueError):
            max_steps = 6
        max_steps = max(1, min(max_steps, MAX_AGENT_STEPS))
        framework = body.framework if body.framework in (
            "auto", "simple", "langgraph",
        ) else "auto"

        try:
            return state.run_agent(
                question=question,
                doc_id=body.doc_id,
                user_id=body.user_id,
                max_steps=max_steps,
                framework=framework,
                history=list(body.history or []),
            )
        except Exception as e:  # noqa: BLE001 - 任何异常都转成 error 字段，不返回 500
            logger.error("Agent 运行失败: %s", e)
            return _empty_agent_result(str(e), framework, state.mock_mode)

    @app.get("/agent/tools")
    def agent_tools():
        """列出 Agent 可用工具（OpenAI tools 格式，供 MCP server / 前端展示）"""
        try:
            from ..agent import default_tool_specs

            return {"tools": default_tool_specs(), "error": ""}
        except Exception as e:  # noqa: BLE001
            logger.error("Agent 工具列表获取失败: %s", e)
            return {"tools": [], "error": str(e)}

    @app.post("/agent/tools/call", response_model=None)
    def agent_tool_call(body: AgentToolCallBody):
        """直接调用单个 Agent 工具（供 MCP 适配器 / 调试使用）"""
        name = body.name or ""
        arguments = dict(body.arguments or {})
        try:
            tools = state.build_agent_tools()
            execution = tools.call(name, arguments)
            return {
                "name": name,
                "arguments": arguments,
                "result": _jsonable(execution.result),
                "error": str(execution.error or ""),
                "latency_ms": int(execution.latency_ms or 0),
            }
        except Exception as e:  # noqa: BLE001 - 未知工具/参数错误不出 500
            logger.error("Agent 工具调用失败 name=%s: %s", name, e)
            return {
                "name": name,
                "arguments": arguments,
                "result": None,
                "error": str(e),
                "latency_ms": 0,
            }

    # ------------------------------------------------------------------
    # 路由：成本看板（文档 6.1 / 6.2 节）
    # ------------------------------------------------------------------

    @app.get("/cost/summary")
    def cost_summary():
        try:
            from ..observability.cost_tracker import CostDashboard

            tracker = state._ensure_cost_tracker()
            dashboard = CostDashboard(tracker)
            return {
                "total_cost": float(tracker.total_cost()),
                "daily": dashboard.daily_summary(),
                "by_model": dashboard.cost_by_model(),
                "by_user": dashboard.cost_by_user(),
                "by_paper": dashboard.cost_by_paper(),
                "cache": dashboard.cache_hit_rate(),
                "error": "",
            }
        except Exception as e:  # noqa: BLE001
            logger.error("成本看板聚合失败: %s", e)
            return {
                "total_cost": 0.0,
                "daily": {},
                "by_model": {},
                "by_user": {},
                "by_paper": {},
                "cache": {"rate": 0.0, "hits": 0, "total": 0},
                "error": str(e),
            }

    return app


# ---------------------------------------------------------------------------
# 应用状态（依赖注入容器）
# ---------------------------------------------------------------------------


# mock 模式上传时使用的固定论文文本：5 段结构（引言/方法/实验/相关工作/结论），
# 含 ImageNet 85.6% / Attention 公式 / BLEU=28.4 / F1=0.92 / ResNet-152 / BERT_BASE
# 等实体，供分块、检索、综述、导出全链路在无 Key 时演示。
MOCK_PAPER_TEXT = """\
# Mock Paper: Attention-based Residual Networks for Vision and Language

## 1. Introduction

This mock paper demonstrates the full pipeline without any DashScope API call.
We study how a ResNet-152 backbone and a Transformer encoder are combined for
image classification and machine translation. On ImageNet-1K our model reaches
85.6% top-5 accuracy, and the attention module follows
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V exactly as in the original paper.
All statistics below are fixed template values for offline demonstration.

## 2. Method

We first extract hierarchical chunks from the uploaded PDF and index them in
memory. Each chunk keeps its chunk_id, token_count and section title so that
retrieval can cite exact evidence. The encoder uses a BERT_BASE style
pre-training objective, and the decoder adopts multi-head attention together
with residual connections and layer normalization. Training uses Adam with a
learning rate of 3e-4 and batch size 64 for 200 epochs.

## 3. Experiments

Experimental results show BLEU=28.4 on WMT 2014 English-to-German translation,
F1=0.92 on the in-house relation extraction set, and mAP=0.81 on COCO-2017
object detection. Ablation studies confirm that removing positional encoding
drops the accuracy by 1.7 points. These numbers are fixed mock values used to
exercise the cost dashboard and the Word export path.

## 4. Related Work

Prior work on self-attention and layer normalization established the foundation
of modern language models. BERT and GPT families popularized pre-training
followed by fine-tuning, while ResNet-152 remains a strong vision baseline.
Retrieval-augmented generation combines a retriever with a generator, and
citation verification keeps generated statements grounded in source chunks.

## 5. Conclusion

We presented a mock multi-lingual paper reading assistant covering PDF parsing,
RAG question answering, review generation and Word export. Future work will plug
a real DashScope key, expand the terminology glossary, and evaluate translation
quality on a larger benchmark. The current version is intentionally template
based so that the whole chain works offline without billing.
"""


# mock VLM 演示论文文本：fake PDF 无法被 fitz 打开时走"mock VLM 演示"路径，
# 5 段正文含 ImageNet 85.6% / BLEU=28.4 / F1=0.92 等指标，供前端演示 VLM 解析结果。
MOCK_VLM_PAPER_TEXT = """\
# Mock VLM Paper: Attention-based Residual Networks for Vision and Language

## 1. Introduction

This mock paper is returned by the optional VLM parsing chain without any real
DashScope API call. The backbone is a ResNet-152 encoder with a Transformer
decoder, and the attention module follows
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V exactly as in the original paper.
On ImageNet-1K the model reaches 85.6% top-5 accuracy.

## 2. Method

We render every page to an image and let Qwen-VL-Max recognize the layout, then
fuse the visual result with the pdfplumber text layer. Formulas are bound to
paragraphs by four rule levels with explicit confidence values, so low-confidence
formulas are marked need_review for human verification.

## 3. Experiments

The model achieves BLEU=28.4 on WMT 2014 English-to-German translation and F1=0.92
on the in-house relation extraction set. The cross-entropy loss is
L = -sum_i y_i log(y_hat_i) and the F1 score is F1 = 2PR/(P+R).

## 4. Related Work

Prior work on self-attention and layer normalization established the foundation of
modern language models. BERT and GPT families popularized pre-training followed by
fine-tuning, while ResNet-152 remains a strong vision baseline. Retrieval-augmented
generation combines a retriever with a generator for grounded answers.

## 5. Conclusion

We presented the optional VLM parsing chain covering PDF page rendering, VLM page
recognition, multi-source fusion and formula binding. Future work will replace the
mock recognizer with a real DashScope key and evaluate formula accuracy on the full
benchmark. All numbers above are fixed mock values for offline demonstration.
"""


# mock VLM 演示公式（含 Attention 公式等 3 条；置信度低于 0.7 的标 need_review）
MOCK_VLM_FORMULAS = (
    {
        "latex": r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
        "confidence": 0.95,
        "page_no": 0,
        "paragraph_idx": 0,
        "formula_id": "(1)",
        "binding_rule": "rule1_paragraph_number",
        "location_desc": "第1段后",
    },
    {
        "latex": r"\mathcal{L}=-\sum_{i}y_i\log(\hat{y}_i)",
        "confidence": 0.88,
        "page_no": 0,
        "paragraph_idx": 2,
        "formula_id": "(2)",
        "binding_rule": "rule2_position_mapping",
        "location_desc": "第二栏中部",
    },
    {
        "latex": r"\mathrm{F1}=2PR/(P+R)",
        "confidence": 0.55,
        "page_no": 1,
        "paragraph_idx": 4,
        "formula_id": "(3)",
        "binding_rule": "rule4_unbound",
        "location_desc": "页面底部",
    },
)


# ---------------------------------------------------------------------------
# 记忆池 prompt 解析辅助（P0-2：记忆池接入 /upload + /ask）
# ---------------------------------------------------------------------------


def _count_memory_items(memory_prompt: str, question: str) -> int:
    """统计 ContextManager.build_prompt 实际注入的实体条目数

    build_prompt 命中实体时返回：
        "已知信息：\\n- [type] value (来源: chunk_id)\\n...\\n\\n问题：..."
    未命中时原样返回 question，因此需要排除"原样返回问题"的情况。

    Args:
        memory_prompt: memory.build_prompt 的返回值
        question: 原始问题（用于识别未命中时的原样返回）
    Returns:
        实际注入的实体条目数（未注入为 0）
    """
    if not memory_prompt or memory_prompt == question:
        return 0
    return sum(
        1 for line in memory_prompt.splitlines()
        if line.lstrip().startswith("- [")
    )


def _extract_memory_entity_values(memory_prompt: str, limit: int = 5) -> List[str]:
    """从记忆池 prompt 中提取实体值（用于 mock 回答里的实体预览）

    行格式："- [dataset] ImageNet-1K (来源: doc_xxx#0.0)"，
    这里取 "] " 与 " (来源:" 之间的实体值，最多 limit 个。
    """
    values: List[str] = []
    for line in memory_prompt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ["):
            continue
        rest = stripped[stripped.find("]") + 1:].strip()
        value = rest.split(" (来源:")[0].strip()
        if value:
            values.append(value)
        if len(values) >= limit:
            break
    return values


class _AppState:
    """FastAPI 应用状态容器

    持有 parser/rag/review 等组件实例，根据 mock_mode 决定是否真实调用 API。
    所有 run_* 方法在 mock 模式下返回模板化响应，真实模式下走完整链路。

    真实编排（run_parser_pipeline / run_ask）：
    - 上传 PDF → PyMuPDF 抽文本 → 层级分块 → 进程内文档索引
    - 问答 → 按 paper_id 从索引取 chunks → 向量检索 → LLM 生成 + 引用
    注：单进程/单 worker 下内存索引可用；多 worker 部署需外置存储（文档已声明）。
    """

    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode or not bool(os.getenv("DASHSCOPE_API_KEY", ""))
        # 进程内文档索引：{doc_id: {"title": str, "chunks": [Chunk, ...]}}
        self._docs: Dict[str, Dict] = {}
        # 进程内记忆池索引（P0-2 接线）：{doc_id: ContextManager}，上传时按 chunk
        # 抽取实体写入；与 _docs 共用 _docs_lock 保护，保证 /ask 读到的记忆池与
        # chunks 一致。对外通过 get_context_manager(doc_id) 读取。
        self._context_managers: Dict[str, "ContextManager"] = {}
        # 历史字段名兼容别名（与 _context_managers 指向同一 dict，避免覆盖旧引用）
        self._memories = self._context_managers
        self._docs_lock = threading.Lock()
        # 可注入组件（测试可替换；None 时按需懒加载）
        self._embedder = None
        self._llm_client = None
        self._chunker = None
        self._translator = None
        self._cost_tracker = None

    # ------------------------------------------------------------------
    # 懒加载组件
    # ------------------------------------------------------------------

    def _ensure_chunker(self):
        """层级分块器（默认 256 token + overlap 64，文档第 5 章）"""
        if self._chunker is None:
            from ..rag.chunker import HierarchicalChunker
            self._chunker = HierarchicalChunker(chunk_size=256, overlap=64)
        return self._chunker

    def _ensure_embedder(self):
        """向量化器：真实模式用 DashScope Embedding（有 key 时）"""
        if self._embedder is None:
            from ..rag.embedder import DashScopeEmbedder
            self._embedder = DashScopeEmbedder()
        return self._embedder

    def _ensure_llm(self):
        """LLM 客户端（真实模式，失败自动降级 Mock）"""
        if self._llm_client is None:
            from ..llm import create_llm_client
            self._llm_client = create_llm_client("auto")
        return self._llm_client

    def _ensure_translator(self):
        """多语种翻译器：mock 模式用 MockTranslator，真实模式 auto（无 Key 自动降级）"""
        if self._translator is None:
            from ..translation import create_translator
            self._translator = create_translator("mock" if self.mock_mode else "auto")
        return self._translator

    def _ensure_cost_tracker(self):
        """成本追踪器（/cost/summary 看板数据源，默认内存版）"""
        if self._cost_tracker is None:
            from ..observability import create_cost_tracker
            self._cost_tracker = create_cost_tracker("auto")
        return self._cost_tracker

    def _index_doc(self, doc_id: str, title: str, chunks) -> None:
        """把分块结果写入进程内文档索引，并按 doc_id 重建记忆池（P0-2）

        实现要点：
        - 实体抽取在锁外完成（CPU 开销，避免阻塞其他请求），随后在
          _docs_lock 内**一次性**写入 _docs 与 _context_managers，保证两者一致；
        - 同一 doc_id 重复上传时覆盖重建（旧实体不残留）；
        - 单 chunk 抽取失败只记 warning，不影响上传主流程。
        """
        chunks_list = list(chunks)
        memory = self._build_doc_memory(doc_id, chunks_list)
        with self._docs_lock:
            self._docs[doc_id] = {"title": title, "chunks": chunks_list}
            self._context_managers[doc_id] = memory

    def _build_doc_memory(self, doc_id: str, chunks) -> Any:
        """按 chunk 抽取实体构建文档记忆池

        Args:
            doc_id: 文档 ID
            chunks: Chunk 列表（需有 text / chunk_id 属性）
        Returns:
            ContextManager（即使全部抽取失败也返回空池实例，/ask 可安全降级）
        """
        from ..rag.context_manager import ContextManager

        memory = ContextManager(max_size=200)
        for chunk in chunks:
            text = getattr(chunk, "text", "") or ""
            chunk_id = getattr(chunk, "chunk_id", "") or ""
            try:
                memory.update(text, chunk_id)
            except Exception as e:  # noqa: BLE001 - 抽取失败不得中断上传
                logger.warning(
                    "记忆池实体抽取失败（已跳过该 chunk）：doc_id=%s, chunk_id=%s, err=%s",
                    doc_id, chunk_id, e,
                )
        logger.info(
            "记忆池构建完成：doc_id=%s, chunks=%d, entities=%d",
            doc_id, len(chunks), len(memory),
        )
        return memory

    def get_context_manager(self, doc_id: str):
        """按 doc_id 取该文献的 ContextManager 记忆池（P0-2 接线，线程安全）

        Args:
            doc_id: 上传时确定的文档 ID
        Returns:
            ContextManager 实例；未上传/尚未解析/无索引时返回 None（调用方安全降级）
        """
        with self._docs_lock:
            return self._context_managers.get(doc_id)

    def get_doc_memory(self, doc_id: str):
        """历史方法名兼容别名：等价于 get_context_manager(doc_id)"""
        return self.get_context_manager(doc_id)

    def get_doc_chunks(self, doc_id: str):
        """按 doc_id 取 chunks（供真实 run_ask / run_review 使用）"""
        with self._docs_lock:
            doc = self._docs.get(doc_id)
        return list(doc["chunks"]) if doc else []

    @staticmethod
    def chunk_previews(chunks) -> List[Dict]:
        """把 Chunk 列表转成响应里的预览 dict（text 截断 400 字，避免响应体过大）"""
        return [
            {
                "chunk_id": c.chunk_id,
                "text": (c.text or "")[:400],
                "token_count": c.token_count,
                "doc_id": c.doc_id,
                "section": c.section,
            }
            for c in chunks
        ]

    def list_docs(self) -> List[Dict]:
        """列出进程内索引中的文档（GET /documents）"""
        with self._docs_lock:
            return [
                {
                    "doc_id": doc_id,
                    "title": doc.get("title", doc_id),
                    "num_chunks": len(doc.get("chunks", [])),
                }
                for doc_id, doc in self._docs.items()
            ]

    # ------------------------------------------------------------------
    # Agent 工具注册（回调闭包捕获当前 state，契约见 app.agent）
    # ------------------------------------------------------------------

    def build_agent_tools(self):
        """构造 Agent 工具注册表（POST /agent 与 /agent/tools/call 共用）

        回调均为位置参数（app.agent 契约 v1）：
        - list_documents() -> List[Dict]
        - search_chunks(query, doc_id, top_k) -> List[Dict]
        - translate_text(text, target_lang) -> Dict
        - generate_review(topic, doc_ids, num_sections) -> Dict

        Returns:
            app.agent.ToolRegistry
        Raises:
            ImportError: app/agent 尚未就绪（路由层转成 error 字段，不返回 500）
        """
        from ..agent import create_default_tools

        return create_default_tools(
            self._agent_list_documents,
            self._agent_search_chunks,
            self._agent_translate_text,
            self._agent_generate_review,
        )

    def _agent_list_documents(self) -> List[Dict]:
        """Agent 工具 list_documents：返回已索引文档列表"""
        return self.list_docs()

    def _agent_search_chunks(
        self,
        query: str,
        doc_id: str = "",
        top_k: int = 5,
    ) -> List[Dict]:
        """Agent 工具 search_chunks：在进程内索引上做混合检索（向量 + BM25 + RRF）

        doc_id 为空时检索全部已索引文档；doc_id 不存在时返回空列表
        （由 Agent 在答案中说明"文档不存在/未解析"）。
        """
        try:
            top_k = int(top_k or 5)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

        with self._docs_lock:
            if doc_id:
                doc = self._docs.get(doc_id)
                chunks = list(doc.get("chunks", [])) if doc else []
            else:
                chunks = [
                    chunk
                    for indexed in self._docs.values()
                    for chunk in indexed.get("chunks", [])
                ]
        if not chunks:
            return []

        try:
            from ..rag.retriever import HybridRetriever

            if self.mock_mode:
                from ..rag.embedder import FakeEmbedder
                embedder = FakeEmbedder(dim=64)
            else:
                embedder = self._ensure_embedder()
            retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
            retriever.add_chunks(chunks)
            results = retriever.search(query or "", top_k=top_k)
            return [
                {
                    "chunk_id": rr.chunk.chunk_id,
                    "doc_id": rr.chunk.doc_id,
                    "text": (rr.chunk.text or "")[:600],
                    "section": rr.chunk.section,
                    "score": round(float(rr.score), 6),
                }
                for rr in results
            ]
        except Exception as e:  # noqa: BLE001 - 工具回调异常由 Agent 错误链记录
            logger.error("Agent search_chunks 失败: %s", e)
            return []

    def _agent_translate_text(
        self,
        text: str,
        target_lang: str = "zh-CN",
    ) -> Dict:
        """Agent 工具 translate_text：复用多语种翻译器（术语校准）"""
        unit = self._ensure_translator().translate_text(
            text or "", "en", target_lang or "zh-CN",
        )
        return {
            "translated": unit.target_text,
            "aligned_terms": list(unit.aligned_terms),
            "target_lang": unit.target_lang,
            "mock_mode": self.mock_mode or unit.from_mock,
        }

    def _agent_generate_review(
        self,
        topic: str,
        doc_ids: Optional[List[str]] = None,
        num_sections: int = 3,
    ) -> Dict:
        """Agent 工具 generate_review：复用 /review 的 run_review 编排

        run_review 已按 self.mock_mode 显式选择 mode="mock"（MockOutlinePlanner +
        MockReviewGenerator + FakeEmbedder + 规则化 NLI），因此这里不再通过
        清空环境变量强制 mock——显式 mock_mode 在 Windows 下同样可靠。
        """
        try:
            num_sections = int(num_sections or 3)
        except (TypeError, ValueError):
            num_sections = 3
        num_sections = max(1, min(num_sections, 6))

        response = self.run_review(
            topic=topic or "",
            doc_ids=list(doc_ids or []),
            user_id="",
            num_sections=num_sections,
            verify_citations=False,
            export_docx=False,
        )
        return {
            "title": response.title,
            "full_text": response.full_text,
            "references": response.references,
            "citation_stats": response.citation_stats,
            "mock_mode": response.mock_mode or self.mock_mode,
            "error": response.error,
        }

    def run_agent(
        self,
        question: str,
        doc_id: str = "",
        user_id: str = "",
        max_steps: int = 6,
        framework: str = "auto",
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """运行 Agent 并记录成本（mock 模式下行内 ScriptedMockLLM，绝不真实调 API）

        Returns:
            AgentResult.to_dict() 的 JSON-safe dict（字段见 AGENT_RESULT_KEYS）
        """
        from ..agent import create_agent_runner

        tools = self.build_agent_tools()
        # mock 模式必须 0 真实 API 调用：显式注入脚本化 mock LLM。
        # 不能依赖 create_agent_runner(llm=None) 的 auto 选型——开发机 .env 里
        # 若存在 DASHSCOPE_API_KEY，auto 会选真实 DashScope 客户端。
        llm = None
        if self.mock_mode:
            from ..agent import ScriptedMockLLM

            llm = ScriptedMockLLM()
        runner = create_agent_runner(
            llm=llm,
            tools=tools,
            framework=framework or "auto",
            max_steps=max_steps,
        )
        result = runner.run(
            question,
            doc_id=doc_id or "",
            user_id=user_id or "",
            history=list(history or []),
        )
        payload = result.to_dict()
        self._record_agent_cost(payload, user_id=user_id, doc_id=doc_id)
        return payload

    def _record_agent_cost(
        self,
        payload: Dict,
        user_id: str = "",
        doc_id: str = "",
    ) -> None:
        """把 Agent 运行成本写入成本追踪器（/cost/summary 的 agent.run 维度）"""
        try:
            from datetime import datetime

            from ..observability import CostRecord

            self._ensure_cost_tracker().record(CostRecord(
                ts=datetime.now(),
                model=str(payload.get("model") or "mock-qwen-plus"),
                step="agent.run",
                input_tokens=int(payload.get("input_tokens") or 0),
                output_tokens=int(payload.get("output_tokens") or 0),
                cost_yuan=float(payload.get("cost_yuan") or 0.0),
                trace_id=str(payload.get("trace_id") or ""),
                user_id=user_id or "",
                paper_id=doc_id or "",
                from_cache=False,
            ))
        except Exception as e:  # noqa: BLE001 - 成本记录失败不影响主流程
            logger.warning("Agent 成本记录失败: %s", e)

    # ------------------------------------------------------------------
    # mock 模式辅助
    # ------------------------------------------------------------------

    def mock_upload(self, doc_id: str, filename: str) -> UploadResponse:
        """mock 模式上传：固定 mock 论文文本 → 层级分块 → 进程内索引

        保证无 API Key 时前端也能看到 num_pages/num_chunks/chunks 预览，
        并让后续 /ask、/review 可按 doc_id 取到 chunk。
        """
        title = filename or "mock_paper.pdf"
        chunks = self._ensure_chunker().chunk(MOCK_PAPER_TEXT, doc_id=doc_id)
        self._index_doc(doc_id, title, chunks)
        return UploadResponse(
            doc_id=doc_id,
            paper_id=doc_id,
            num_pages=1,
            num_chunks=len(chunks),
            formula_resolved=True,
            need_review=False,
            title=title,
            chunks=self.chunk_previews(chunks),
            parse_mode="text",
        )

    def mock_vlm_upload(self, doc_id: str, filename: str) -> UploadResponse:
        """mock VLM 演示上传（fake PDF 无法被 fitz 打开时兜底）

        无 API Key 时也要能演示 VLM 解析链路的前端效果：
        固定 5 段正文（含 ImageNet 85.6% / BLEU=28.4 / F1=0.92）+ 3 条模拟公式
        → 层级分块 + 进程内索引 → 返回 formulas / vlm_used=True / parse_mode="vlm"。
        全程不触达 DashScope API。
        """
        title = filename or "mock_paper_vlm.pdf"
        chunks = self._ensure_chunker().chunk(MOCK_VLM_PAPER_TEXT, doc_id=doc_id)
        self._index_doc(doc_id, title, chunks)

        formulas: List[Dict] = []
        for raw_formula in MOCK_VLM_FORMULAS:
            formula = dict(raw_formula)
            formula["need_review"] = formula["confidence"] < VLM_FORMULA_REVIEW_THRESHOLD
            formulas.append(formula)

        high_conf = [f for f in formulas if f["confidence"] >= VLM_FORMULA_REVIEW_THRESHOLD]
        return UploadResponse(
            doc_id=doc_id,
            paper_id=doc_id,
            num_pages=2,
            num_chunks=len(chunks),
            formula_resolved=bool(high_conf),
            need_review=any(f["need_review"] for f in formulas),
            title=title,
            chunks=self.chunk_previews(chunks),
            vlm_used=True,
            formulas=formulas,
            parse_mode="vlm",
        )

    def mock_ask(
        self,
        paper_id: str,
        question: str,
        user_id: str,
        use_memory: bool = True,
        history: Optional[List[Dict]] = None,
    ) -> AskResponse:
        """mock 模式问答响应（P0-2：同样从记忆池注入实体，可离线验证）

        - 该 doc 有记忆池且命中相关实体：memory_used=True、memory_entities=N，
          模板回答**前缀**附「（记忆池已注入 N 条实体：实体1、实体2…）」；
        - 无该 doc / 无相关实体：安全降级（memory_used=False、memory_entities=0），
          仍返回模板回答，不报错。
        - memory_items 为历史字段名兼容别名（与 memory_entities 同值）。
        注意：mock 路径不真实消费 history（只有真实 run_ask 会把会话历史拼进
        prompt），因此 memory_used 只反映记忆池是否真的注入了实体。
        """
        memory_entities = 0
        memory_prompt = ""
        if use_memory:
            memory = self.get_context_manager(paper_id)
            if memory is not None:
                try:
                    memory_prompt = memory.build_prompt(question, top_k=10)
                    memory_entities = _count_memory_items(memory_prompt, question)
                except Exception as e:  # noqa: BLE001 - 记忆池异常不得影响问答
                    logger.warning("mock_ask 记忆池注入失败（降级为纯模板回答）: %s", e)
                    memory_entities = 0

        preview = question.strip().replace("\n", " ")[:60]
        answer = (
            f"[Mock 回答] 关于「{preview}...」的模板化回答。"
            f"实际部署需配置 DASHSCOPE_API_KEY 后由 Qwen-Plus 生成。"
        )
        if memory_entities > 0:
            # P0-2 契约：实体摘要拼进 mock 答案前缀，直观体现记忆池已生效
            entity_values = _extract_memory_entity_values(memory_prompt)
            answer = (
                f"（记忆池已注入 {memory_entities} 条实体："
                f"{'、'.join(entity_values) if entity_values else ''}…）"
                + answer
            )
        return AskResponse(
            answer=answer,
            citations=[],
            trace_id="mock-trace",
            from_cache=False,
            latency_ms=1,
            model="mock-qwen-plus",
            input_tokens=max(1, len(question) // 4),
            output_tokens=max(1, len(answer) // 4),
            cost_yuan=0.0,
            mock_mode=True,
            memory_used=memory_entities > 0,
            memory_entities=memory_entities,
            memory_items=memory_entities,  # 兼容别名
        )

    # ------------------------------------------------------------------
    # 真实模式调用（延迟导入，避免无 key 时报错）
    # ------------------------------------------------------------------

    def run_parser_pipeline(
        self,
        filename: str,
        raw: bytes,
        user_id: str,
    ) -> UploadResponse:
        """真实 PDF 解析流水线（mock 模式不会进入此方法）

        文档 4.1 节流水线在本进程内编排：
        PDF bytes → PyMuPDF 抽文本（fitz） → 层级分块 → 进程内文档索引
        （公式/表格的 VLM 识别需要 DASHSCOPE_API_KEY 且按页转图，当前编排
         先以文本层完成入库；formula_resolved=False 标注公式未走 VLM 识别。）
        """
        import hashlib
        import io

        doc_id = "doc_" + hashlib.md5(raw).hexdigest()[:12] if raw else "doc_empty"
        if not raw:
            return UploadResponse(doc_id=doc_id, paper_id=doc_id, error="file_b64 为空")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF 未安装，无法真实解析 PDF")
            return UploadResponse(
                doc_id=doc_id, paper_id=doc_id,
                error="PyMuPDF 未安装，无法解析 PDF",
            )

        try:
            with fitz.open(stream=io.BytesIO(raw), filetype="pdf") as pdf:
                # 文档 6.5 节：单 PDF 页数限制
                if len(pdf) > MAX_PDF_PAGES:
                    return UploadResponse(
                        doc_id=doc_id, paper_id=doc_id,
                        error=f"PDF 页数超过 {MAX_PDF_PAGES} 上限（文档 6.5 节）",
                    )
                num_pages = len(pdf)
                # 逐页抽文本（第 1 页起）
                page_texts = []
                for page_no in range(num_pages):
                    page = pdf.load_page(page_no)
                    page_texts.append(page.get_text("text") or "")
                full_text = "\n\n".join(page_texts)
        except Exception as e:  # noqa: BLE001
            logger.error("PDF 解析失败（非 PDF 或损坏）: %s", e)
            return UploadResponse(
                doc_id=doc_id, paper_id=doc_id,
                error=f"PDF 解析失败: {e}",
            )

        # 层级分块并写入进程内文档索引
        chunker = self._ensure_chunker()
        chunks = chunker.chunk(full_text, doc_id=doc_id)
        self._index_doc(doc_id, filename, chunks)

        logger.info(
            "PDF 解析完成: doc_id=%s, pages=%d, chunks=%d",
            doc_id, num_pages, len(chunks),
        )
        return UploadResponse(
            doc_id=doc_id,
            paper_id=doc_id,
            num_pages=num_pages,
            num_chunks=len(chunks),
            formula_resolved=False,  # 文本层入库；公式 VLM 识别需真实 Key 按页处理
            need_review=num_pages > 0,
            title=filename,
            chunks=self.chunk_previews(chunks),
            parse_mode="text",
        )

    # ------------------------------------------------------------------
    # 可选 VLM 解析链路（/upload use_vlm=True，文档 4.1 节）
    # ------------------------------------------------------------------

    def run_parser_pipeline_vlm(
        self,
        filename: str,
        raw: bytes,
        user_id: str,
    ) -> UploadResponse:
        """可选 VLM 解析链路（文档 4.1 节多源融合）

        编排：PDF bytes → 临时文件 → PdfToImageConverter（前 MAX_VLM_PAGES 页）
        → QwenVlmClient.recognize_pages_sync → PdfplumberExtractor 版面
        → FusionEngine.fuse（公式四级绑定）→ 层级分块 → 进程内索引。

        成本控制：默认 /upload 走文本层；只有显式 use_vlm=True 才进入本方法，
        且最多处理前 10 页。mock 模式下强制 VLM 客户端 mock（并临时清空
        DASHSCOPE_API_KEY 兜底），保证 0 次真实 API 调用。

        fake PDF 无法被 fitz 打开时：mock 模式返回 "mock VLM 演示" 结果；
        真实模式回退文本层并标注 parse_mode="text_fallback"（不允许 500）。
        """
        import hashlib
        import tempfile

        doc_id = "doc_" + hashlib.md5(raw).hexdigest()[:12] if raw else "doc_empty"
        if not raw:
            return UploadResponse(doc_id=doc_id, paper_id=doc_id, error="file_b64 为空")

        tmp_path = ""
        try:
            # 1) parser 组件只接受文件路径：把 raw 落临时 PDF，finally 删除
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="vlm_parse_")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(raw)

            # 2) 前 MAX_VLM_PAGES 页转图（页面级输入，文档 4.1 节）
            from ..parser.pdf_to_image import PdfToImageConverter

            images = PdfToImageConverter().convert(
                tmp_path, page_range=(0, MAX_VLM_PAGES - 1),
            )
            if not images:
                # fake PDF / PyMuPDF 不可用：mock 演示；真实模式回退文本层
                if self.mock_mode:
                    return self.mock_vlm_upload(doc_id, filename)
                return self._vlm_text_fallback(
                    filename, raw, user_id, "PDF 转图失败（PyMuPDF 不可用或文件损坏）",
                )

            # 3) VLM 整页识别（mock 模式强制 mock，绝不真实调 API）
            vlm_results, vlm_client = self._recognize_pages_with_vlm(images, raw)

            # 4) pdfplumber 文本层版面（未安装/失败时降级为 VLM-only 融合）
            page_layouts = self._extract_page_layouts(tmp_path)

            # 5) 多源融合 + 公式四级绑定
            from ..parser.fusion_engine import FusionEngine

            fused = FusionEngine(vlm_client=vlm_client).fuse(
                doc_id, vlm_results, page_layouts, images,
            )

            # 6) 融合正文 → 层级分块 → 进程内索引
            full_text = self._fused_text(fused)
            if not full_text:
                raise RuntimeError("VLM 融合结果无正文")
            chunks = self._ensure_chunker().chunk(full_text, doc_id=doc_id)
            self._index_doc(doc_id, filename, chunks)

            # 7) 公式预览 + 解析质量标记
            formulas = self._collect_fused_formulas(fused)
            high_conf = [
                f for f in formulas
                if f["confidence"] >= VLM_FORMULA_REVIEW_THRESHOLD
            ]
            need_review = any(f["need_review"] for f in formulas)
            num_pages = len(images)

            logger.info(
                "VLM 解析完成: doc_id=%s, pages=%d, chunks=%d, formulas=%d, need_review=%s",
                doc_id, num_pages, len(chunks), len(formulas), need_review,
            )
            return UploadResponse(
                doc_id=doc_id,
                paper_id=doc_id,
                num_pages=num_pages,
                num_chunks=len(chunks),
                formula_resolved=bool(high_conf),
                need_review=need_review,
                title=filename,
                chunks=self.chunk_previews(chunks),
                vlm_used=True,
                formulas=formulas,
                parse_mode="vlm",
            )
        except Exception as e:  # noqa: BLE001 - 任何异常都不允许 500
            logger.error("VLM 解析链路失败: %s", e)
            if self.mock_mode:
                # mock 模式：走固定演示数据，保证前端联调可用
                return self.mock_vlm_upload(doc_id, filename)
            return self._vlm_text_fallback(filename, raw, user_id, str(e))
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning("临时 PDF 删除失败: %s", tmp_path)

    def _recognize_pages_with_vlm(self, images, raw: bytes):
        """调用 QwenVlmClient 识别前 N 页；mock 模式强制 mock，不触达 API

        Args:
            images: PageImage 列表（含 page_no + image_b64）
            raw: PDF 原始字节（计算 MD5 用于 VLM 结果缓存）
        Returns:
            (VlmPageResult 列表, 已配置好的 QwenVlmClient)
        """
        import hashlib

        from ..parser.vlm_client import QwenVlmClient

        pdf_md5 = hashlib.md5(raw).hexdigest()
        saved_key: Optional[str] = None
        if self.mock_mode:
            # 兜底：临时清空环境变量，避免 .env 中的真实 key 让 VLM 客户端走真实 API
            saved_key = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            client = QwenVlmClient()
            if self.mock_mode:
                # 双保险：即使客户端读到了 key，也强制 mock 模式
                client._mock_mode = True
            results = client.recognize_pages_sync(
                images=[(img.page_no, img.image_b64) for img in images],
                pdf_md5=pdf_md5,
            )
            return results, client
        finally:
            if saved_key is not None:
                os.environ["DASHSCOPE_API_KEY"] = saved_key

    def _extract_page_layouts(self, pdf_path: str):
        """pdfplumber 抽取页面版面（未安装/失败时返回空列表，降级 VLM-only 融合）"""
        try:
            from ..parser.text_extractor import PdfplumberExtractor

            return PdfplumberExtractor().extract(
                pdf_path, page_range=(0, MAX_VLM_PAGES - 1),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("pdfplumber 版面抽取失败（降级 VLM-only 融合）: %s", e)
            return []

    @staticmethod
    def _fused_text(fused) -> str:
        """从 FusedDocument 提取正文（按页内段落顺序拼接）"""
        parts: List[str] = []
        for page in getattr(fused, "pages", None) or []:
            for para in getattr(page, "paragraphs", None) or []:
                text = (getattr(para, "text", "") or "").strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _collect_fused_formulas(fused, limit: int = 20) -> List[Dict]:
        """从 FusedDocument 收集公式预览（最多 limit 条）

        字段：latex / confidence / page_no / paragraph_idx / need_review，
        另附 formula_id / binding_rule / location_desc 便于前端溯源展示。
        """
        formulas: List[Dict] = []

        def _append(bound, page_no: int) -> None:
            formulas.append({
                "latex": getattr(bound, "latex", ""),
                "confidence": round(float(getattr(bound, "confidence", 0.0) or 0.0), 3),
                "page_no": page_no,
                "paragraph_idx": getattr(bound, "bound_paragraph_idx", None),
                "need_review": bool(getattr(bound, "need_review", False)),
                "formula_id": getattr(bound, "formula_id", ""),
                "binding_rule": getattr(bound, "binding_rule", ""),
                "location_desc": getattr(bound, "location_desc", ""),
            })

        for page in getattr(fused, "pages", None) or []:
            page_no = int(getattr(page, "page_no", 0) or 0)
            for para in getattr(page, "paragraphs", None) or []:
                for bound in getattr(para, "bound_formulas", None) or []:
                    if len(formulas) >= limit:
                        return formulas
                    _append(bound, page_no)
            for bound in getattr(page, "unbound_formulas", None) or []:
                if len(formulas) >= limit:
                    return formulas
                _append(bound, page_no)
        return formulas

    def _vlm_text_fallback(
        self,
        filename: str,
        raw: bytes,
        user_id: str,
        reason: str,
    ) -> UploadResponse:
        """VLM 链路失败时回退文本层解析，并标注 parse_mode="text_fallback" """
        logger.warning("VLM 解析降级为文本层: %s", reason)
        try:
            response = self.run_parser_pipeline(filename, raw, user_id)
        except Exception as e:  # noqa: BLE001 - 降级也失败时返回 error，不抛 500
            return UploadResponse(
                error=f"VLM 解析失败（{reason}），文本层降级也失败: {e}",
                parse_mode="text_fallback",
            )
        response.vlm_used = False
        response.formulas = []
        response.parse_mode = "text_fallback"
        return response

    def run_ask(
        self,
        paper_id: str,
        question: str,
        user_id: str,
        use_memory: bool,
        top_k: int,
        history: Optional[List[Dict]] = None,
    ) -> AskResponse:
        """真实 RAG 问答（mock 模式不会进入此方法）

        文档第 3 章架构图：User → RAG 检索 → 记忆池 → LLM。
        按 paper_id 从进程内索引取 chunks → 向量检索 → LLM 生成。
        记忆池（P0-2）：上传时按 chunk 建好的 ContextManager 按问题检索实体
        （top_k=10），注入 prompt 的"记忆池实体"块；memory_entities 记录注入条目数
        （响应同时保留历史字段名 memory_items 作为兼容别名）。
        history 为前端会话记忆（[{"q": str, "a": str}, ...]），
        启用记忆池时把最近 3 轮 Q/A 注入 prompt 前缀（既有行为保留）。
        """
        chunks = self.get_doc_chunks(paper_id)
        if not chunks:
            return AskResponse(
                error=f"论文 {paper_id} 未找到或尚未解析，请先 POST /upload",
            )

        try:
            from ..rag.retriever import HybridRetriever
            embedder = self._ensure_embedder()
            # 文档 4.2 节定位：向量 + BM25 + RRF 混合检索（纯向量在短文档/特殊 embedder 下可能召回为空）
            retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
            retriever.add_chunks(chunks)

            t0 = time.time()
            retrieved = retriever.search(question, top_k=top_k)
            if not retrieved:
                return AskResponse(error="RAG 检索无结果")

            ctx = "\n".join(f"[{c.chunk.chunk_id}] {c.chunk.text}" for c in retrieved)[:3000]

            # --- P0-2 记忆池实体注入（文档 4.2 节"关键实体主动注入视野"）---
            # 注入位置：会话 history 前缀之后、参考正文之后、问题之前。
            # memory_prompt 自带"问题：..."行，作为"已知信息"块放在正文后，
            # 最终仍由末尾统一的问题行收尾（重复问题仅作强调，不影响 LLM 语义）。
            # 放在正文后还避免影响依赖 prompt"参考正文"前缀的 mock LLM 引用回显。
            memory_entities = 0
            memory_block = ""
            if use_memory:
                memory = self.get_context_manager(paper_id)
                if memory is not None:
                    try:
                        memory_prompt = memory.build_prompt(question, top_k=10)
                        memory_entities = _count_memory_items(memory_prompt, question)
                        if memory_entities > 0:
                            memory_block = (
                                f"—— 记忆池实体（{memory_entities} 条）：\n"
                                f"{memory_prompt}\n——\n\n"
                            )
                    except Exception as e:  # noqa: BLE001 - 降级为纯 RAG，不报错
                        logger.warning("记忆池注入失败（降级为纯 RAG）: %s", e)
                        memory_entities = 0

            # 会话历史注入：格式与前端现有逻辑保持一致（最近 3 轮 Q/A）
            history = history or []
            memory_prefix = ""
            history_used = False
            if use_memory and history:
                recent = history[-3:]
                memory_prefix = "—— 会话上下文（最近 3 轮）：\n" + "\n".join(
                    f"Q{i + 1}: {str(h.get('q', ''))[:80]} / "
                    f"A{i + 1}: {str(h.get('a', ''))[:80]}"
                    for i, h in enumerate(recent)
                ) + "\n——\n"
                history_used = True

            # memory_used 语义 = 本次是否真的注入了记忆上下文（实体池或会话历史）
            memory_used = memory_entities > 0 or history_used

            llm = self._ensure_llm()
            system = "你是学术问答助手。仅根据参考论文正文回答问题，并在关键结论处附 [chunk_id] 引用。"
            prompt = (
                f"{memory_prefix}"
                f"参考正文（chunk_id 前缀标注）：\n---\n{ctx}\n---\n\n"
                f"{memory_block}"
                f"问题：{question}\n\n"
                f"要求：答案 50-150 字，使用 [chunk_id] 作为引用编号。"
            )
            resp = llm.chat(prompt, system=system, temperature=0.2, max_tokens=512)
            latency_ms = int((time.time() - t0) * 1000)

            # 写入成本记录（文档 6.1 节）：缓存命中同样记录（cost_yuan=0），
            # 便于 /cost/summary 统计缓存命中率
            if resp.text:
                from datetime import datetime

                from ..observability import CostRecord

                self._ensure_cost_tracker().record(CostRecord(
                    ts=datetime.now(),
                    model=resp.model,
                    step="qa.ask",
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost_yuan=resp.cost_yuan,
                    user_id=user_id,
                    paper_id=paper_id,
                    from_cache=resp.from_cache,
                ))

            # 收集答案中实际出现的引用
            citations: List[Dict] = []
            for rr in retrieved:
                cid = rr.chunk.chunk_id
                if f"[{cid}]" in resp.text or f"{cid}" in resp.text:
                    citations.append({
                        "chunk_id": cid,
                        "doc_id": paper_id,
                        "text_preview": rr.chunk.text[:200],
                    })

            return AskResponse(
                answer=resp.text,
                citations=citations,
                trace_id="",
                from_cache=resp.from_cache,
                latency_ms=latency_ms if not resp.from_cache else resp.latency_ms,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_yuan=resp.cost_yuan,
                mock_mode=False,
                memory_used=memory_used,
                memory_entities=memory_entities,
                memory_items=memory_entities,  # 兼容别名
                error=resp.error,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("run_ask 失败: %s", e)
            return AskResponse(error=str(e))

    def run_review(
        self,
        topic: str,
        doc_ids: List[str],
        user_id: str,
        num_sections: int,
        verify_citations: bool,
        export_docx: bool,
    ) -> ReviewResponse:
        """综述生成（mock 模式同样走，pipeline 内部自动降级）

        P0-3/P0-4 接线：本方法构造共享 embedder + retriever 传给
        create_review_pipeline，使 Map-Reduce 的 MMR 去重与 CitationVerifier
        的二次检索/NLI 三色判定真正执行；响应透出：
        - generator_mode: 实际生成器模式（"mock"/"llm"，按 pipeline.generator 类型判断）；
        - mmr_enabled: MMR 能力是否启用（LLM 生成器路径 + embedder 已传 → True；
          MockReviewGenerator 路径固定 False，实际是否执行看 dedup_stats.calls / mmr_applied）；
        - dedup_stats / mmr_applied: MMR 执行统计（跳过也计数，不允许静默）；
        - verification_skipped / verification_skip_reason: 引用验证是否被跳过及原因
          （verification_error 为同值兼容别名）。
        本次 doc_ids 均无可用 chunks 时：不构造空 retriever，retriever/embedder 置 None，
        显式返回 verification_skipped=True + reason，而不是静默跳过。
        """
        import base64 as _base64
        import re as _re
        import time as _time

        # 构造 doc_infos / doc_chunks：真实场景从进程内文档索引取已解析的 chunks；
        # 未上传/未解析的 doc_id 用空 chunks（综述生成器会跳过并给出提示段落）。
        doc_infos: List[Dict] = []
        doc_chunks: Dict[str, List] = {}
        for did in doc_ids:
            with self._docs_lock:
                doc = self._docs.get(did)
            if doc is not None:
                doc_infos.append({"doc_id": did, "title": doc.get("title", did)})
                doc_chunks[did] = list(doc.get("chunks", []))
            else:
                doc_infos.append({"doc_id": did, "title": f"文献 {did}（未解析）"})
                doc_chunks[did] = []

        # --- P0-3 / P0-4 接线：构造共享 embedder + retriever ---
        # 背景：此前 retriever=None / embedder=None 传入 create_review_pipeline，
        # 导致 generator 的 MMR 去重与 CitationVerifier 的二次检索/NLI 被静默跳过。
        # 现在按本次 doc_chunks 建索引：
        # - mock 模式用 FakeEmbedder（确定性哈希向量，0 次真实 API 调用）；
        # - 真实模式复用进程内 embedder（DashScope text-embedding-v3）；
        # - 本次没有任何可用 chunks（未上传/未解析）时不构造空 retriever：
        #   retriever/embedder 保持 None，后续响应显式标记 verification_skipped。
        from ..rag.embedder import FakeEmbedder
        from ..rag.retriever import HybridRetriever

        all_chunks = [c for chunks in doc_chunks.values() for c in chunks]
        embedder = None
        retriever = None
        no_retriever_reason = ""
        if all_chunks:
            try:
                embedder = FakeEmbedder(dim=64) if self.mock_mode else self._ensure_embedder()
                retriever = HybridRetriever(embedder, use_bm25=True, reranker=None)
                retriever.add_chunks(all_chunks)
            except Exception as e:  # noqa: BLE001 - 构造失败降级为纯生成，不返回 500
                logger.warning("综述检索器构造失败（降级为无 MMR/无引用验证）: %s", e)
                embedder = None
                retriever = None
                no_retriever_reason = f"检索器/向量化器构造失败（{type(e).__name__}）"
        else:
            no_retriever_reason = "doc_ids 均未解析或没有可用 chunks"
            logger.info(
                "综述无可用 chunks，跳过检索器构建与引用验证: doc_ids=%s", doc_ids
            )

        # 调 ReviewPipeline
        try:
            from ..review.generator import REDUCE_TOP_K, MockReviewGenerator
            from ..review.pipeline import create_review_pipeline
            t0 = _time.time()
            pipeline = create_review_pipeline(
                retriever=retriever,
                embedder=embedder,
                mode="auto" if not self.mock_mode else "mock",
                verify_citations=verify_citations,
            )
            # mock 模式强制规则化 NLI（NoopReranker：chunk 存在 → neutral，
            # chunk 不存在 → contradiction）。这样即使开发机 .env 里配了真实 key，
            # mock 请求也绝不触达真实大模型 NLI API。
            if self.mock_mode and pipeline.citation_verifier is not None:
                from ..rag.reranker import NoopReranker
                pipeline.citation_verifier.nli_reranker = NoopReranker()

            document = pipeline.run(
                topic=topic,
                doc_infos=doc_infos,
                doc_chunks=doc_chunks,
                num_sections=num_sections,
                verify_citations=verify_citations,
            )
            latency_ms = int((_time.time() - t0) * 1000)

            # 可选 Word 导出：内存渲染（不传 output_path，避免写服务器 cwd），
            # 结果 base64 回传由前端触发下载
            docx_path = ""
            docx_b64 = ""
            docx_mode = ""
            docx_name = ""
            docx_size = 0
            if export_docx:
                try:
                    from ..export import create_docx_renderer
                    renderer = create_docx_renderer("auto")
                    # 文件名清洗：主题可能含 / \ : * ? " < > | 等 Windows 非法字符
                    raw_topic = document.topic or "review"
                    safe_topic = _re.sub(r'[\\/:*?"<>|\r\n\t]', "_", raw_topic)[:20]
                    render_result = renderer.render(document)
                    if render_result.mode == "docx":
                        raw = render_result.bytes
                        suffix = "docx"
                    else:
                        raw = render_result.markdown.encode("utf-8")
                        suffix = "md"
                    docx_b64 = _base64.b64encode(raw).decode("ascii")
                    docx_mode = render_result.mode
                    docx_name = f"review_{safe_topic}.{suffix}"
                    docx_size = render_result.size
                except Exception as e:  # noqa: BLE001
                    logger.warning("Word 导出失败: %s", e)

            # --- P0-3 MMR 统计：成功执行/跳过都计数，不允许静默 ---
            mmr_stats = getattr(pipeline.generator, "mmr_stats", None) or {}
            mmr_calls = int(mmr_stats.get("calls", 0) or 0)
            dedup_stats: Dict[str, Any] = {
                "mmr_applied": bool(mmr_calls),
                "calls": mmr_calls,
                "input_summaries": int(mmr_stats.get("input_summaries", 0) or 0),
                "selected_summaries": int(
                    mmr_stats.get("selected_summaries", 0) or 0
                ),
                "skipped": int(mmr_stats.get("skipped", 0) or 0),
                "reduce_top_k": int(
                    getattr(pipeline.generator, "REDUCE_TOP_K", REDUCE_TOP_K)
                ),
                "mmr_lambda": getattr(pipeline.generator, "mmr_lambda", 0.7),
            }

            # --- P0-3/P0-4 契约字段：生成器模式 / MMR 是否启用 / 引用验证跳过原因 ---
            # generator_mode 按 pipeline.generator 的**实际类型**判断（不依赖入参），
            # 因此真实模式即便内部降级为 mock 组件，也能如实区分 LLM 路径与 mock 路径。
            generator_mode = (
                "mock" if isinstance(pipeline.generator, MockReviewGenerator) else "llm"
            )
            # mmr_enabled 语义（契约）：LLM 生成器路径且 embedder 已传入 → 能力启用；
            # MockReviewGenerator 路径固定 False（避免文档把 mock 路径写成“MMR 已生效”）。
            # 实际是否执行 MMR 看 dedup_stats.calls / mmr_applied（跳过也计数）。
            mmr_enabled = bool(
                embedder is not None
                and not isinstance(pipeline.generator, MockReviewGenerator)
            )

            # --- P0-4 引用验证状态：retriever 已传 → CitationVerifier 自动创建 ---
            verification_skipped = bool(
                verify_citations and pipeline.citation_verifier is None
            )
            verification_skip_reason = ""
            if verification_skipped:
                if retriever is None:
                    verification_skip_reason = (
                        no_retriever_reason
                        or "未构建检索器（无可用 chunks），引用事实验证被跳过"
                    )
                else:
                    verification_skip_reason = (
                        "CitationVerifier 创建失败，引用事实验证被跳过"
                    )
                logger.warning("综述引用验证被跳过: %s", verification_skip_reason)
            # verified 语义自洽：verify_citations=False 或验证被跳过时一律 False；
            # 其余情况采用 pipeline 的真实验证结果（不静默改成 False）
            verified = bool(document.verified) and not verification_skipped

            return ReviewResponse(
                title=document.title,
                full_text=document.full_text,
                full_html=document.full_html,
                outline=document.outline,
                paragraphs=document.paragraphs,
                citation_stats=document.citation_stats,
                references=document.references,
                verified=verified,
                generator_mode=generator_mode,
                mmr_enabled=mmr_enabled,
                mmr_applied=dedup_stats["mmr_applied"],
                dedup_stats=dedup_stats,
                verification_skipped=verification_skipped,
                verification_skip_reason=verification_skip_reason,
                verification_error=verification_skip_reason,  # 兼容别名
                docx_path=docx_path,
                docx_b64=docx_b64,
                docx_mode=docx_mode,
                docx_name=docx_name,
                docx_size=docx_size,
                trace_id="",
                latency_ms=latency_ms,
                mock_mode=self.mock_mode,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("run_review 失败: %s", e)
            return ReviewResponse(error=str(e))


# ---------------------------------------------------------------------------
# 模块级单例 + 便捷启动
# ---------------------------------------------------------------------------


_app_cache = None


def get_app():
    """获取全局 FastAPI 应用单例

    首次调用时根据 DASHSCOPE_API_KEY 决定 mock 模式。
    """
    global _app_cache
    if _app_cache is None:
        mock = not bool(os.getenv("DASHSCOPE_API_KEY", ""))
        _app_cache = create_app(mock_mode=mock)
    return _app_cache


def run_dev(host: str = "0.0.0.0", port: int = 8000) -> None:
    """开发模式启动 FastAPI（uvicorn）

    Args:
        host: 监听地址
        port: 监听端口
    """
    try:
        import uvicorn  # type: ignore
    except ImportError:
        logger.error("uvicorn 未安装，无法启动开发服务器。pip install uvicorn")
        return

    app = get_app()
    if app is None:
        logger.error("FastAPI 应用创建失败（fastapi 未安装？）")
        return

    logger.info("启动开发服务器: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)

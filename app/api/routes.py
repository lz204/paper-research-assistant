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
- 请求/响应 schema dataclass：UploadRequest / AskRequest / ReviewRequest
- create_app：创建 FastAPI 应用（fastapi 可用时）
- 路由：POST /upload、POST /ask、POST /review、GET /health
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文档 6.5 节安全限制
# ---------------------------------------------------------------------------


MAX_QUESTION_LEN = 2000     # 用户提问限 2000 字
MAX_REVIEW_TOPIC_LEN = 500  # 综述主题限 500 字
MAX_PDF_PAGES = 200         # 单 PDF 页数限制
MAX_PDF_SIZE_MB = 50        # 单 PDF 文件大小限制


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
    """

    filename: str = ""
    file_b64: str = ""
    user_id: str = ""


@dataclass
class UploadResponse:
    """PDF 上传响应"""

    doc_id: str = ""
    paper_id: str = ""
    num_pages: int = 0
    num_chunks: int = 0
    formula_resolved: bool = True
    need_review: bool = False
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


@dataclass
class AskResponse:
    """论文问答响应"""

    answer: str = ""
    citations: List[Dict] = field(default_factory=list)
    trace_id: str = ""
    from_cache: bool = False
    latency_ms: int = 0
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
    """综述生成响应"""

    title: str = ""
    full_text: str = ""
    full_html: str = ""
    outline: List[Dict] = field(default_factory=list)
    paragraphs: List[Dict] = field(default_factory=list)
    citation_stats: Dict[str, int] = field(default_factory=dict)
    references: List[Dict] = field(default_factory=list)
    verified: bool = False
    docx_path: str = ""
    trace_id: str = ""
    latency_ms: int = 0
    error: str = ""


@dataclass
class HealthResponse:
    """健康检查响应"""

    status: str = "ok"
    version: str = "0.1.0"
    mock_mode: bool = True
    components: Dict[str, str] = field(default_factory=dict)


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

    class AskBody(_BaseModel):
        paper_id: str = ""
        question: str = ""
        user_id: str = ""
        use_memory: bool = True
        top_k: int = 5

    class ReviewBody(_BaseModel):
        topic: str = ""
        doc_ids: List[str] = []
        user_id: str = ""
        num_sections: int = 4
        verify_citations: bool = True
        export_docx: bool = False

except ImportError:  # pragma: no cover - fastapi/pydantic 未安装时的降级占位
    # fastapi 不可用时 create_app 直接返回 None，路由不会用到这些模型，
    # 这里仅保证模块级 import 不报错。
    class UploadBody:  # type: ignore[no-redef]
        pass

    class AskBody:  # type: ignore[no-redef]
        pass

    class ReviewBody:  # type: ignore[no-redef]
        pass


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

        # mock 模式：跳过实际解析，返回模板化结果
        if state.mock_mode:
            return UploadResponse(
                doc_id=doc_id,
                paper_id=doc_id,
                num_pages=0,
                num_chunks=0,
                formula_resolved=False,
                need_review=True,
            )

        # 真实模式：调 app.parser 流水线
        try:
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

        # mock 模式：返回模板化回答
        if state.mock_mode:
            return state.mock_ask(
                body.paper_id, body.question, body.user_id,
            )

        # 真实模式：调 RAG + LLM
        try:
            result = state.run_ask(
                paper_id=body.paper_id,
                question=body.question,
                user_id=body.user_id,
                use_memory=body.use_memory,
                top_k=body.top_k,
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
            return ReviewResponse(error="doc_ids 不能为空")

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

    return app


# ---------------------------------------------------------------------------
# 应用状态（依赖注入容器）
# ---------------------------------------------------------------------------


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
        self._docs_lock = threading.Lock()
        # 可注入组件（测试可替换；None 时按需懒加载）
        self._embedder = None
        self._llm_client = None
        self._chunker = None

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

    def _index_doc(self, doc_id: str, title: str, chunks) -> None:
        """把分块结果写入进程内文档索引"""
        with self._docs_lock:
            self._docs[doc_id] = {"title": title, "chunks": list(chunks)}

    def get_doc_chunks(self, doc_id: str):
        """按 doc_id 取 chunks（供真实 run_ask / run_review 使用）"""
        with self._docs_lock:
            doc = self._docs.get(doc_id)
        return list(doc["chunks"]) if doc else []

    # ------------------------------------------------------------------
    # mock 模式辅助
    # ------------------------------------------------------------------

    @staticmethod
    def mock_ask(
        paper_id: str,
        question: str,
        user_id: str,
    ) -> AskResponse:
        """mock 模式问答响应"""
        preview = question.strip().replace("\n", " ")[:60]
        return AskResponse(
            answer=(
                f"[Mock 回答] 关于「{preview}...」的模板化回答。"
                f"实际部署需配置 DASHSCOPE_API_KEY 后由 Qwen-Plus 生成。"
            ),
            citations=[],
            trace_id="mock-trace",
            from_cache=False,
            latency_ms=1,
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
        )

    def run_ask(
        self,
        paper_id: str,
        question: str,
        user_id: str,
        use_memory: bool,
        top_k: int,
    ) -> AskResponse:
        """真实 RAG 问答（mock 模式不会进入此方法）

        文档第 3 章架构图：User → RAG 检索 → 记忆池 → LLM。
        按 paper_id 从进程内索引取 chunks → 向量检索 → LLM 生成。
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

            llm = self._ensure_llm()
            system = "你是学术问答助手。仅根据参考论文正文回答问题，并在关键结论处附 [chunk_id] 引用。"
            prompt = (
                f"参考正文（chunk_id 前缀标注）：\n---\n{ctx}\n---\n\n"
                f"问题：{question}\n\n"
                f"要求：答案 50-150 字，使用 [chunk_id] 作为引用编号。"
            )
            resp = llm.chat(prompt, system=system, temperature=0.2, max_tokens=512)
            latency_ms = int((time.time() - t0) * 1000)

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
        """综述生成（mock 模式同样走，pipeline 内部自动降级）"""
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

        # 调 ReviewPipeline
        try:
            from ..review.pipeline import create_review_pipeline
            t0 = _time.time()
            pipeline = create_review_pipeline(
                retriever=None,  # API 层不直接持有 retriever
                embedder=None,
                mode="auto",
                verify_citations=verify_citations,
            )
            document = pipeline.run(
                topic=topic,
                doc_infos=doc_infos,
                doc_chunks=doc_chunks,
                num_sections=num_sections,
                verify_citations=verify_citations,
            )
            latency_ms = int((_time.time() - t0) * 1000)

            # 可选 Word 导出
            docx_path = ""
            if export_docx:
                try:
                    from ..export import create_docx_renderer
                    renderer = create_docx_renderer("auto")
                    # 文件名清洗：主题可能含 / \ : * ? " < > | 等 Windows 非法字符
                    safe_topic = _re.sub(r'[\\/:*?"<>|\r\n\t]', "_", document.topic or "review")[:40]
                    out_path = os.path.join(
                        os.getcwd(), f"review_{safe_topic}.docx"
                    )
                    render_result = renderer.render(document, out_path)
                    docx_path = render_result.path or ""
                except Exception as e:  # noqa: BLE001
                    logger.warning("Word 导出失败: %s", e)

            return ReviewResponse(
                title=document.title,
                full_text=document.full_text,
                full_html=document.full_html,
                outline=document.outline,
                paragraphs=document.paragraphs,
                citation_stats=document.citation_stats,
                references=document.references,
                verified=document.verified,
                docx_path=docx_path,
                trace_id="",
                latency_ms=latency_ms,
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

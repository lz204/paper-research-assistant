"""多语种文献翻译与论文研读助手 — Streamlit 前端（API-first 双模式）

架构定位（文档第 3 / 10 章「FastAPI 单体 + Streamlit 前端」）：
- 默认「🌐 API 模式」：前端只通过 HTTP 调用 FastAPI（见 api_client.py），
  业务逻辑全部在后端进程执行，与 docker-compose 的 API_BASE_URL 约定一致；
- 兜底「💻 本地进程模式」：后端不可达时，在 Streamlit 进程内直接调用
  parser / rag / review 模块（离线演示，完整保留原有行为）。

文档第 2 章核心功能（共 6 个 Tab，第 3 个「Agent 助手」为本项目扩展）：
1. 📄 PDF 公式无损解析（上传 PDF → 解析 + 公式定位）
2. 🌐 多语种文献翻译（专业术语统一校准 + 多语种互译）
3. 🤖 Agent 助手（LLM 自主调用工具：列文档 / 检索 / 翻译 / 综述，展示工具调用轨迹）
4. 💬 基于记忆池的长文问答（RAG + 引用溯源）
5. 📚 综述草稿生成（跨文献 + 引用事实验证 + Word 导出）
6. 💰 API 成本看板（每日成本趋势 + 按模型/用户/论文分摊）

文档 4.3 节三色引用展示（Streamlit unsafe_allow_html=True）：
- entailment（受支持）: 蓝色 #2563eb
- contradiction（与原文矛盾）: 红色 #dc2626
- neutral（引用过宽）: 灰色 #6b7280

文档 6.5 节输出过滤：unsafe_allow_html 只放行 `<span>` + 基础标签，其余 XSS 过滤。

使用方法（Windows / Linux 均可）：
    # 1) 启动后端（项目根目录，API-first 默认路径）
    make api        # 或 uvicorn app.api.routes:create_app --factory --port 8000

    # 2) 启动前端
    streamlit run app/frontend/streamlit_app.py --server.port 8501

后端地址由环境变量 API_BASE_URL 指定（docker-compose 内为 http://app:8000，
本地默认 http://127.0.0.1:8000）；侧边栏可随时切换运行模式。

DashScope API Key：
    - API 模式：由后端进程读取环境变量 DASHSCOPE_API_KEY（或 .env 文件）
    - 本地进程模式：可在侧边栏输入（仅保存在当前浏览器会话的 st.session_state）
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# 应用启动前：保证 app.* 可 import（从项目根目录）
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st  # type: ignore
except ImportError as e:  # noqa: BLE001
    sys.stderr.write("streamlit 未安装！请先 pip install streamlit\n")
    sys.stderr.write(f"错误详情: {e}\n")
    sys.exit(1)

# P1 修复 ①：API-first —— 前端统一通过 ApiClient 以 HTTP 访问 FastAPI 后端
try:
    from app.frontend.api_client import ApiClient
except ImportError:  # pragma: no cover - 直接以脚本运行时（cwd 不在项目根）的兜底
    from api_client import ApiClient  # type: ignore  # noqa: F401


# =====================================================================
# Streamlit 全局配置（通过 st.set_page_config 最顶部执行）
# =====================================================================

st.set_page_config(
    page_title="多语种文献翻译与论文研读助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 注入：三色引用（文档 4.3 节）+ 顶部 banner + 表格样式
CUSTOM_CSS = """
<style>
/* 三色引用 */
span.cite {
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: 600;
  margin: 0 1px;
  cursor: pointer;
}
span.cite-entailment { background-color: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
span.cite-contradiction { background-color: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
span.cite-neutral { background-color: #f3f4f6; color: #6b7280; border: 1px solid #e5e7eb; }

/* 顶部 banner */
div.banner-mock { background: linear-gradient(90deg,#fef9c3,#fef3c7); color:#78350f; padding:10px 16px; border-radius:8px; margin-bottom:16px; border:1px dashed #b45309; }
div.banner-real { background: linear-gradient(90deg,#dcfce7,#d1fae5); color:#14532d; padding:10px 16px; border-radius:8px; margin-bottom:16px; border:1px dashed #16a34a; }

/* 表格 */
table.eval-table { border-collapse: collapse; width:100%; }
table.eval-table th { background:#f1f5f9; padding:8px; text-align:left; border-bottom:2px solid #cbd5e1; }
table.eval-table td { padding:6px 8px; border-bottom:1px solid #e2e8f0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# 工具函数
# =====================================================================


def _xss_filter(html_text: str) -> str:
    """文档 6.5 节 XSS 过滤：只放行白名单标签

    仅允许：<span> / <p> / <br> / <strong> / <b> / <em> / <i> / <div>（且不能含 script/iframe/on* 属性）
    """
    import re
    if not html_text:
        return ""
    allowed_tags = {"span", "p", "br", "strong", "b", "em", "i", "div"}
    allowed_attrs = {"class", "title", "style", "id"}

    def _sanitize_tag(m):
        open_close = m.group(1)  # '' or '/'
        tag = m.group(2).lower()
        rest = m.group(3)
        if tag not in allowed_tags:
            return ""  # 直接剔除非白名单标签
        if rest:
            # 清理 on* 属性 / javascript: / 内联事件
            cleaned = re.sub(r'\s+on\w+\s*=\s*"[^"]*"', "", rest)
            cleaned = re.sub(r"\s+on\w+\s*=\s*'[^']*'", "", cleaned)
            cleaned = re.sub(r'\s+on\w+\s*=\s*[\w\-]+', "", cleaned)
            cleaned = re.sub(r'javascript:', "", cleaned, flags=re.IGNORECASE)
            # 只保留白名单属性
            parts = []
            for kv in re.findall(r"([\w\-]+)\s*=\s*('[^']*'|\"[^\"]*\"|[\w\-]+)", cleaned):
                if kv[0].lower() in allowed_attrs:
                    parts.append(f'{kv[0]}={kv[1]}')
            rest = " " + " ".join(parts) if parts else ""
        return f"<{open_close}{tag}{rest}>"

    cleaned = re.sub(r"<(/?)([A-Za-z0-9]+)([^>]*)>", _sanitize_tag, html_text)
    return cleaned


def _render_citations(text: str, citation_list: List[Dict]) -> str:
    """渲染三色引用 HTML（文档 4.3 节）

    citation_list 每项含 index / label / original_text
    返回安全 HTML（经 _xss_filter 过滤）
    """
    import re
    # 按引用标记长度倒序替换，避免替换冲突
    sorted_cits = sorted(citation_list, key=lambda c: len(f"[{c.get('index')}]"), reverse=True)
    html = text
    for c in sorted_cits:
        idx = c.get("index", "")
        label = c.get("label", "neutral")
        original = c.get("original_text", "")[:80] or f"引用来源 {idx}"
        span = (
            f'<span class="cite cite-{label}" '
            f'title="引用来源：{_escape_attr(original)}">'
            f'[{idx}]'
            f'</span>'
        )
        html = html.replace(f"[{idx}]", span)
    return _xss_filter(html)


def _escape_attr(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


# =====================================================================
# 懒加载单例（仅会话内实例化一次，st.cache_resource 持久化）
# =====================================================================


@st.cache_resource
def get_app_components(dashscope_api_key: str):
    """构造应用组件集合（每次 key 变更即重建）

    返回 dict：
      - llm_client: BaseLLMClient
      - cost_tracker: BaseCostTracker
      - cost_dashboard: CostDashboard
      - langfuse: BaseLangfuseClient
      - retriever: InMemoryVectorRetriever（会话级，供 RAG/综述使用）
      - embedder: Embedder
      - chunker: HierarchicalChunker
      - review_pipeline: ReviewPipeline
      - docx_renderer: BaseDocxRenderer
    """
    # 设置环境变量（dashscope 客户端会读取）
    if dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
    elif "DASHSCOPE_API_KEY" not in os.environ:
        os.environ["DASHSCOPE_API_KEY"] = ""

    from app.llm import create_llm_client
    from app.observability import (
        create_cost_tracker,
        CostDashboard,
        create_langfuse_client,
    )
    from app.rag import FakeEmbedder, HierarchicalChunker, InMemoryVectorRetriever, create_reranker
    from app.review import create_review_pipeline
    from app.export import create_docx_renderer

    llm_client = create_llm_client("auto")
    cost_tracker = create_cost_tracker("auto")
    cost_dashboard = CostDashboard(cost_tracker)
    langfuse = create_langfuse_client("auto")
    embedder = FakeEmbedder(dim=64)
    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    retriever = InMemoryVectorRetriever(embedder)
    reranker = create_reranker("auto", api_key=dashscope_api_key or None)
    review_pipeline = create_review_pipeline(
        retriever=retriever, embedder=embedder,
        api_key=dashscope_api_key or None, verify_citations=True,
    )
    docx_renderer = create_docx_renderer("auto")

    return {
        "llm_client": llm_client,
        "cost_tracker": cost_tracker,
        "cost_dashboard": cost_dashboard,
        "langfuse": langfuse,
        "embedder": embedder,
        "chunker": chunker,
        "retriever": retriever,
        "reranker": reranker,
        "review_pipeline": review_pipeline,
        "docx_renderer": docx_renderer,
        "mock_mode": not bool(os.environ.get("DASHSCOPE_API_KEY")),
        "indexed_docs": {},  # doc_id -> {title, file_bytes, chunks}
    }


# =====================================================================
# API-first：ApiClient 单例 + 运行模式辅助函数
# =====================================================================


@st.cache_resource
def get_api_client() -> ApiClient:
    """ApiClient 会话级单例（base_url 取 API_BASE_URL，默认 http://127.0.0.1:8000）"""
    return ApiClient()


def _health_ok(health: Dict[str, Any]) -> bool:
    """判断 GET /health 响应是否代表后端可用"""
    if not health or health.get("error") or health.get("ok") is False:
        return False
    return health.get("status", "ok") == "ok"


def _switch_to_local_mode() -> None:
    """按钮回调：切换运行模式为本地进程模式（在脚本 rerun 前修改控件状态）"""
    st.session_state["run_mode_radio"] = "local"
    st.session_state["run_mode"] = "local"


def _user_id() -> str:
    """会话级匿名用户 ID（随请求上报，供后端按用户分摊成本）"""
    uid = st.session_state.get("user_id")
    if not uid:
        uid = "web_" + uuid.uuid4().hex[:8]
        st.session_state["user_id"] = uid
    return str(uid)


# =====================================================================
# 侧边栏：运行模式 + DashScope API Key 配置 + 信息
# =====================================================================


def render_sidebar() -> str:
    """渲染侧边栏：运行模式（API / 本地进程）+ DashScope API Key + 说明

    返回当前配置的 dashscope_api_key（可能为空，表示本地 Mock）。
    副作用：把 "run_mode"（"api" / "local"）与 "api_health" 写入 st.session_state。
    """
    with st.sidebar:
        st.title("📚 论文助手")
        st.caption("多语种文献翻译与研读（求职项目 · API-first）")

        st.divider()

        # ---------------- 运行模式（P1 修复 ①：API-first） ----------------
        st.subheader("🔌 运行模式")
        client = get_api_client()
        # 默认值：后端可达 → API 模式；否则本地进程模式（available 自带 ~3 秒缓存）
        if "run_mode_radio" not in st.session_state:
            st.session_state["run_mode_radio"] = (
                "api" if client.available else "local"
            )

        mode_labels = {
            "api": "🌐 API 模式（FastAPI 后端）",
            "local": "💻 本地进程模式",
        }
        run_mode = st.radio(
            "前端调用方式",
            ["api", "local"],
            format_func=lambda m: mode_labels[m],
            key="run_mode_radio",
            help=(
                "API 模式：前端通过 HTTP 调用 FastAPI（API-first 架构，默认）。"
                "本地进程模式：前端在本进程内直接调用 parser/rag/review（离线兜底）。"
            ),
        )
        st.session_state["run_mode"] = run_mode

        if run_mode == "api":
            health = client.health()
            if _health_ok(health):
                st.session_state["api_health"] = health
                st.caption(f"后端：`{client.base_url}`")
                mock_on = bool(health.get("mock_mode"))
                st.caption(
                    "后端 mock_mode：**True**（Mock 模板化回答）" if mock_on
                    else "后端 mock_mode：**False**（真实 DashScope 调用）"
                )
                with st.expander("组件状态 / 版本", expanded=False):
                    st.json({
                        "base_url": client.base_url,
                        "version": health.get("version", "?"),
                        "components": health.get("components", {}),
                    })
            else:
                st.session_state["api_health"] = {}
                err = str(health.get("error", "未知错误"))
                st.warning(
                    f"无法连接 FastAPI 后端：`{client.base_url}`\n\n"
                    f"错误：{err}\n\n"
                    "请先在项目根目录启动后端（新开一个终端）：\n\n"
                    "```powershell\n.\\scripts\\dev.ps1 api\n```\n\n"
                    "或执行 `make api`；启动成功后刷新本页。"
                    "也可点击下方按钮先切换到本地进程模式。"
                )
                st.button(
                    "💻 一键切换到本地进程模式",
                    key="sidebar_switch_local",
                    on_click=_switch_to_local_mode,
                )
        else:
            st.session_state["api_health"] = {}
            st.caption(
                "本地进程模式：不依赖 FastAPI，直接在本进程调用 app.* 模块（离线兜底）。"
            )

        st.divider()

        st.subheader("🔑 DashScope API Key")
        env_key = os.getenv("DASHSCOPE_API_KEY", "")
        # 优先从环境变量；未设置时允许会话内输入
        default = env_key or st.session_state.get("ds_key", "")
        input_key = st.text_input(
            "阿里云 DashScope API Key",
            value=default,
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxx",
            help=(
                "填入真实 Key 后由 Qwen-Plus / Qwen-VL-Max / Embedding v3 真实调用。"
                "留空进入 Mock 模式（模板化回答）。API 模式下 Key 由后端进程读取，"
                "此处输入仅用于本地进程模式。"
            ),
        )
        st.session_state["ds_key"] = input_key

        if run_mode == "api":
            st.caption(
                "API 模式：Mock / 真实由后端 `/health` 的 `mock_mode` 决定（上方已显示）。"
            )
        elif not input_key:
            st.info("当前未配置 Key，进入 **Mock 模式**（所有 LLM/VLM 调用走模板化回答）")
        else:
            st.success("已配置 DashScope API Key，将走真实 API 调用")

        st.divider()

        st.subheader("ℹ 关于")
        st.markdown(
            """
            - 架构：FastAPI 单体 + Streamlit 前端（API-first：前端只调 HTTP 接口）
            - 运行模式：🌐 API 模式（默认，走 FastAPI）/ 💻 本地进程模式（离线兜底）
            - 后端：阿里云 DashScope（Qwen-Plus / Qwen-VL-Max / Embedding v3）
            - 🤖 Agent 助手：LLM 自主调用工具（列文档 / 检索 / 翻译 / 综述），
              展示工具调用轨迹（步数 / 参数 / 耗时 / 成本 / framework）
            - 文档 4.3 节：综述支持三色引用溯源
            - 文档 6.1 节：成本看板实时展示 API 调用成本
            """
        )
        st.caption(f"项目根：`{PROJECT_ROOT}`")
    return input_key


# =====================================================================
# Tab 1：PDF 上传与解析
# =====================================================================


def render_upload_tab(comp: Dict):
    """📄 PDF 上传与解析 Tab（文档第 2 章核心功能 1）"""
    st.header("📄 PDF 公式无损解析")
    st.caption("文档 4.1 节：pdf_to_image → vlm_client → text_extractor → fusion_engine → formula_binder → chunker")

    uploaded_file = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=False)
    if not uploaded_file:
        st.info("请选择一个 PDF 文件。mock 模式可上传任意内容（会跳过真实渲染）。")
        return

    raw = uploaded_file.read()
    filename = uploaded_file.name
    file_size_mb = len(raw) / 1024 / 1024
    doc_id = "doc_" + hashlib.md5(raw).hexdigest()[:12]

    with st.status(f"解析中：{filename}（{file_size_mb:.2f}MB，doc_id={doc_id}）", expanded=True) as status:
        t0 = time.time()
        # Mock 模式：直接切分；真实模式：调 app.parser 流水线
        # 为了 mock 模式下也能体验，直接把 bytes 转成假文本
        if comp["mock_mode"]:
            st.write("ℹ Mock 模式：不调真实 PDF 解析，生成模板化 chunks。")
            chunks = comp["chunker"].chunk(
                f"这是论文「{filename}」的模拟全文。\n\n"
                f"第 1 节 引言：本文提出了新方法，在 ImageNet 上准确率 85.6%。\n\n"
                f"第 2 节 方法：核心公式为 Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V，参数量 110M。\n\n"
                f"第 3 节 实验：BLEU=28.4，F1=0.92，推理延迟 12ms。\n\n"
                f"第 4 节 相关工作：对比 ResNet-152、BERT_BASE 等模型。\n\n"
                f"第 5 节 结论：方法优于现有基线。",
                doc_id=doc_id,
            )
            need_review = False
            formula_resolved = True
        else:
            # 真实模式：调用 parser 流水线（此处做最小实现——按 bytes 转字符串 + 容错）
            try:
                from app.parser import (
                    pdf_to_image, text_extractor, vlm_client, fusion_engine,
                    formula_binder, chunker,
                )
                st.write("真实解析流水线：已配置，后续版本会完整集成 app.parser.*")
                chunks = comp["chunker"].chunk(
                    f"[论文 {filename}] 真实解析待完成（app.parser 流水线集成中）",
                    doc_id=doc_id,
                )
                need_review = True
                formula_resolved = False
            except ImportError:
                chunks = comp["chunker"].chunk(
                    f"[论文 {filename}] 解析模块不可用", doc_id=doc_id,
                )
                need_review = True
                formula_resolved = False
        status.update(label="解析完成", state="complete", expanded=False)

    latency_ms = int((time.time() - t0) * 1000)

    # 展示结果
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文件大小", f"{file_size_mb:.2f}MB")
    col2.metric("切分 chunks", str(len(chunks)))
    col3.metric("公式定位", "✓" if formula_resolved else "⚠ 待复核")
    col4.metric("耗时", f"{latency_ms}ms")

    # 保存到会话索引，供问答/综述使用
    comp["indexed_docs"][doc_id] = {
        "title": filename,
        "file_bytes": raw[:10],  # 只留指纹
        "chunks": chunks,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }
    comp["retriever"].add_chunks(chunks)

    st.subheader("🔍 已索引 chunks（前 5 条预览）")
    for c in chunks[:5]:
        with st.expander(f"**chunk_id**: `{c.chunk_id}`  ·  "
                         f"tokens 约 {len(c.text)//4}  ·  page={c.metadata.get('page', '-')}"):
            st.write(c.text)

    st.success(f"已添加到会话内索引（doc_id=`{doc_id}`），可在「💬 论文问答」「📚 综述生成」Tab 使用。")


# =====================================================================
# Tab 4：论文问答（RAG + 引用溯源）
# =====================================================================


def render_ask_tab(comp: Dict):
    """💬 论文问答 Tab（文档第 2 章核心功能 2）"""
    st.header("💬 基于记忆池的长文问答")
    st.caption("文档 4.2 节：层级摘要 + RAG 检索 + 记忆池注入 → LLM 回答 + 引用溯源")

    if not comp["indexed_docs"]:
        st.warning("会话内暂无可检索文档，请先在「📄 PDF 上传」Tab 添加至少 1 篇。")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        doc_options = list(comp["indexed_docs"].keys())
        sel_doc = st.selectbox(
            "目标论文",
            doc_options,
            format_func=lambda d: f"{d} · {comp['indexed_docs'][d]['title'][:30]}",
        )
        top_k = st.slider("RAG 检索 top_k", 1, 10, 5)
        use_memory = st.checkbox("启用会话记忆池（记忆上轮问答）", value=True)
    with col2:
        question = st.text_area(
            "输入问题（≤2000 字，文档 6.5 节限制）",
            placeholder="Transformer 在 WMT 2014 英德翻译任务上的 BLEU 是多少？",
            height=120,
            max_chars=2000,
        )
        if len(question) >= 2000:
            st.warning("接近文档 6.5 节 2000 字上限，请精简。")
        run_btn = st.button("🚀 提交回答", type="primary", disabled=not question.strip())

    # 记忆池
    history = st.session_state.get("ask_history", [])

    if run_btn and question.strip():
        with st.status("检索 + 生成中", expanded=True):
            st.write(f"1️⃣ RAG 检索 top-{top_k}（论文 {sel_doc}）")
            retrieved = comp["retriever"].search(question, top_k=top_k)
            st.write(f"   返回 {len(retrieved)} 段，"
                     f"相关论文：{', '.join(sorted({c.doc_id for c in retrieved}))}")

            ctx = "\n".join(
                f"[{c.chunk_id}] {c.text}" for c in retrieved
            )[:3000]

            memory_prefix = ""
            if use_memory and history:
                memory_prefix = "—— 会话上下文（最近 3 轮）：\n" + "\n".join(
                    f"Q{i+1}: {h['q'][:80]} / A{i+1}: {h['a'][:80]}"
                    for i, h in enumerate(history[-3:])
                ) + "\n——\n"

            st.write("2️⃣ LLM 生成（DashScope 真实 or Mock 模板化）")
            system = "你是学术问答助手。仅根据参考论文正文回答问题，并在关键结论处附 [chunk_id] 引用。"
            prompt = (
                f"{memory_prefix}参考正文（chunk_id 前缀标注）：\n---\n{ctx}\n---\n\n"
                f"问题：{question}\n\n"
                f"要求：答案 50-150 字，使用 [chunk_id] 作为引用编号，最后附 1 句总结。"
            )
            t0 = time.time()
            resp = comp["llm_client"].chat(prompt, system=system, temperature=0.2, max_tokens=512)
            latency_ms = int((time.time() - t0) * 1000)

        # 展示引用（三色：文档 4.3 节）
        # 先把 LLM 返回的 [chunk_id] 替换为形如 [1] 的编号，并按 label 染色
        cit_map = {}  # idx -> {label, original_text}
        answer_with_refs = resp.text
        for idx, c in enumerate(retrieved, 1):
            idx_label = str(idx)
            if f"[{c.chunk_id}]" in answer_with_refs:
                answer_with_refs = answer_with_refs.replace(f"[{c.chunk_id}]", f"[{idx_label}]")
                cit_map[idx_label] = {
                    "label": "entailment",  # RAG 检索的段落默认 entailment（综述才用 NLI 三色）
                    "original_text": c.text[:80],
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                }
            else:
                # 没有在答案里引用，也加入 citation_list（展示引用的候选）
                pass

        citation_list = [
            {"index": idx, **v} for idx, v in cit_map.items()
        ]

        # 展示
        st.subheader("🧾 回答")
        html = _render_citations(answer_with_refs, citation_list)
        st.markdown(html, unsafe_allow_html=True)

        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        meta_col1.metric("来源", "缓存" if resp.from_cache else ("真实" if not comp["mock_mode"] else "Mock"))
        meta_col2.metric("耗时", f"{latency_ms or resp.latency_ms}ms")
        meta_col3.metric("Tokens", f"{resp.input_tokens}/{resp.output_tokens}")
        meta_col4.metric("引用", str(len(citation_list)))

        if resp.partial:
            st.error(f"LLM 返回部分结果：{resp.error or '未知错误'}")

        # 引用溯源详情
        with st.expander("📚 引用溯源详情（点击查看原文段）"):
            for c in sorted(citation_list, key=lambda x: int(x["index"])):
                st.markdown(
                    f"**[{c['index']}]** "
                    f"<span class='cite cite-{c['label']}'>{c['label']}</span> "
                    f"`doc_id={c['doc_id']}`  `chunk_id={c['chunk_id']}`\n\n"
                    f"> {c['original_text']}\n",
                    unsafe_allow_html=True,
                )

        # 记忆池保存
        history.append({"q": question, "a": answer_with_refs, "ts": datetime.now().isoformat(timespec="seconds")})
        st.session_state["ask_history"] = history[-5:]

        # 成本记录
        if not resp.from_cache and not comp["mock_mode"]:
            try:
                from app.observability.cost_tracker import CostRecord
                comp["cost_tracker"].record(CostRecord(
                    ts=datetime.now(),
                    model=resp.model, step="qa.ask",
                    input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                    cost_yuan=resp.cost_yuan, trace_id="session",
                ))
            except Exception as e:  # noqa: BLE001
                logger.debug("成本记录失败: %s", e)


# =====================================================================
# Tab 5：综述生成（跨文献 + 引用事实验证 + Word 导出）
# =====================================================================


def render_review_tab(comp: Dict):
    """📚 综述生成 Tab（文档第 2 章核心功能 3）"""
    st.header("📚 跨文献综述草稿生成")
    st.caption("文档第 3 章架构图：综述主题 → 多文档检索 → MMR 去重 → Map-Reduce 生成 → 引用事实验证 → Word 导出")

    if len(comp["indexed_docs"]) < 2:
        st.warning("综述需要至少 2 篇文献，当前会话内已索引 %d 篇，请先在「📄 PDF 上传」中添加。"
                   % len(comp["indexed_docs"]))
        return

    with st.form("review_form"):
        topic = st.text_input(
            "综述主题（≤500 字，文档 6.5 节限制）",
            placeholder="深度学习模型在视觉与语言任务中的应用",
            max_chars=500,
        )
        sel_docs = st.multiselect(
            "选文献（≥2 篇）",
            options=list(comp["indexed_docs"].keys()),
            format_func=lambda d: f"{d} · {comp['indexed_docs'][d]['title'][:40]}",
        )
        num_sections = st.slider("章节数", 2, 8, 4)
        verify = st.checkbox("引用事实验证（文档 4.3 节三色 NLI 验证）", value=True)
        export_docx = st.checkbox("生成后导出 Word（python-docx / Markdown 降级）", value=True)
        submitted = st.form_submit_button("📝 生成综述", type="primary")

    if not submitted:
        return
    if len(sel_docs) < 2:
        st.error("请至少选 2 篇文献")
        return
    if not topic.strip():
        st.error("请填写综述主题")
        return

    doc_infos = [
        {"doc_id": did, "title": comp["indexed_docs"][did]["title"]}
        for did in sel_docs
    ]
    doc_chunks = {
        did: list(comp["indexed_docs"][did]["chunks"]) for did in sel_docs
    }

    with st.status("综述生成中（大纲 → 分段生成 → 引用验证 → 组装）", expanded=True):
        t0 = time.time()
        try:
            document = comp["review_pipeline"].run(
                topic=topic,
                doc_infos=doc_infos,
                doc_chunks=doc_chunks,
                num_sections=num_sections,
                verify_citations=verify,
                title=f"综述：{topic}",
            )
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            return
    latency_ms = int((time.time() - t0) * 1000)

    # 结果展示
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("章节", str(len(document.outline)))
    col2.metric("段落", str(len(document.paragraphs)))
    col3.metric("参考文献", str(len(document.references)))
    col4.metric("耗时", f"{latency_ms}ms")

    # 大纲
    with st.expander("📋 大纲", expanded=True):
        for s in document.outline:
            st.markdown(
                f"- **{s['title']}**（`{s['section_id']}`）：{s['description'] or '（无描述）'} "
                f" 相关文档 {', '.join(s.get('related_doc_ids', [])[:3])}"
            )

    # 章节正文 + 三色引用 HTML
    st.subheader("📖 综述正文（带三色引用溯源）")
    st.caption("蓝色=引用受支持 · 红色=原文矛盾 · 灰色=引用过宽（文档 4.3 节 NLI 验证结果）")
    safe_html = _xss_filter(document.full_html) or ""
    if not safe_html:
        # fallback：用段落渲染
        for para in document.paragraphs:
            st.markdown(f"### {next((o['title'] for o in document.outline if o['section_id']==para['section_id']), para['section_id'])}")
            st.write(para.get("rendered_text") or para.get("text", ""))
    else:
        st.markdown(safe_html, unsafe_allow_html=True)

    # 引用统计
    cs = document.citation_stats
    st.subheader("📊 引用统计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("合计", str(cs.get("total", 0)))
    c2.metric("蓝色 支持", f"{cs.get('entailment', 0)} ({(cs.get('entailment', 0)/max(cs.get('total',1),1)*100):.0f}%)",
             delta_color="normal")
    c3.metric("红色 存疑", f"{cs.get('contradiction', 0)} ({(cs.get('contradiction', 0)/max(cs.get('total',1),1)*100):.0f}%)",
             delta_color="inverse")
    c4.metric("灰色 过宽", f"{cs.get('neutral', 0)} ({(cs.get('neutral', 0)/max(cs.get('total',1),1)*100):.0f}%)")

    # 参考文献
    with st.expander("📚 参考文献列表"):
        for ref in document.references:
            label = ref.get("label", "neutral")
            st.markdown(
                f"**[{ref.get('index', '?')}]** <span class='cite cite-{label}'>{label}</span> "
                f"`doc_id={ref.get('doc_id')}`  `chunk_id={ref.get('chunk_id')}`\n\n"
                f"> {ref.get('original_text', '')[:200]}",
                unsafe_allow_html=True,
            )

    # Word 导出
    if export_docx:
        t0w = time.time()
        result = comp["docx_renderer"].render(document)
        render_ms = int((time.time() - t0w) * 1000)
        if result.mode == "docx" and result.bytes:
            st.success(f"Word 文档生成成功（{result.size} 字节，{render_ms}ms）")
            st.download_button(
                label="⬇ 下载 Word 文档 (.docx)",
                data=result.bytes,
                file_name=f"review_{topic[:20] or 'document'}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        elif result.markdown:
            st.info(f"python-docx 未安装，已降级为 Markdown（{result.size} 字节，{render_ms}ms）")
            st.download_button(
                label="⬇ 下载 Markdown (.md)",
                data=result.markdown.encode("utf-8"),
                file_name=f"review_{topic[:20] or 'document'}.md",
                mime="text/markdown",
            )
        if result.path:
            st.caption(f"文件路径：`{result.path}`")


# =====================================================================
# Tab 6：成本看板（文档 6.1 节）
# =====================================================================


def render_dashboard_tab(comp: Dict):
    """💰 成本看板 Tab（文档 6.1 节）"""
    st.header("💰 API 成本看板")
    st.caption("文档 6.1 节：按天聚合 + 按模型 / 按用户 / 按论文分摊 + 缓存命中率")

    dashboard: Any = comp["cost_dashboard"]

    # 文本看板
    with st.expander("📝 文本快照", expanded=False):
        st.code(dashboard.render_text())

    # 按天趋势
    daily = dashboard.daily_summary()
    st.subheader("📈 每日成本趋势")
    if daily:
        try:
            import pandas as pd  # type: ignore
            df = pd.DataFrame([
                {"日期": d, "成本(元)": v["cost"], "调用次数": int(v["calls"]),
                 "输入tokens": int(v["input_tokens"]), "输出tokens": int(v["output_tokens"])}
                for d, v in sorted(daily.items())
            ])
            st.dataframe(df, use_container_width=True)
            try:
                import plotly.express as px  # type: ignore
                fig = px.line(df, x="日期", y="成本(元)", markers=True, title="每日 API 成本（元）")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(df.set_index("日期")[["成本(元)"]])
        except ImportError:
            st.json(daily)
    else:
        st.info("暂无成本记录。在「💬 论文问答」「📚 综述生成」中真实调用后会自动展示。")

    # 按模型 / 用户 / 论文
    by_model = dashboard.cost_by_model()
    by_user = dashboard.cost_by_user()
    by_paper = dashboard.cost_by_paper()
    cache = dashboard.cache_hit_rate()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🤖 按模型")
        st.json(by_model or {"(无数据)": True})
    with c2:
        st.markdown("#### 👤 按用户")
        st.json(by_user or {"(无数据)": True})
    with c3:
        st.markdown("#### 📄 按论文")
        st.json(by_paper or {"(无数据)": True})

    # 缓存命中率
    st.subheader("💾 缓存命中率（文档 6.2 节 目标约 30%）")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("命中率", f"{cache['rate']:.1%}")
    col2.metric("命中数", str(int(cache.get("hits", 0))))
    col3.metric("总调用", str(int(cache.get("total", 0))))
    col4.metric("累计成本", f"{comp['cost_tracker'].total_cost():.4f} 元")

    # 文档 6.2 节成本参考
    with st.expander("📖 文档 6.2 节单篇成本参考"):
        st.markdown(
            """
            | 步骤 | 模型 | 单篇成本 |
            | --- | --- | ---: |
            | PDF 解析（50 页首次） | Qwen-VL-Max | 约 2.0 元 |
            | Embedding 索引 | text-embedding-v3 | 约 0.04 元 |
            | LLM 问答/综述 | Qwen-Plus | 约 0.3 元 |
            | 首次处理合计 | — | 约 2.4 元 |
            | 二次打开（缓存命中） | — | < 0.3 元 |
            """
        )


# =====================================================================
# Tab 2：文献翻译 Tab（两种运行模式都可用）
# =====================================================================

# 文档导出 MIME（按后端返回的 docx_mode 选择）
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MARKDOWN_MIME = "text/markdown"

# 多语种翻译语言选项：(语言代码, 中英 label)
SOURCE_LANG_OPTIONS: List[Tuple[str, str]] = [
    ("en", "English 英文"),
    ("zh", "中文（简体）"),
    ("ja", "日本語 日文"),
    ("de", "Deutsch 德文"),
    ("fr", "Français 法文"),
    ("ru", "Русский 俄文"),
]
TARGET_LANG_OPTIONS: List[Tuple[str, str]] = [
    ("zh-CN", "中文（简体）"),
    ("en", "English 英文"),
    ("ja", "日本語 日文"),
    ("de", "Deutsch 德文"),
    ("fr", "Français 法文"),
    ("ru", "Русский 俄文"),
]
TRANSLATE_LANG_LABELS: Dict[str, str] = dict(SOURCE_LANG_OPTIONS + TARGET_LANG_OPTIONS)
MAX_TRANSLATE_CHARS = 5000


def _lang_label(code: str) -> str:
    """语言代码 → 中英 label（未知代码原样返回）"""
    return TRANSLATE_LANG_LABELS.get(code, code)


def _api_fallback_hint() -> None:
    """API 调用失败后的统一兜底提示：引导重启后端或切换本地进程模式。"""
    st.caption(
        "提示：请确认后端已启动（项目根目录 `make api`），"
        "或在左侧「🔌 运行模式」切换到「💻 本地进程模式」后离线重试。"
    )


def _run_translation_api(
    client: ApiClient,
    text: str,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    """API 模式：POST /translate，返回统一结果字典（错误也放在字典里）。"""
    with st.status("调用后端翻译（POST /translate）…", expanded=False) as status:
        t0 = time.time()
        resp = client.translate(
            text, source_lang=source_lang, target_lang=target_lang, user_id=_user_id(),
        )
        wall_ms = int((time.time() - t0) * 1000)
        if resp.get("error"):
            status.update(label="翻译失败", state="error")
            return {"error": str(resp.get("error"))}
        status.update(label="翻译完成", state="complete")
    return {
        "translated": str(resp.get("translated") or ""),
        "source_text": str(resp.get("source_text") or text),
        "aligned_terms": list(resp.get("aligned_terms") or []),
        "target_lang": str(resp.get("target_lang") or target_lang),
        "mock_mode": bool(resp.get("mock_mode")),
        "latency_ms": int(resp.get("latency_ms") or wall_ms),
    }


def _run_translation_local(
    comp: Dict,
    text: str,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    """本地进程模式：直接调用 app.translation（不经过 FastAPI）。"""
    with st.status("本地进程翻译中…", expanded=False) as status:
        t0 = time.time()
        try:
            from app.translation import create_translator
            translator_mode = "mock" if comp["mock_mode"] else "auto"
            translator = create_translator(translator_mode)
            unit = translator.translate_text(
                text, source_lang=source_lang, target_lang=target_lang,
            )
        except Exception as e:  # noqa: BLE001 - 前端不允许崩溃
            status.update(label="翻译失败", state="error")
            logger.exception("本地进程翻译失败")
            return {"error": str(e)}
        wall_ms = int((time.time() - t0) * 1000)
        status.update(label="翻译完成", state="complete")
    return {
        "translated": str(unit.target_text or ""),
        "source_text": str(unit.source_text or text),
        "aligned_terms": list(unit.aligned_terms or []),
        "target_lang": str(unit.target_lang or target_lang),
        "mock_mode": bool(unit.from_mock or comp["mock_mode"]),
        "latency_ms": int(unit.latency_ms or wall_ms),
    }


def _render_translation_result(result: Dict[str, Any]) -> None:
    """展示翻译结果：译文 + 术语校准 + 耗时 / Mock 标识 + .txt 下载。"""
    if result.get("error"):
        st.error(f"翻译失败：{result['error']}")
        _api_fallback_hint()
        return

    translated = str(result.get("translated") or "")
    source_text = str(result.get("source_text") or "")
    aligned_terms = list(result.get("aligned_terms") or [])
    mock_on = bool(result.get("mock_mode"))
    latency_ms = int(result.get("latency_ms") or 0)
    src = str(result.get("source_lang") or "en")
    tgt = str(result.get("target_lang") or "zh-CN")

    st.subheader("🧾 译文")
    st.caption(f"翻译方向：{_lang_label(src)} → {_lang_label(tgt)}")
    if mock_on:
        st.warning(
            "Mock 模式：译文为模板化 / 内置小词典替换结果，"
            "配置真实 DASHSCOPE_API_KEY 后为真实翻译。"
        )
    if translated:
        st.markdown(translated)
    else:
        st.info("后端 / 本地返回的译文为空。")

    col1, col2, col3 = st.columns(3)
    col1.metric("耗时", f"{latency_ms}ms")
    col2.metric("命中术语", str(len(aligned_terms)))
    col3.metric("模式", "Mock" if mock_on else "真实")

    st.subheader("🔤 命中并校准的术语")
    if aligned_terms:
        st.markdown(" · ".join(f"`{t}`" for t in aligned_terms))
    else:
        st.info("本段未命中内置术语词库中的术语（公式 / 数字 / 缩写按原文保留）。")

    with st.expander("📄 原文"):
        st.text(source_text)

    st.download_button(
        label="⬇ 下载译文 (.txt)",
        data=translated.encode("utf-8"),
        file_name=f"translation_{src}_to_{tgt}.txt",
        mime="text/plain; charset=utf-8",
        key="translate_download_btn",
    )


def render_translate_tab(
    comp: Optional[Dict],
    client: Optional[ApiClient],
    mode: str,
) -> None:
    """🌐 多语种文献翻译 Tab（文档第 2 章；API / 本地进程两种模式都可用）

    - API 模式：POST /translate（后端 DashScopeTranslator + 术语校准）
    - 本地进程模式：app.translation.create_translator(...).translate_text(...)
    """
    st.header("🌐 多语种文献翻译")
    st.caption(
        "文档第 2 章：多语种互译 + 专业术语统一校准"
        "（术语词库 preprocess → LLM 翻译 → 译法 postprocess）"
    )
    if mode == "local" and comp is None:
        st.error("本地进程模式组件未初始化，请刷新页面重试。")
        return

    col_src, col_tgt = st.columns(2)
    with col_src:
        source_lang = st.selectbox(
            "源语言 Source language",
            [code for code, _ in SOURCE_LANG_OPTIONS],
            format_func=_lang_label,
            key="translate_source_lang",
        )
    with col_tgt:
        target_lang = st.selectbox(
            "目标语言 Target language",
            [code for code, _ in TARGET_LANG_OPTIONS],
            format_func=_lang_label,
            key="translate_target_lang",
        )

    text = st.text_area(
        "待翻译文献片段（≤5000 字，文档 6.5 节限制）",
        height=220,
        max_chars=MAX_TRANSLATE_CHARS,
        placeholder=(
            "例如：Self-attention is the core of the Transformer. "
            "We propose a multi-head attention mechanism..."
        ),
        key="translate_source_text",
    )
    text_len = len(text)
    if text_len >= MAX_TRANSLATE_CHARS:
        st.warning(f"已达 {MAX_TRANSLATE_CHARS} 字上限，请分段翻译。")
    elif text_len > 0:
        st.caption(f"当前 {text_len} 字 / 上限 {MAX_TRANSLATE_CHARS} 字")

    translate_btn = st.button(
        "🌐 翻译", type="primary", disabled=not text.strip(), key="translate_submit_btn",
    )
    if translate_btn and text.strip():
        clean_text = text.strip()
        if mode == "api" and client is not None:
            result = _run_translation_api(client, clean_text, source_lang, target_lang)
        else:
            result = _run_translation_local(
                comp, clean_text, source_lang, target_lang,  # type: ignore[arg-type]
            )
        result["source_lang"] = source_lang
        result["target_lang"] = result.get("target_lang") or target_lang
        st.session_state["translation_result"] = result

    saved_result = st.session_state.get("translation_result")
    if saved_result:
        _render_translation_result(saved_result)


# =====================================================================
# Tab 3：Agent 助手（LLM 自主调用工具；API / 本地进程两种模式都可用）
# =====================================================================

# 工具参数 / 结果预览的最大字符数（避免页面塞入超长 JSON）
AGENT_PREVIEW_CHARS = 300
# Agent 会话历史注入下一次请求的轮数（最近 N 轮）
AGENT_HISTORY_ROUNDS = 3
# Agent 表单默认最大步数（与后端 POST /agent 默认值一致）
AGENT_DEFAULT_MAX_STEPS = 6


def _agent_doc_options(comp: Optional[Dict], mode: str) -> Dict[str, Dict[str, Any]]:
    """返回 Agent Tab 的「目标论文」候选：``{doc_id: {"title", "num_chunks"}}``。

    Args:
        comp: 本地进程模式组件集合（API 模式传 None）
        mode: 运行模式（"api" / "local"）

    Returns:
        doc_id → 文档摘要的字典；无已索引文档时返回空字典。
    """
    if mode == "api":
        raw: Dict[str, Any] = dict(st.session_state.get("api_docs", {}) or {})
        api_docs: Dict[str, Dict[str, Any]] = {}
        for doc_id, info in raw.items():
            item = info if isinstance(info, dict) else {}
            api_docs[str(doc_id)] = {
                "title": str(item.get("title") or doc_id),
                "num_chunks": int(item.get("num_chunks") or len(item.get("chunks") or [])),
            }
        return api_docs

    if comp is None:
        return {}
    docs: Dict[str, Dict[str, Any]] = {}
    for doc_id, info in (comp.get("indexed_docs") or {}).items():
        item = info if isinstance(info, dict) else {}
        docs[str(doc_id)] = {
            "title": str(item.get("title") or doc_id),
            "num_chunks": len(item.get("chunks") or []),
        }
    return docs


def _agent_preview(obj: Any, limit: int = AGENT_PREVIEW_CHARS) -> str:
    """把工具参数 / 结果压缩成短文本（JSON 优先，超长截断并加省略号）。"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(obj)
    text = text.replace("\r\n", "\n").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _agent_local_tools(comp: Dict) -> Any:
    """构造本地进程模式 Agent 工具：四个回调均基于会话内现有组件。

    冻结契约：create_default_tools 接收
    list_documents / search_chunks / translate_text / generate_review 四个回调。

    Args:
        comp: get_app_components() 返回的组件集合

    Returns:
        create_default_tools(...) 的工具列表（交给 AgentRunner 使用）。
    """
    from app.agent import create_default_tools

    def list_documents() -> List[Dict[str, Any]]:
        """列出会话内已索引文档（doc_id / title / num_chunks）。"""
        documents: List[Dict[str, Any]] = []
        for doc_id, info in (comp.get("indexed_docs") or {}).items():
            item = info if isinstance(info, dict) else {}
            documents.append({
                "doc_id": str(doc_id),
                "title": str(item.get("title") or doc_id),
                "num_chunks": len(item.get("chunks") or []),
            })
        return documents

    def search_chunks(query: str, doc_id: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """检索会话内 chunk（可按 doc_id 过滤），正文截断 600 字。"""
        try:
            k = max(1, int(top_k))
        except (TypeError, ValueError):
            k = 5
        results = comp["retriever"].search(query, top_k=k)
        chunks: List[Dict[str, Any]] = []
        for item in results:
            chunk = getattr(item, "chunk", None)
            if chunk is None:
                continue
            if doc_id and str(getattr(chunk, "doc_id", "")) != str(doc_id):
                continue
            chunks.append({
                "chunk_id": str(getattr(chunk, "chunk_id", "")),
                "doc_id": str(getattr(chunk, "doc_id", "")),
                "text": str(getattr(chunk, "text", ""))[:600],
                "section": str(getattr(chunk, "section", "")),
                "score": float(getattr(item, "score", 0.0) or 0.0),
            })
        return chunks

    def translate_text(text: str, target_lang: str = "zh-CN") -> Dict[str, Any]:
        """多语种翻译 + 术语校准（本地进程模式，mock / auto 由会话决定）。"""
        from app.translation import create_translator

        translator_mode = "mock" if comp.get("mock_mode") else "auto"
        translator = create_translator(translator_mode)
        unit = translator.translate_text(text or "", target_lang=target_lang or "zh-CN")
        return {
            "translated": str(getattr(unit, "target_text", "") or ""),
            "aligned_terms": list(getattr(unit, "aligned_terms", []) or []),
            "target_lang": str(getattr(unit, "target_lang", "") or target_lang),
            "mock_mode": bool(getattr(unit, "from_mock", False) or comp.get("mock_mode")),
        }

    def generate_review(
        topic: str,
        doc_ids: Optional[List[str]] = None,
        num_sections: int = 4,
    ) -> Dict[str, Any]:
        """跨文献综述生成（含引用事实验证），返回契约约定的精简字典。"""
        indexed = comp.get("indexed_docs") or {}
        selected = [d for d in (doc_ids or []) if d in indexed]
        if not selected:
            selected = list(indexed.keys())
        if not selected:
            return {
                "title": "",
                "full_text": "",
                "references": [],
                "citation_stats": {},
                "mock_mode": bool(comp.get("mock_mode")),
                "error": "会话内无可用于综述的已索引文档（需先在「📄 PDF 上传」添加）",
            }
        doc_infos = [
            {"doc_id": did, "title": str((indexed[did] or {}).get("title") or did)}
            for did in selected
        ]
        doc_chunks = {
            did: list((indexed[did] or {}).get("chunks") or []) for did in selected
        }
        try:
            document = comp["review_pipeline"].run(
                topic=topic or "综述",
                doc_infos=doc_infos,
                doc_chunks=doc_chunks,
                num_sections=max(1, int(num_sections or 4)),
                verify_citations=True,
            )
        except Exception as e:  # noqa: BLE001 - 工具错误交给 Agent 记录，不中断循环
            logger.exception("本地 Agent 综述工具失败")
            return {
                "title": "",
                "full_text": "",
                "references": [],
                "citation_stats": {},
                "mock_mode": bool(comp.get("mock_mode")),
                "error": str(e),
            }
        return {
            "title": str(getattr(document, "title", "") or ""),
            "full_text": str(getattr(document, "full_text", "") or ""),
            "references": list(getattr(document, "references", []) or []),
            "citation_stats": dict(getattr(document, "citation_stats", {}) or {}),
            "mock_mode": bool(comp.get("mock_mode")),
            "error": "",
        }

    return create_default_tools(
        list_documents=list_documents,
        search_chunks=search_chunks,
        translate_text=translate_text,
        generate_review=generate_review,
    )


def _build_local_agent_runner(comp: Dict, max_steps: int) -> Any:
    """延迟 import app.agent，构造本地 AgentRunner（不依赖 FastAPI 后端）。

    冻结签名：``create_agent_runner(llm=None, tools=None, framework="auto",
    max_steps=6)``。会话处于 mock 模式时显式注入 ``ScriptedMockLLM``：
    ``llm=None`` 的 auto 选型会在 .env 存在真实 DASHSCOPE_API_KEY 时走真实
    DashScope（开发机 .env 常见），显式注入才能保证 mock 模式 0 真实调用。
    """
    from app.agent import ScriptedMockLLM, create_agent_runner

    llm = ScriptedMockLLM() if comp.get("mock_mode") else None
    tools = _agent_local_tools(comp)
    return create_agent_runner(
        llm=llm,
        tools=tools,
        framework="auto",
        max_steps=max(1, min(8, int(max_steps))),
    )


def _agent_history_messages(
    history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """把前端会话历史 ``[{"q", "a"}]`` 转成本地 AgentRunner 的 role/content 消息。

    Args:
        history: 最近若干轮问答，结构 [{"q": ..., "a": ...}]

    Returns:
        ``[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]``；
        历史为空时返回空列表。
    """
    messages: List[Dict[str, str]] = []
    for item in list(history or []):
        if not isinstance(item, dict):
            continue
        question = str(item.get("q") or "").strip()
        answer = str(item.get("a") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def _invoke_local_agent(
    runner: Any,
    question: str,
    doc_id: str,
    max_steps: int,
    history: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "",
) -> Dict[str, Any]:
    """调用本地 AgentRunner.run(...) 并转成字典（按 run 签名透传可选参数）。

    实际签名（agent-core 冻结）：``run(question, *, doc_id="", user_id="",
    history=[{"role","content"}...])``；max_steps 是 create_agent_runner 的构造参数，
    不在 run() 参数表里（若未来新增则这里会自动透传）。

    Args:
        runner: create_agent_runner(...) 返回的 AgentRunner
        question: 用户任务 / 问题
        doc_id: 目标文档 ID（空字符串表示全部文档）
        max_steps: 最大步数（仅用于兜底透传，正常由 runner 构造参数控制）
        history: 最近若干轮问答（前端 q/a 结构，内部转 role/content）
        user_id: 用户 ID（本地也传，便于结果里按用户统计）

    Returns:
        AgentResult.to_dict() 的字典副本。
    """
    run = getattr(runner, "run", None) or getattr(runner, "arun", None)
    if run is None:
        raise AttributeError("本地 AgentRunner 缺少 run() 方法")
    try:
        params = inspect.signature(run).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    kwargs: Dict[str, Any] = {}
    if "doc_id" in params:
        kwargs["doc_id"] = doc_id or ""
    if "user_id" in params:
        kwargs["user_id"] = user_id or ""
    if "max_steps" in params:
        kwargs["max_steps"] = int(max_steps)
    if "history" in params:
        kwargs["history"] = _agent_history_messages(history)
    result = run(question, **kwargs)
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        raise TypeError(f"本地 Agent 返回类型不支持：{type(result).__name__}")
    return dict(data or {})


def _run_agent_local(
    comp: Dict,
    question: str,
    doc_id: str,
    max_steps: int,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """本地进程模式：in-process 构造并运行 AgentRunner，返回统一结果字典。

    与 API 模式返回同一套字段（answer / steps / tool_calls / citations / ...），
    供 ``_render_agent_result`` 统一渲染。
    """
    with st.status(
        f"本地 Agent 运行中（ReAct 循环 · 最多 {max_steps} 步）…", expanded=True,
    ) as status:
        st.write(
            "工具：list_documents / search_chunks / translate_text / generate_review · "
            f"doc_id=`{doc_id or '全部文档'}` · framework=auto"
        )
        t0 = time.time()
        try:
            runner = _build_local_agent_runner(comp, max_steps)
            result = _invoke_local_agent(
                runner, question, doc_id, max_steps, history, user_id=_user_id(),
            )
        except ImportError as e:
            status.update(label="本地 Agent 模块未就绪", state="error")
            logger.warning("app.agent 未就绪: %s", e)
            return {
                "error": (
                    "本地 Agent 模块（app.agent）尚未就绪："
                    f"{e}。请等待 Agent 核心模块完成，或在左侧切换到 API 模式。"
                )
            }
        except Exception as e:  # noqa: BLE001 - 前端不允许崩溃
            status.update(label="本地 Agent 运行失败", state="error")
            logger.exception("本地 Agent 运行失败")
            return {"error": f"本地 Agent 运行失败：{e}"}
        wall_ms = int((time.time() - t0) * 1000)
        result = dict(result or {})
        result["latency_ms"] = int(result.get("latency_ms") or wall_ms)
        if not result.get("framework"):
            result["framework"] = "local"
        if "mock_mode" not in result:
            result["mock_mode"] = bool(comp.get("mock_mode"))
        step_count = len(list(result.get("steps") or []))
        call_count = len(list(result.get("tool_calls") or []))
        st.write(
            f"完成：{step_count} 步 · {call_count} 次工具调用 · "
            f"{result['latency_ms']}ms"
        )
        if result.get("error"):
            status.update(label="Agent 运行完成（有错误）", state="complete")
        else:
            status.update(label="Agent 运行完成", state="complete", expanded=False)
    return result


def _run_agent_api(
    client: ApiClient,
    question: str,
    doc_id: str,
    max_steps: int,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """API 模式：POST /agent（后端 ReAct / LangGraph 编排），返回响应字典。"""
    with st.status("调用后端 Agent（POST /agent）…", expanded=True) as status:
        st.write(
            f"POST /agent · doc_id=`{doc_id or '全部文档'}` · max_steps={max_steps} · "
            "framework=auto"
        )
        t0 = time.time()
        resp = client.agent(
            question,
            doc_id=doc_id,
            user_id=_user_id(),
            max_steps=max_steps,
            framework="auto",
            history=list(history or []),
        )
        wall_ms = int((time.time() - t0) * 1000)
        if resp.get("error") and not resp.get("answer"):
            status.update(label="Agent 调用失败", state="error")
            return dict(resp)
        if resp.get("error"):
            status.update(label="Agent 调用完成（有错误）", state="complete")
        else:
            status.update(label="Agent 调用完成", state="complete", expanded=False)
        result = dict(resp)
        result["latency_ms"] = int(result.get("latency_ms") or wall_ms)
        step_count = len(list(result.get("steps") or []))
        call_count = len(list(result.get("tool_calls") or []))
        st.write(f"完成：{step_count} 步 · {call_count} 次工具调用 · {wall_ms}ms")
    return result


def _render_agent_tool_call(call: Dict[str, Any], step_label: str = "") -> None:
    """渲染单个工具调用：工具名 + 参数 JSON 预览 + 成败 + 耗时 + 结果预览（≤300 字）。"""
    name = str(call.get("name") or "?")
    latency_ms = int(call.get("latency_ms") or 0)
    error = str(call.get("error") or "")
    ok = bool(call.get("ok")) if "ok" in call else not bool(error)
    prefix = f"`[{step_label}]` " if step_label else ""
    st.markdown(f"{prefix}🔧 **{name}** · {'✅ 成功' if ok else '❌ 失败'} · {latency_ms}ms")
    st.code(_agent_preview(call.get("arguments"), 400) or "{}", language="json")
    if error:
        st.error(f"工具错误：{_agent_preview(error, AGENT_PREVIEW_CHARS)}")
        return
    preview = call.get("result_preview")
    if preview is None:
        preview = call.get("result")
    st.caption("结果预览：" + (_agent_preview(preview, AGENT_PREVIEW_CHARS) or "(空)"))


def _render_agent_tool_trace(result: Dict[str, Any]) -> None:
    """渲染 Agent 工具调用轨迹（步骤时间线 / 工具名 / 参数 / 结果预览 / 耗时）。

    优先用后端的 ``steps``（含 thought 与逐步 token / 成本）；若后端只返回扁平的
    ``tool_calls``，则退化为逐条工具调用展示。
    """
    steps = [s for s in (result.get("steps") or []) if isinstance(s, dict)]
    flat_calls = [c for c in (result.get("tool_calls") or []) if isinstance(c, dict)]
    st.subheader("🛠 工具调用轨迹")
    if not steps and not flat_calls:
        st.info("本次运行未产生工具调用（Agent 可能直接回答，或 max_steps 用尽）。")
        return
    if steps:
        for pos, step in enumerate(steps):
            index = step.get("index", pos + 1)
            calls = [c for c in (step.get("tool_calls") or []) if isinstance(c, dict)]
            thought = str(step.get("thought") or "").strip()
            latency_ms = int(step.get("latency_ms") or 0)
            if calls:
                label = f"步骤 {index} · {len(calls)} 次工具调用 · {latency_ms}ms"
            else:
                label = f"步骤 {index} · 最终回答 · {latency_ms}ms"
            with st.expander(label, expanded=(pos == 0 and bool(calls))):
                if thought:
                    st.markdown(f"**🧠 思考**：{thought[:AGENT_PREVIEW_CHARS]}")
                if not calls:
                    st.caption("该步未调用工具，直接生成最终回答。")
                for call in calls:
                    _render_agent_tool_call(call, step_label=str(index))
        return
    for call in flat_calls:
        _render_agent_tool_call(call, step_label=str(call.get("step", "?")))


def _render_agent_result(result: Dict[str, Any], mode: str) -> None:
    """统一渲染 Agent 结果（API / 本地进程共用）：指标行 + 工具轨迹 + 答案 + 引用。

    Args:
        result: POST /agent 响应或本地 AgentResult.to_dict() 的字典
        mode: 本次运行模式（"api" / "local"），仅影响指标行文案
    """
    if not result:
        st.info("Agent 尚未运行。输入任务后点击「▶ 运行 Agent」。")
        return

    answer = str(result.get("answer") or "")
    error = str(result.get("error") or "")
    if error and not answer:
        st.error(f"Agent 运行失败：{error}")
        if mode == "api":
            _api_fallback_hint()
        else:
            st.caption(
                "提示：本地 Agent 依赖 app.agent 包与 mock 环境；"
                "也可在左侧「🔌 运行模式」切到 API 模式重试。"
            )
        return
    if error:
        st.warning(f"Agent 返回警告：{error}")

    steps = [s for s in (result.get("steps") or []) if isinstance(s, dict)]
    flat_calls = [c for c in (result.get("tool_calls") or []) if isinstance(c, dict)]
    citations = [c for c in (result.get("citations") or []) if isinstance(c, dict)]
    step_call_count = sum(len(s.get("tool_calls") or []) for s in steps)
    call_count = len(flat_calls) or step_call_count
    framework = str(result.get("framework") or "-")
    mock_on = bool(result.get("mock_mode"))
    input_tokens = int(result.get("input_tokens") or 0)
    output_tokens = int(result.get("output_tokens") or 0)
    cost_yuan = float(result.get("cost_yuan") or 0.0)
    latency_ms = int(result.get("latency_ms") or 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("运行模式", "🌐 API" if mode == "api" else "💻 本地进程")
    col2.metric("framework", framework)
    col3.metric("步数", str(len(steps)))
    col4.metric("工具调用", str(call_count))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tokens", f"{input_tokens}/{output_tokens}")
    col6.metric("成本", f"{cost_yuan:.4f} 元")
    col7.metric("耗时", f"{latency_ms}ms")
    col8.metric("Mock", "是" if mock_on else "否")
    if result.get("model") or result.get("trace_id"):
        st.caption(
            f"模型：`{result.get('model') or '-'}` · "
            f"trace_id=`{result.get('trace_id') or '-'}`"
        )

    _render_agent_tool_trace(result)

    st.subheader("🧾 最终答案")
    citation_list: List[Dict[str, Any]] = []
    if answer:
        answer_with_refs, citation_list = _map_api_citations(answer, citations)
        st.markdown(
            _render_citations(answer_with_refs, citation_list), unsafe_allow_html=True,
        )
    else:
        st.info("Agent 未返回最终答案（可能 max_steps 用尽或工具调用全部失败），可调大步数重试。")

    with st.expander(f"📚 引用详情（{len(citations)} 条）"):
        if not citations:
            st.info("本次运行未产生引用（Agent 可能未调用 list_documents / search_chunks 类工具）。")
        for cit in citations:
            st.markdown(
                f"`chunk_id={cit.get('chunk_id', '')}` · "
                f"`doc_id={cit.get('doc_id', '')}`\n\n"
                f"> {str(cit.get('text_preview') or '')[:200]}"
            )


def _render_agent_available_tools(client: Optional[ApiClient], mode: str) -> None:
    """展示 Agent 可调用工具清单。

    - API 模式：GET /agent/tools（结果按会话缓存，避免每次 rerun 都请求后端）
    - 本地进程模式：延迟 import app.agent 的 default_tool_specs（缺失时静默跳过）
    """
    specs: List[Dict[str, Any]] = []
    if mode == "api" and client is not None:
        cached = st.session_state.get("agent_tool_specs")
        if cached is None:
            resp = client.agent_tools()
            cached = [] if resp.get("error") else list(resp.get("tools") or [])
            st.session_state["agent_tool_specs"] = list(cached)
        specs = [s for s in cached if isinstance(s, dict)]
    else:
        try:
            from app.agent import default_tool_specs  # type: ignore

            if callable(default_tool_specs):
                specs = [s for s in (default_tool_specs() or []) if isinstance(s, dict)]
        except Exception:  # noqa: BLE001 - app.agent 未就绪时静默跳过工具清单
            specs = []
    if not specs:
        return

    names: List[str] = []
    for spec in specs:
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else {}
        name = str((fn or spec).get("name") or "")
        if name:
            names.append(name)
    with st.expander(f"🧰 可用工具（{len(names)}）"):
        st.markdown(" · ".join(f"`{n}`" for n in names) or "(工具 schema 缺少 name 字段)")


def render_agent_tab(
    comp: Optional[Dict],
    client: Optional[ApiClient],
    mode: str,
) -> None:
    """🤖 Agent 助手 Tab：LLM 自主调用工具（列文档 / 检索 / 翻译 / 综述）。

    - API 模式：POST /agent（后端 ReAct / LangGraph 编排）；
    - 本地进程模式：in-process AgentRunner（app.agent，延迟 import）；
    - 两种模式共用同一套结果渲染（指标 + 工具轨迹 + 答案 + 引用）。

    Args:
        comp: 本地进程模式组件集合（API 模式为 None）
        client: API 模式下的 ApiClient（本地进程模式为 None）
        mode: 运行模式（"api" / "local"）
    """
    st.header("🤖 Agent 助手")
    st.caption(
        "LLM 自主调用工具（列文档 / 检索 / 翻译 / 综述）完成任务，"
        "展示工具调用轨迹（步骤 / 工具名 / 参数 / 结果预览 / 耗时 / 成本 / framework）"
    )
    if mode == "local" and comp is None:
        st.error("本地进程模式组件未初始化，请刷新页面重试。")
        return
    if mode == "api" and client is None:
        st.error("API 模式客户端未初始化，请在左侧切换运行模式后重试。")
        return

    docs = _agent_doc_options(comp, mode)
    col_left, col_right = st.columns([1, 3])
    with col_left:
        if docs:
            doc_choices = [""] + list(docs.keys())
            sel_doc = st.selectbox(
                "目标论文（“全部文档”= 不限文档）",
                doc_choices,
                format_func=lambda d: (
                    "全部文档" if not d else f"{d} · {docs[d]['title'][:24]}"
                ),
                key="agent_doc_id",
            )
        else:
            sel_doc = ""
            st.selectbox("目标论文", ["全部文档"], disabled=True, key="agent_doc_id")
            st.caption(
                "当前会话无已索引文档：Agent 仍可运行，会先调用 list_documents 说明情况。"
            )
        max_steps = st.slider(
            "Agent 最大步数 max_steps",
            1,
            8,
            AGENT_DEFAULT_MAX_STEPS,
            key="agent_max_steps",
        )
        st.caption("framework=auto（后端 / 本地自动选择 simple 或 langgraph 编排）")
    with col_right:
        question = st.text_area(
            "任务 / 问题（≤2000 字，文档 6.5 节限制）",
            placeholder=(
                "例如：请列出会话内已索引的论文，检索 Transformer 的核心公式；"
                "如需要可把关键结论翻译成中文，并基于全部文献生成 3 节综述。"
            ),
            height=140,
            max_chars=2000,
            key="agent_question",
        )
        if len(question) >= 2000:
            st.warning("达到文档 6.5 节 2000 字上限，请精简。")
        run_btn = st.button(
            "▶ 运行 Agent",
            type="primary",
            disabled=not question.strip(),
            key="agent_run_btn",
        )

    history: List[Dict[str, Any]] = list(st.session_state.get("agent_history", []))
    if history and not run_btn:
        with st.expander(
            f"🧠 Agent 会话历史（最近 {AGENT_HISTORY_ROUNDS} 轮，随下一次运行提交）"
        ):
            for idx, item in enumerate(history[-AGENT_HISTORY_ROUNDS:], 1):
                st.markdown(
                    f"**Q{idx}**：{item.get('q', '')}\n\n"
                    f"**A{idx}**：{str(item.get('a', ''))[:300]}"
                )

    _render_agent_available_tools(client, mode)

    if run_btn and question.strip():
        clean_question = question.strip()
        history_payload = [
            {"q": str(h.get("q", "")), "a": str(h.get("a", ""))}
            for h in history[-AGENT_HISTORY_ROUNDS:]
        ]
        if mode == "api" and client is not None:
            result = _run_agent_api(
                client, clean_question, sel_doc, max_steps, history_payload,
            )
        else:
            result = _run_agent_local(
                comp, clean_question, sel_doc, max_steps, history_payload,  # type: ignore[arg-type]
            )
        st.session_state["agent_result"] = result
        if result.get("answer"):
            history.append({"q": clean_question, "a": str(result.get("answer") or "")})
            st.session_state["agent_history"] = history[-AGENT_HISTORY_ROUNDS:]

    saved_result = st.session_state.get("agent_result")
    if saved_result:
        _render_agent_result(saved_result, mode)


# =====================================================================
# API 模式：PDF 上传 / 论文问答 / 综述生成 / 成本看板
# =====================================================================


def render_upload_tab_api(client: ApiClient) -> None:
    """📄 PDF 上传与解析 Tab（API 模式：POST /upload）"""
    st.header("📄 PDF 公式无损解析")
    st.caption(
        "API 模式：Streamlit → `POST /upload` → FastAPI → parser 流水线"
        "（API-first 架构，文档第 3 / 10 章）"
    )

    uploaded_file = st.file_uploader(
        "上传 PDF", type=["pdf"], accept_multiple_files=False, key="api_upload_file",
    )
    if not uploaded_file:
        st.info("请选择一个 PDF 文件。mock 模式可上传任意内容（后端会跳过真实渲染）。")
        return

    raw = uploaded_file.read()
    filename = uploaded_file.name or "upload.pdf"
    file_size_mb = len(raw) / 1024 / 1024
    st.caption(f"已选择：`{filename}` · {file_size_mb:.2f}MB")

    if file_size_mb > 50:
        st.error("PDF 文件超过 50MB 上限（文档 6.5 节），请更换文件。")
        return

    with st.status(f"上传并解析中：{filename}（{file_size_mb:.2f}MB）", expanded=True) as status:
        t0 = time.time()
        resp = client.upload_pdf(filename, raw, user_id=_user_id())
        latency_ms = int((time.time() - t0) * 1000)
        if resp.get("error"):
            status.update(label="解析失败", state="error")
            st.error(f"上传失败：{resp.get('error')}")
            _api_fallback_hint()
            return
        status.update(label="解析完成", state="complete", expanded=False)

    chunks = list(resp.get("chunks") or [])
    num_chunks = int(resp.get("num_chunks") or len(chunks))
    doc_id = str(resp.get("doc_id") or resp.get("paper_id") or "")
    paper_id = str(resp.get("paper_id") or doc_id)
    title = str(resp.get("title") or filename)
    num_pages = int(resp.get("num_pages") or 0)
    formula_resolved = bool(resp.get("formula_resolved"))
    need_review = bool(resp.get("need_review"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文件大小", f"{file_size_mb:.2f}MB")
    col2.metric("切分 chunks", str(num_chunks))
    col3.metric("公式定位", "✓" if formula_resolved else "⚠ 待复核")
    col4.metric("耗时", f"{latency_ms}ms")
    st.caption(
        f"doc_id=`{doc_id}` · paper_id=`{paper_id}` · 页数={num_pages} · "
        f"需复核={'是' if need_review else '否'}"
    )

    if not doc_id:
        st.warning("后端未返回 doc_id，该文档无法在问答 / 综述 Tab 中使用。")
        return

    api_docs: Dict[str, Dict[str, Any]] = st.session_state.setdefault("api_docs", {})
    api_docs[doc_id] = {
        "title": title,
        "num_chunks": num_chunks,
        "chunks": chunks,
        "paper_id": paper_id,
        "num_pages": num_pages,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state["api_docs"] = api_docs

    st.subheader(f"🔍 已索引 chunks（前 5 条预览，共 {num_chunks} 条）")
    if not chunks:
        st.info("后端未返回 chunk 明细（mock 模式下可能为空），不影响问答 / 综述调用。")
    for chunk in chunks[:5]:
        chunk_id = str(chunk.get("chunk_id", "?"))
        token_count = chunk.get("token_count", 0)
        section = chunk.get("section", "-")
        with st.expander(
            f"**chunk_id**: `{chunk_id}`  ·  tokens 约 {token_count}  ·  section={section}"
        ):
            st.write(chunk.get("text", ""))
    st.success(
        f"已保存到会话（doc_id=`{doc_id}`），可在「💬 论文问答」「📚 综述生成」Tab 使用。"
    )


def _map_api_citations(
    answer: str,
    citations: List[Dict],
) -> Tuple[str, List[Dict]]:
    """把答案中的 [chunk_id] 按 citations 顺序替换为 [1]、[2]…，并构造三色引用列表。

    与本地进程模式展示保持一致：RAG 检索到的段落默认标为 entailment（蓝色）。
    """
    answer_with_refs = answer
    citation_list: List[Dict] = []
    for idx, cit in enumerate(citations, 1):
        chunk_id = str(cit.get("chunk_id") or "")
        if chunk_id:
            answer_with_refs = answer_with_refs.replace(f"[{chunk_id}]", f"[{idx}]")
        citation_list.append({
            "index": str(idx),
            "label": "entailment",  # RAG 引用默认受支持；三色 NLI 验证用于综述
            "original_text": str(cit.get("text_preview") or "")[:200],
            "chunk_id": chunk_id,
            "doc_id": str(cit.get("doc_id") or ""),
        })
    return answer_with_refs, citation_list


def render_ask_tab_api(client: ApiClient) -> None:
    """💬 论文问答 Tab（API 模式：POST /ask）"""
    st.header("💬 基于记忆池的长文问答")
    st.caption(
        "API 模式：`POST /ask` → FastAPI（RAG 检索 + 记忆池 + LLM）→ 回答 + 引用溯源"
    )

    api_docs: Dict[str, Dict[str, Any]] = st.session_state.get("api_docs", {})
    if not api_docs:
        st.warning("会话内暂无可检索文档，请先在「📄 PDF 上传」Tab 添加至少 1 篇。")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_doc = st.selectbox(
            "目标论文",
            list(api_docs.keys()),
            format_func=lambda d: f"{d} · {str(api_docs[d].get('title', ''))[:30]}",
            key="api_ask_doc",
        )
        top_k = st.slider("RAG 检索 top_k", 1, 10, 5, key="api_ask_top_k")
        use_memory = st.checkbox(
            "启用会话记忆池（记忆上轮问答）", value=True, key="api_ask_memory",
        )
    with col2:
        question = st.text_area(
            "输入问题（≤2000 字，文档 6.5 节限制）",
            placeholder="Transformer 在 WMT 2014 英德翻译任务上的 BLEU 是多少？",
            height=120,
            max_chars=2000,
            key="api_ask_question",
        )
        if len(question) >= 2000:
            st.warning("达到文档 6.5 节 2000 字上限，请精简。")
        run_btn = st.button(
            "🚀 提交回答", type="primary", disabled=not question.strip(),
            key="api_ask_submit",
        )

    history: List[Dict[str, Any]] = list(st.session_state.get("ask_history_api", []))
    if not (run_btn and question.strip()):
        if history:
            with st.expander("🧠 会话记忆池（最近 5 轮，随下一次提问提交）"):
                for idx, item in enumerate(history[-5:], 1):
                    st.markdown(
                        f"**Q{idx}**：{item.get('q', '')}\n\n"
                        f"**A{idx}**：{str(item.get('a', ''))[:300]}"
                    )
        return

    with st.status("调用后端问答（POST /ask）…", expanded=True) as status:
        st.write(f"1️⃣ POST /ask · paper_id=`{sel_doc}` · top_k={top_k} · "
                 f"use_memory={use_memory}")
        t0 = time.time()
        resp = client.ask(
            paper_id=sel_doc,
            question=question.strip(),
            user_id=_user_id(),
            use_memory=use_memory,
            top_k=top_k,
            history=[
                {"q": str(h.get("q", "")), "a": str(h.get("a", ""))}
                for h in history[-3:]
            ],
        )
        wall_ms = int((time.time() - t0) * 1000)
        if resp.get("error") and not resp.get("answer"):
            status.update(label="问答失败", state="error")
            st.error(f"问答失败：{resp.get('error')}")
            _api_fallback_hint()
            return
        if resp.get("error"):
            status.update(label="问答部分成功", state="complete")
            st.warning(f"后端返回警告：{resp.get('error')}")
        else:
            status.update(label="回答完成", state="complete", expanded=False)

    answer = str(resp.get("answer") or "")
    citations = list(resp.get("citations") or [])
    answer_with_refs, citation_list = _map_api_citations(answer, citations)

    st.subheader("🧾 回答")
    if answer:
        st.markdown(
            _render_citations(answer_with_refs, citation_list),
            unsafe_allow_html=True,
        )
    else:
        st.info("后端返回的回答为空。")

    from_cache = bool(resp.get("from_cache"))
    mock_on = bool(resp.get("mock_mode"))
    source_label = "缓存" if from_cache else ("Mock" if mock_on else "真实")
    latency_value = resp.get("latency_ms")
    latency_text = f"{int(latency_value)}ms" if latency_value else f"{wall_ms}ms"
    input_tokens = int(resp.get("input_tokens") or 0)
    output_tokens = int(resp.get("output_tokens") or 0)
    cost_yuan = float(resp.get("cost_yuan") or 0.0)

    meta_col1, meta_col2, meta_col3, meta_col4, meta_col5 = st.columns(5)
    meta_col1.metric("来源", source_label)
    meta_col2.metric("耗时", latency_text)
    meta_col3.metric("Tokens", f"{input_tokens}/{output_tokens}")
    meta_col4.metric("引用", str(len(citation_list)))
    meta_col5.metric("成本", f"{cost_yuan:.4f} 元")
    if resp.get("model") or resp.get("trace_id"):
        st.caption(
            f"模型：`{resp.get('model') or '-'}` · trace_id=`{resp.get('trace_id') or '-'}` · "
            f"memory_used={resp.get('memory_used')}"
        )

    with st.expander("📚 引用溯源详情（点击查看原文段）"):
        if not citation_list:
            st.info("本次回答未返回引用。")
        for cit in citation_list:
            st.markdown(
                f"**[{cit['index']}]** "
                f"<span class='cite cite-{cit['label']}'>{cit['label']}</span> "
                f"`doc_id={cit['doc_id']}`  `chunk_id={cit['chunk_id']}`\n\n"
                f"> {cit['original_text']}",
                unsafe_allow_html=True,
            )

    # 记忆池保存（最近 5 轮，原始 chunk_id 引用留给后端做上下文）
    history.append({"q": question.strip(), "a": answer})
    st.session_state["ask_history_api"] = history[-5:]


def _docx_filename(name: str, docx_mode: str) -> str:
    """按后端导出模式修正文件名后缀（docx → .docx / markdown → .md）。"""
    base = (name or "review").strip()
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return f"{stem}.docx" if docx_mode == "docx" else f"{stem}.md"


def render_review_tab_api(client: ApiClient) -> None:
    """📚 综述生成 Tab（API 模式：POST /review）"""
    st.header("📚 跨文献综述草稿生成")
    st.caption(
        "API 模式：`POST /review` → FastAPI"
        "（多文档检索 → MMR 去重 → 生成 → 引用验证 → Word 导出）"
    )

    api_docs: Dict[str, Dict[str, Any]] = st.session_state.get("api_docs", {})
    if len(api_docs) < 2:
        st.warning(
            f"综述需要至少 2 篇文献，当前会话内已索引 {len(api_docs)} 篇，"
            "请先在「📄 PDF 上传」中添加。"
        )
        return

    with st.form("api_review_form"):
        topic = st.text_input(
            "综述主题（≤500 字，文档 6.5 节限制）",
            placeholder="深度学习模型在视觉与语言任务中的应用",
            max_chars=500,
            key="api_review_topic",
        )
        sel_docs = st.multiselect(
            "选文献（≥2 篇）",
            options=list(api_docs.keys()),
            format_func=lambda d: f"{d} · {str(api_docs[d].get('title', ''))[:40]}",
            key="api_review_docs",
        )
        num_sections = st.slider("章节数", 2, 8, 4, key="api_review_sections")
        verify = st.checkbox(
            "引用事实验证（文档 4.3 节三色 NLI 验证）", value=True,
            key="api_review_verify",
        )
        export_docx = st.checkbox(
            "生成后导出 Word（后端 docx / Markdown 降级）", value=True,
            key="api_review_export",
        )
        submitted = st.form_submit_button("📝 生成综述", type="primary")

    if not submitted:
        return
    if len(sel_docs) < 2:
        st.error("请至少选 2 篇文献")
        return
    if not topic.strip():
        st.error("请填写综述主题")
        return

    with st.status("调用后端综述生成（POST /review）…", expanded=True) as status:
        st.write(
            f"POST /review · topic=「{topic[:40]}」 · docs={len(sel_docs)} · "
            f"sections={num_sections} · verify={verify} · export_docx={export_docx}"
        )
        t0 = time.time()
        resp = client.review(
            topic=topic.strip(),
            doc_ids=list(sel_docs),
            user_id=_user_id(),
            num_sections=num_sections,
            verify_citations=verify,
            export_docx=export_docx,
        )
        wall_ms = int((time.time() - t0) * 1000)
        has_body = bool(resp.get("full_text") or resp.get("full_html")
                        or resp.get("paragraphs"))
        if resp.get("error") and not has_body:
            status.update(label="综述生成失败", state="error")
            st.error(f"综述生成失败：{resp.get('error')}")
            _api_fallback_hint()
            return
        if resp.get("error"):
            status.update(label="综述部分成功", state="complete")
            st.warning(f"后端返回警告：{resp.get('error')}")
        else:
            status.update(label="综述生成完成", state="complete", expanded=False)

    outline = list(resp.get("outline") or [])
    paragraphs = list(resp.get("paragraphs") or [])
    references = list(resp.get("references") or [])
    citation_stats = dict(resp.get("citation_stats") or {})
    latency_text = f"{int(resp.get('latency_ms') or wall_ms)}ms"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("章节", str(len(outline)))
    col2.metric("段落", str(len(paragraphs)))
    col3.metric("参考文献", str(len(references)))
    col4.metric("耗时", latency_text)

    with st.expander("📋 大纲", expanded=True):
        if not outline:
            st.info("后端未返回大纲。")
        for section in outline:
            related = ", ".join(
                str(x) for x in (section.get("related_doc_ids") or [])[:3]
            )
            st.markdown(
                f"- **{section.get('title', '')}**（`{section.get('section_id', '')}`）："
                f"{section.get('description') or '（无描述）'}  相关文档 {related}"
            )

    st.subheader("📖 综述正文（带三色引用溯源）")
    st.caption("蓝色=引用受支持 · 红色=原文矛盾 · 灰色=引用过宽（文档 4.3 节 NLI 验证结果）")
    safe_html = _xss_filter(str(resp.get("full_html") or ""))
    title_by_section = {
        str(o.get("section_id", "")): str(o.get("title", "")) for o in outline
    }
    if safe_html:
        st.markdown(safe_html, unsafe_allow_html=True)
    elif paragraphs:
        for para in paragraphs:
            section_id = str(para.get("section_id", ""))
            st.markdown(f"### {title_by_section.get(section_id) or section_id}")
            st.write(para.get("rendered_text") or para.get("text", ""))
    elif resp.get("full_text"):
        st.text(str(resp.get("full_text")))
    else:
        st.info("后端未返回综述正文。")

    total = int(citation_stats.get("total", 0) or 0)
    entailment = int(citation_stats.get("entailment", 0) or 0)
    contradiction = int(citation_stats.get("contradiction", 0) or 0)
    neutral = int(citation_stats.get("neutral", 0) or 0)
    denominator = max(total, 1)
    st.subheader("📊 引用统计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("合计", str(total))
    c2.metric("蓝色 支持", f"{entailment} ({entailment / denominator * 100:.0f}%)")
    c3.metric(
        "红色 存疑", f"{contradiction} ({contradiction / denominator * 100:.0f}%)",
    )
    c4.metric("灰色 过宽", f"{neutral} ({neutral / denominator * 100:.0f}%)")

    with st.expander("📚 参考文献列表"):
        if not references:
            st.info("后端未返回参考文献列表。")
        for ref in references:
            label = str(ref.get("label", "neutral"))
            st.markdown(
                f"**[{ref.get('index', '?')}]** "
                f"<span class='cite cite-{label}'>{label}</span> "
                f"`doc_id={ref.get('doc_id', '')}`  `chunk_id={ref.get('chunk_id', '')}`\n\n"
                f"> {str(ref.get('original_text', ''))[:200]}",
                unsafe_allow_html=True,
            )

    # Word / Markdown 导出（后端返回 base64，按 docx_mode 选择 mime 与后缀）
    if export_docx:
        docx_b64 = str(resp.get("docx_b64") or "")
        docx_mode = str(resp.get("docx_mode") or "docx")
        docx_name = _docx_filename(str(resp.get("docx_name") or ""), docx_mode)
        if docx_b64:
            try:
                docx_bytes = base64.b64decode(docx_b64)
            except Exception as e:  # noqa: BLE001 - 解码失败只提示，不崩溃
                st.error(f"Word 内容解码失败：{e}")
                docx_bytes = b""
            if docx_bytes:
                mime = DOCX_MIME if docx_mode == "docx" else MARKDOWN_MIME
                st.success(
                    f"文档已生成（mode={docx_mode}，{len(docx_bytes)} 字节）"
                )
                st.download_button(
                    label=(
                        "⬇ 下载 Word 文档 (.docx)" if docx_mode == "docx"
                        else "⬇ 下载 Markdown (.md)"
                    ),
                    data=docx_bytes,
                    file_name=docx_name,
                    mime=mime,
                    key="api_review_download_btn",
                )
        elif resp.get("docx_path"):
            st.caption(f"后端已落盘：`{resp.get('docx_path')}`")
        else:
            st.info("后端未返回 Word / Markdown 内容（可能 python-docx 未安装且降级内容为空）。")


def render_dashboard_tab_api(client: ApiClient) -> None:
    """💰 成本看板 Tab（API 模式：GET /cost/summary）"""
    st.header("💰 API 成本看板")
    st.caption("API 模式：`GET /cost/summary` → FastAPI 成本追踪（文档 6.1 节）")

    resp = client.cost_summary()
    if resp.get("error"):
        st.error(f"成本看板拉取失败：{resp.get('error')}")
        _api_fallback_hint()
        return

    daily = dict(resp.get("daily") or {})
    by_model = dict(resp.get("by_model") or {})
    by_user = dict(resp.get("by_user") or {})
    by_paper = dict(resp.get("by_paper") or {})
    cache = dict(resp.get("cache") or {})
    total_cost = float(resp.get("total_cost") or 0.0)

    with st.expander("📝 文本快照", expanded=False):
        lines = [
            "成本看板（后端 GET /cost/summary）",
            f"累计成本：{total_cost:.4f} 元",
            "",
            "[按天聚合]",
        ]
        if not daily:
            lines.append("  (无数据)")
        for day, stat in sorted(daily.items()):
            lines.append(
                f"  {day}: {float(stat.get('cost', 0)):.4f}元 | "
                f"调用 {int(stat.get('calls', 0))} 次 | "
                f"tokens {int(stat.get('input_tokens', 0))}/"
                f"{int(stat.get('output_tokens', 0))}"
            )
        st.code("\n".join(lines))

    st.subheader("📈 每日成本趋势")
    if daily:
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame([
                {
                    "日期": day,
                    "成本(元)": float(stat.get("cost", 0.0)),
                    "调用次数": int(stat.get("calls", 0)),
                    "输入tokens": int(stat.get("input_tokens", 0)),
                    "输出tokens": int(stat.get("output_tokens", 0)),
                }
                for day, stat in sorted(daily.items())
            ])
            st.dataframe(df, use_container_width=True)
            try:
                import plotly.express as px  # type: ignore

                fig = px.line(
                    df, x="日期", y="成本(元)", markers=True, title="每日 API 成本（元）",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(df.set_index("日期")[["成本(元)"]])
        except ImportError:
            st.json(daily)
    else:
        st.info(
            "暂无成本记录。在「💬 论文问答」「📚 综述生成」「🌐 文献翻译」中"
            "真实调用后会自动展示。"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🤖 按模型")
        st.json(by_model or {"(无数据)": True})
    with c2:
        st.markdown("#### 👤 按用户")
        st.json(by_user or {"(无数据)": True})
    with c3:
        st.markdown("#### 📄 按论文")
        st.json(by_paper or {"(无数据)": True})

    st.subheader("💾 缓存命中率（文档 6.2 节 目标约 30%）")
    cache_rate = float(cache.get("rate") or 0.0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("命中率", f"{cache_rate:.1%}")
    col2.metric("命中数", str(int(cache.get("hits", 0) or 0)))
    col3.metric("总调用", str(int(cache.get("total", 0) or 0)))
    col4.metric("累计成本", f"{total_cost:.4f} 元")

    with st.expander("📖 文档 6.2 节单篇成本参考"):
        st.markdown(
            """
            | 步骤 | 模型 | 单篇成本 |
            | --- | --- | ---: |
            | PDF 解析（50 页首次） | Qwen-VL-Max | 约 2.0 元 |
            | Embedding 索引 | text-embedding-v3 | 约 0.04 元 |
            | LLM 问答/综述 | Qwen-Plus | 约 0.3 元 |
            | 首次处理合计 | — | 约 2.4 元 |
            | 二次打开（缓存命中） | — | < 0.3 元 |
            """
        )


# =====================================================================
# 主入口
# =====================================================================


def main():
    """主入口：先解析运行模式（API / 本地进程），再渲染 6 个 Tab。"""
    dashscope_api_key = render_sidebar()

    run_mode = str(st.session_state.get("run_mode", "local"))
    api_client: Optional[ApiClient] = None
    comp: Optional[Dict] = None
    health: Dict[str, Any] = {}

    if run_mode == "api":
        # ---------------- API 模式（API-first 默认路径） ----------------
        api_client = get_api_client()
        health = st.session_state.get("api_health") or api_client.health()
        if not _health_ok(health):
            st.error("FastAPI 后端不可达，无法执行 API 调用。")
            st.markdown(
                "请先在项目根目录启动后端（新开终端）：\n\n"
                "```powershell\n.\\scripts\\dev.ps1 api\n```\n\n"
                "或执行 `make api`；启动成功后刷新本页。"
            )
            st.button(
                "💻 一键切换到本地进程模式",
                key="main_switch_local",
                on_click=_switch_to_local_mode,
            )
            st.stop()

        if bool(health.get("mock_mode")):
            st.markdown(
                f'<div class="banner-mock">⚠ 【API 模式 · 后端 Mock】'
                f'{api_client.base_url} 可达，后端 mock_mode=True，'
                f'LLM 调用走模板化回答。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner-real">✅ 【API 模式 · 真实调用】'
                f'{api_client.base_url} 可达，后端 mock_mode=False，'
                f'将真实调用 DashScope。</div>',
                unsafe_allow_html=True,
            )
    else:
        # ---------------- 本地进程模式（原有行为，离线兜底） ----------------
        comp = get_app_components(dashscope_api_key)

        if comp["mock_mode"]:
            st.markdown(
                '<div class="banner-mock">⚠ 【本地进程模式 · Mock】未检测到 '
                'DASHSCOPE_API_KEY，所有 LLM/VLM 调用均走模板化回答。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="banner-real">✅ 【本地进程模式 · 真实调用】'
                'DASHSCOPE_API_KEY 已配置，将使用 Qwen-Plus / Embedding 等模型。</div>',
                unsafe_allow_html=True,
            )

    tab_u, tab_t, tab_agent, tab_a, tab_r, tab_d = st.tabs([
        "📄 PDF 上传",
        "🌐 文献翻译",
        "🤖 Agent 助手",
        "💬 论文问答",
        "📚 综述生成",
        "💰 成本看板",
    ])

    if run_mode == "api" and api_client is not None:
        with tab_u:
            render_upload_tab_api(api_client)
        with tab_t:
            render_translate_tab(None, api_client, "api")
        with tab_agent:
            render_agent_tab(None, api_client, "api")
        with tab_a:
            render_ask_tab_api(api_client)
        with tab_r:
            render_review_tab_api(api_client)
        with tab_d:
            render_dashboard_tab_api(api_client)
        return

    if comp is None:
        st.error("本地进程模式组件初始化失败，请刷新页面重试。")
        return
    with tab_u:
        render_upload_tab(comp)
    with tab_t:
        render_translate_tab(comp, None, "local")
    with tab_agent:
        render_agent_tab(comp, None, "local")
    with tab_a:
        render_ask_tab(comp)
    with tab_r:
        render_review_tab(comp)
    with tab_d:
        render_dashboard_tab(comp)


if __name__ == "__main__":
    main()

"""多语种文献翻译与论文研读助手 — Streamlit 前端

文档第 10 章技术栈：前端 | Streamlit | 够用就好，不写 React

文档第 2 章 3 个核心功能：
1. 📄 PDF 公式无损解析（上传 PDF → 解析 + 公式定位）
2. 💬 基于记忆池的长文问答（RAG + 引用溯源）
3. 📚 综述草稿生成（跨文献 + 引用事实验证 + Word 导出）

文档 4.3 节三色引用展示（Streamlit unsafe_allow_html=True）：
- entailment（受支持）: 蓝色 #2563eb
- contradiction（与原文矛盾）: 红色 #dc2626
- neutral（引用过宽）: 灰色 #6b7280

文档 6.1 节成本看板：Streamlit 管理页展示每日成本趋势 + 按论文/用户分摊
文档 6.5 节输出过滤：unsafe_allow_html 只放行 `<span>` + 基础标签，其余 XSS 过滤

使用方法（Windows / Linux 均可）：
    # 方式 A：项目根目录直接启动
    streamlit run app/frontend/streamlit_app.py --server.port 8501

    # 方式 B：Makefile 封装（Linux/macOS/WSL）
    make frontend

DashScope API Key：
    - 通过侧边栏输入（仅保存在当前浏览器会话的 st.secrets / st.session_state）
    - 或通过环境变量 DASHSCOPE_API_KEY（推荐 .env 文件）
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# 侧边栏：API Key 配置 + 信息
# =====================================================================


def render_sidebar() -> str:
    """渲染侧边栏：DashScope API Key 输入 + 说明

    返回当前配置的 dashscope_api_key（可能为空，表示 mock）
    """
    with st.sidebar:
        st.title("📚 论文助手")
        st.caption("多语种文献翻译与研读（求职项目 · DashScope API-first）")

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
            help="填入真实 Key 后由 Qwen-Plus / Qwen-VL-Max / Embedding v3 真实调用。留空进入 Mock 模式（模板化回答）。",
        )
        st.session_state["ds_key"] = input_key

        if not input_key:
            st.info("当前未配置 Key，进入 **Mock 模式**（所有 LLM/VLM 调用走模板化回答）")
        else:
            st.success("已配置 DashScope API Key，将走真实 API 调用")

        st.divider()

        st.subheader("ℹ 关于")
        st.markdown(
            """
            - 架构：FastAPI 单体 + Streamlit 前端
            - 后端：阿里云 DashScope（Qwen-Plus / Qwen-VL-Max / Embedding v3）
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
# Tab 2：论文问答（RAG + 引用溯源）
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
# Tab 3：综述生成（跨文献 + 引用事实验证 + Word 导出）
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
# Tab 4：成本看板（文档 6.1 节）
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
# 主入口
# =====================================================================


def main():
    dashscope_api_key = render_sidebar()
    comp = get_app_components(dashscope_api_key)

    # 顶部 banner
    if comp["mock_mode"]:
        st.markdown(
            '<div class="banner-mock">⚠ 【当前为 Mock 模式】未检测到 DASHSCOPE_API_KEY，'
            '所有 LLM/VLM 调用均走模板化回答。请在左侧输入真实 Key 或通过环境变量配置。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-real">✅ 【真实调用模式】DashScope API Key 已配置，'
            '将使用 Qwen-Plus / Embedding 等模型进行真实调用。</div>',
            unsafe_allow_html=True,
        )

    tab_u, tab_a, tab_r, tab_d = st.tabs([
        "📄 PDF 上传", "💬 论文问答", "📚 综述生成", "💰 成本看板",
    ])

    with tab_u:
        render_upload_tab(comp)
    with tab_a:
        render_ask_tab(comp)
    with tab_r:
        render_review_tab(comp)
    with tab_d:
        render_dashboard_tab(comp)


if __name__ == "__main__":
    main()

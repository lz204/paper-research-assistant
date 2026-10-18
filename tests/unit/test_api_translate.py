"""后端 API 扩展 pytest 测试（多语种翻译 / 文档索引 / 成本看板 + 上传问答综述扩展）。

全部使用 create_app(mock_mode=True) + FastAPI TestClient，不触发任何真实 DashScope 调用。
覆盖冻结契约 v1：
- POST /translate：mock 成功 / 空文本 / 超长文本
- POST /upload：mock 分块 + 进程内索引，GET /documents 能列出该 doc_id
- POST /ask：mock 回答 + model/tokens/mock_mode 字段
- POST /review：export_docx 内存渲染 base64 回传
- GET /cost/summary：看板结构完整
- GET /health：translation 组件标记

运行：
    pytest tests/unit/test_api_translate.py -v
"""

from __future__ import annotations

import base64

import pytest

from app.api import (
    MAX_TRANSLATE_LEN,
    TranslateRequest,
    TranslateResponse,
    create_app,
)

SAMPLE_EN = "Self-attention is the core of Transformer."


@pytest.fixture()
def client():
    """mock 模式 FastAPI TestClient（fastapi 未安装时跳过）"""
    app = create_app(mock_mode=True)
    if app is None:  # pragma: no cover - 环境缺 fastapi
        pytest.skip("fastapi 未安装，跳过 API 测试")
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename: str, raw: bytes) -> dict:
    """上传 base64 编码的假 PDF，返回响应 JSON"""
    resp = client.post("/upload", json={
        "filename": filename,
        "file_b64": base64.b64encode(raw).decode("ascii"),
        "user_id": "u_test",
    })
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# 0. 模块级 schema / 常量（前端契约：默认值必须稳定）
# ---------------------------------------------------------------------------


def test_translate_schema_and_limit() -> None:
    """TranslateRequest / TranslateResponse 默认值 + MAX_TRANSLATE_LEN 常量"""
    req = TranslateRequest(text="hello")
    assert req.source_lang == "en"
    assert req.target_lang == "zh-CN"
    assert req.user_id == ""

    resp = TranslateResponse()
    assert resp.translated == ""
    assert resp.aligned_terms == []
    assert resp.target_lang == "zh-CN"
    assert resp.mock_mode is False
    assert resp.error == ""
    assert MAX_TRANSLATE_LEN == 5000


# ---------------------------------------------------------------------------
# 1. POST /translate（多语种文献翻译）
# ---------------------------------------------------------------------------


def test_translate_mock_success(client) -> None:
    """mock 翻译成功：译文非空 + 术语校准 + mock_mode=True"""
    resp = client.post("/translate", json={
        "text": SAMPLE_EN,
        "source_lang": "en",
        "target_lang": "zh-CN",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    assert data["translated"], "译文不能为空"
    assert data["source_text"] == SAMPLE_EN
    assert data["target_lang"] == "zh-CN"
    assert data["mock_mode"] is True
    assert isinstance(data["aligned_terms"], list)
    assert data["aligned_terms"], "自注意力/Transformer 应命中内置术语词库"
    assert isinstance(data["latency_ms"], int)


def test_translate_empty_text_rejected(client) -> None:
    """空文本（含纯空白）→ error 非空，不做翻译"""
    resp = client.post("/translate", json={"text": "   "})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] != ""
    assert data["translated"] == ""


def test_translate_too_long_rejected(client) -> None:
    """超长文本（6000 字 > MAX_TRANSLATE_LEN）→ error 非空"""
    resp = client.post("/translate", json={"text": "x" * 6000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] != ""
    assert str(MAX_TRANSLATE_LEN) in data["error"]


# ---------------------------------------------------------------------------
# 2. POST /upload（mock 分块 + 进程内索引）+ GET /documents
# ---------------------------------------------------------------------------


def test_upload_mock_chunks_and_docs(client) -> None:
    """mock 上传不再返回 0 页 0 chunk，且 GET /documents 能列出索引中的文档"""
    data = _upload(client, "mock_paper.pdf", b"%PDF-1.4 fake pdf")
    assert data["error"] == ""
    assert data["doc_id"].startswith("doc_")
    assert data["num_pages"] == 1
    assert data["num_chunks"] > 0
    assert data["num_chunks"] == len(data["chunks"])
    assert data["title"] == "mock_paper.pdf"
    assert data["formula_resolved"] is True
    assert data["need_review"] is False

    # chunk dict 统一 key：chunk_id / text（≤400 字）/ token_count / doc_id / section
    first = data["chunks"][0]
    assert set(first) == {"chunk_id", "text", "token_count", "doc_id", "section"}
    assert first["chunk_id"]
    assert first["text"]
    assert len(first["text"]) <= 400
    assert first["doc_id"] == data["doc_id"]
    assert isinstance(first["token_count"], int)
    assert isinstance(first["section"], str)

    docs_resp = client.get("/documents")
    assert docs_resp.status_code == 200
    docs_data = docs_resp.json()
    assert docs_data["error"] == ""
    assert docs_data["docs"], "上传后文档索引不应为空"
    for item in docs_data["docs"]:
        assert set(item) == {"doc_id", "title", "num_chunks"}
    indexed = {item["doc_id"]: item for item in docs_data["docs"]}
    assert data["doc_id"] in indexed
    assert indexed[data["doc_id"]]["num_chunks"] == data["num_chunks"]
    assert indexed[data["doc_id"]]["title"] == "mock_paper.pdf"


# ---------------------------------------------------------------------------
# 3. POST /ask（mock 回答 + 成本字段）
# ---------------------------------------------------------------------------


def test_ask_mock(client) -> None:
    """mock 问答：answer 非空 + model/tokens/mock_mode 字段齐全"""
    resp = client.post("/ask", json={
        "paper_id": "p1",
        "question": "什么是 Transformer？",
        "user_id": "u_test",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    assert data["answer"]
    assert data["model"] == "mock-qwen-plus"
    assert data["mock_mode"] is True
    assert data["input_tokens"] > 0
    assert data["output_tokens"] > 0
    assert data["cost_yuan"] == 0.0
    assert data["memory_used"] is False


# ---------------------------------------------------------------------------
# 4. POST /review（mock + Word/Markdown 内存渲染 base64 回传）
# ---------------------------------------------------------------------------


def test_review_mock_export_docx(client) -> None:
    """两篇不同 PDF → 两个 doc_id；export_docx 返回 base64 文档内容"""
    up1 = _upload(client, "paper_a.pdf", b"%PDF-1.4 fake pdf A")
    up2 = _upload(client, "paper_b.pdf", b"%PDF-1.4 fake pdf B")
    assert up1["doc_id"] != up2["doc_id"]

    resp = client.post("/review", json={
        "topic": "Transformer 在视觉与语言任务中的对比",
        "doc_ids": [up1["doc_id"], up2["doc_id"]],
        "num_sections": 2,
        "verify_citations": False,
        "export_docx": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "", f"综述生成失败: {data['error']}"
    assert data["full_text"]
    assert data["docx_b64"], "export_docx=True 应回传 base64 文档"
    assert base64.b64decode(data["docx_b64"]), "docx_b64 应可解码且非空"
    assert data["docx_mode"] in ("docx", "markdown")
    assert data["docx_name"]
    assert data["docx_name"].startswith("review_")
    assert data["docx_name"].endswith(".docx" if data["docx_mode"] == "docx" else ".md")
    assert data["docx_size"] > 0
    assert data["docx_path"] == ""
    assert data["mock_mode"] is True


def test_review_without_export_docx(client) -> None:
    """export_docx=False 时不返回 base64 文档（默认行为不破坏）"""
    up = _upload(client, "paper_c.pdf", b"%PDF-1.4 fake pdf C")
    resp = client.post("/review", json={
        "topic": "小样本学习综述",
        "doc_ids": [up["doc_id"]],
        "num_sections": 2,
        "verify_citations": False,
        "export_docx": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    assert data["docx_b64"] == ""
    assert data["docx_mode"] == ""
    assert data["docx_name"] == ""
    assert data["docx_size"] == 0


# ---------------------------------------------------------------------------
# 5. GET /cost/summary（成本看板结构）+ GET /health
# ---------------------------------------------------------------------------


def test_cost_summary_structure(client) -> None:
    """成本看板 JSON 结构完整（total_cost 为 float，各聚合维度存在）"""
    resp = client.get("/cost/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == ""
    assert isinstance(data["total_cost"], float)
    assert data["total_cost"] >= 0.0
    for key in ("daily", "by_model", "by_user", "by_paper", "cache"):
        assert isinstance(data[key], dict), f"{key} 应为 dict"
    for key in ("rate", "hits", "total"):
        assert key in data["cache"], f"cache 缺少 {key} 字段"
    assert data["cache"]["total"] >= data["cache"]["hits"] >= 0


def test_health_translation_component(client) -> None:
    """health.components 增加 translation 组件状态"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["components"]["translation"] == "mock"


# ---------------------------------------------------------------------------
# 6. 真实编排路径（组件注入 mock，绝不发真实 API）：记忆池注入 + 成本记录
# ---------------------------------------------------------------------------


def test_run_ask_history_injection_and_cost_record() -> None:
    """run_ask：history 注入最近 3 轮 Q/A，并写入 qa.ask 成本记录（文档 6.1 节）"""
    from app.api.routes import _AppState
    from app.llm import LLMResponse
    from app.rag.chunker import HierarchicalChunker
    from app.rag.embedder import FakeEmbedder

    class _RecordingLLM:
        """记录 prompt 的假 LLM，避免真实 DashScope 调用"""

        model = "mock-qwen-plus"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat(self, prompt, system=None, temperature=0.7, max_tokens=2048):
            self.prompts.append(prompt)
            return LLMResponse(
                text="答案 [doc_x#0.0]",
                model=self.model,
                input_tokens=10,
                output_tokens=5,
                cost_yuan=0.0,
            )

    state = _AppState(mock_mode=False)  # 直接调 run_ask 真实编排（不经路由 mock 分支）
    chunks = HierarchicalChunker(chunk_size=256, overlap=64).chunk(
        "# P\n\n## 1. Method\n\nWe use ResNet-152 on ImageNet-1K with 85.6% accuracy.",
        doc_id="doc_x",
    )
    state._index_doc("doc_x", "P", chunks)
    state._embedder = FakeEmbedder(dim=64)
    llm = _RecordingLLM()
    state._llm_client = llm

    resp = state.run_ask(
        paper_id="doc_x",
        question="ImageNet accuracy?",
        user_id="u_test",
        use_memory=True,
        top_k=3,
        history=[
            {"q": "什么是 ResNet？", "a": "一种残差网络。"},
            {"q": "ImageNet 数据集？", "a": "图像分类基准。"},
        ],
    )
    assert resp.error == ""
    assert resp.answer
    assert resp.memory_used is True
    assert resp.mock_mode is False
    assert resp.model == "mock-qwen-plus"
    # prompt 前缀格式与前端本地模式一致
    assert "会话上下文（最近 3 轮）" in llm.prompts[0]
    assert "Q1: 什么是 ResNet？" in llm.prompts[0]
    assert "A2: 图像分类基准。" in llm.prompts[0]

    # 成本记录已写入（mock LLM 成本 0，但仍记录 token/用户/论文维度）
    records = state._ensure_cost_tracker().get_records(step="qa.ask")
    assert len(records) == 1
    assert records[0].model == "mock-qwen-plus"
    assert records[0].user_id == "u_test"
    assert records[0].paper_id == "doc_x"
    assert records[0].input_tokens == 10
    assert records[0].output_tokens == 5

"""P0-2/3/4 接线验证：记忆池 / MMR 去重 / NLI 引用验证真正进入 API 主路径。

覆盖契约（全部 mock_mode=True，绝不触达真实 DashScope API）：
- P0-2：/upload 按 doc_id 建 ContextManager；/ask 注入记忆池实体并返回
  memory_used / memory_items；mock 模式同样生效；无 doc 安全降级；
- P0-3：/review 传入 embedder+retriever，Map-Reduce 的 MMR 去重真正执行，
  dedup_stats 透出 calls / input_summaries / selected_summaries / skipped /
  reduce_top_k / mmr_lambda（跳过也计数，不允许静默）；
- P0-4：/review 传入 retriever 后 CitationVerifier 自动创建，NLI 三色判定可用；
  伪造 chunk_id 判 contradiction；verify_citations=False 时 verified=False 且
  verification_skipped=False。

运行：
    pytest tests/unit/test_api_memory_mmr_nli.py -v
"""

from __future__ import annotations

import base64

import pytest

from app.api import AskResponse, ReviewResponse, create_app

# ---------------------------------------------------------------------------
# 测试数据与公共工具
# ---------------------------------------------------------------------------

# 各用例用不同字节，避免 doc_id（文件 MD5）在不同用例/参数间串号
FAKE_PDF_MEM = b"%PDF-1.4 fake pdf for memory pool test"
FAKE_PDF_MEM_VLM = b"%PDF-1.4 fake pdf for memory pool vlm test"
FAKE_PDF_ASK = b"%PDF-1.4 fake pdf for ask memory test"
FAKE_PDF_REVIEW_A = b"%PDF-1.4 fake pdf for review mmr nli test A"
FAKE_PDF_REVIEW_B = b"%PDF-1.4 fake pdf for review mmr nli test B"
FAKE_PDF_MMR_A = b"%PDF-1.4 fake pdf for mmr execution test A"
FAKE_PDF_MMR_B = b"%PDF-1.4 fake pdf for mmr execution test B"
FAKE_PDF_FAKE_CIT = b"%PDF-1.4 fake pdf for fake citation test"


@pytest.fixture()
def client():
    """mock 模式 FastAPI TestClient（显式 mock_mode=True，不依赖清空环境变量）"""
    app = create_app(mock_mode=True)
    if app is None:  # pragma: no cover - 环境缺 fastapi
        pytest.skip("fastapi 未安装，跳过 API 测试")
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename: str, raw: bytes, *, use_vlm: bool = False) -> dict:
    """上传 base64 编码的假 PDF，返回响应 JSON"""
    payload = {
        "filename": filename,
        "file_b64": base64.b64encode(raw).decode("ascii"),
        "user_id": "u_memory_mmr_nli",
    }
    if use_vlm:
        payload["use_vlm"] = True
    resp = client.post("/upload", json=payload)
    assert resp.status_code == 200
    return resp.json()


def _ask(client, paper_id: str, question: str, *, use_memory: bool = True) -> dict:
    """POST /ask 并返回 JSON"""
    resp = client.post("/ask", json={
        "paper_id": paper_id,
        "question": question,
        "user_id": "u_memory_mmr_nli",
        "use_memory": use_memory,
    })
    assert resp.status_code == 200
    return resp.json()


def _review(
    client,
    topic: str,
    doc_ids: list,
    *,
    num_sections: int = 2,
    verify_citations: bool = True,
) -> dict:
    """POST /review 并返回 JSON"""
    resp = client.post("/review", json={
        "topic": topic,
        "doc_ids": doc_ids,
        "user_id": "u_memory_mmr_nli",
        "num_sections": num_sections,
        "verify_citations": verify_citations,
    })
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# 0. 响应 schema 新字段默认值（保持既有字段不变的向后兼容）
# ---------------------------------------------------------------------------


def test_response_schema_new_defaults() -> None:
    """AskResponse.memory_items / ReviewResponse 新字段默认值稳定"""
    ask = AskResponse()
    assert ask.memory_used is False
    assert ask.memory_items == 0

    review = ReviewResponse()
    assert review.verified is False
    assert review.mmr_applied is False
    assert review.dedup_stats == {}
    assert review.verification_skipped is False
    assert review.verification_error == ""


# ---------------------------------------------------------------------------
# 1. P0-2 记忆池：/upload 建池 + /ask 注入（mock 模式生效）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_vlm", "raw"),
    [(False, FAKE_PDF_MEM), (True, FAKE_PDF_MEM_VLM)],
    ids=["text_parse", "vlm_parse"],
)
def test_upload_builds_doc_memory(client, use_vlm: bool, raw: bytes) -> None:
    """上传后按 doc_id 建好记忆池，实体可被存储层读取"""
    upload = _upload(client, "memory_paper.pdf", raw, use_vlm=use_vlm)
    assert upload["error"] == ""
    doc_id = upload["doc_id"]

    state = client.app.state.app_state
    memory = state.get_doc_memory(doc_id)
    assert memory is not None, "上传后应创建该 doc 的记忆池"
    stats = memory.stats()
    assert stats["max_size"] == 200
    assert stats["total"] > 0, "MOCK 论文文本应抽出实体（数据集/模型/指标）"
    # 未上传 doc 安全返回 None（get_doc_memory 不报错）
    assert state.get_doc_memory("doc_never_uploaded") is None


@pytest.mark.parametrize(
    ("use_vlm", "raw"),
    [(False, FAKE_PDF_ASK), (True, FAKE_PDF_MEM)],
    ids=["text_parse", "vlm_parse"],
)
def test_ask_injects_memory_entities(client, use_vlm: bool, raw: bytes) -> None:
    """/ask 同 doc 问数据集：memory_used=True、memory_items>0、error 为空"""
    upload = _upload(client, "memory_ask.pdf", raw, use_vlm=use_vlm)
    assert upload["error"] == ""
    doc_id = upload["doc_id"]

    data = _ask(client, doc_id, "本文用了什么数据集？")
    assert data["error"] == "", f"问答失败: {data['error']}"
    assert data["answer"], "回答不能为空"
    assert data["memory_used"] is True, "命中实体时应注入记忆池"
    assert data["memory_items"] > 0, "应注入至少 1 条实体"
    # mock 模板回答中附实体预览（契约要求）
    assert "记忆池已注入" in data["answer"]
    assert "ImageNet" in data["answer"] or "COCO" in data["answer"]

    # use_memory=False → 不注入（不破坏既有开关语义）
    off = _ask(client, doc_id, "本文用了什么数据集？", use_memory=False)
    assert off["error"] == ""
    assert off["memory_used"] is False
    assert off["memory_items"] == 0
    assert "记忆池已注入" not in off["answer"]


def test_ask_without_memory_degrades_safely(client) -> None:
    """未上传 doc 询问：memory_used=False、memory_items=0，不报错不 500"""
    data = _ask(client, "doc_never_uploaded", "本文用了什么模型？")
    assert data["error"] == "", "无 doc 时 mock 仍应返回模板回答"
    assert data["answer"], "降级路径仍需返回模板回答"
    assert data["memory_used"] is False
    assert data["memory_items"] == 0
    assert "记忆池已注入" not in data["answer"]


def test_reupload_same_doc_rebuilds_memory(client) -> None:
    """同一 doc_id 重复上传：记忆池覆盖重建（新实例），不累加旧实体"""
    first = _upload(client, "memory_rebuild.pdf", FAKE_PDF_MEM)
    doc_id = first["doc_id"]
    state = client.app.state.app_state
    memory_before = state.get_doc_memory(doc_id)
    assert memory_before is not None
    total_before = memory_before.stats()["total"]

    second = _upload(client, "memory_rebuild.pdf", FAKE_PDF_MEM)
    assert second["doc_id"] == doc_id
    memory_after = state.get_doc_memory(doc_id)
    assert memory_after is not None
    assert memory_after is not memory_before, "重复上传应重建记忆池实例"
    assert memory_after.stats()["total"] == total_before, "实体总量不应累加"


def test_run_ask_real_path_injects_memory() -> None:
    """run_ask 真实编排（注入 mock 组件，不触网）：记忆池实体进 prompt 且计数"""
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
        user_id="u_memory_mmr_nli",
        use_memory=True,
        top_k=3,
        history=[{"q": "什么是 ResNet？", "a": "一种残差网络。"}],
    )
    assert resp.error == ""
    assert resp.answer
    assert resp.memory_used is True
    assert resp.memory_items > 0, "真实 run_ask 应统计注入的记忆池实体数"
    prompt = llm.prompts[0]
    assert "记忆池实体" in prompt, "记忆池 prompt 应进入真实 LLM prompt"
    assert "已知信息" in prompt, "应包含 build_prompt 的实体块"
    # 既有会话历史注入行为保留
    assert "会话上下文（最近 3 轮）" in prompt


# ---------------------------------------------------------------------------
# 2. P0-3 MMR 统计：/review 透出 dedup_stats（跳过也计数）
# ---------------------------------------------------------------------------


def test_review_wiring_mmr_and_nli(client) -> None:
    """/review 两篇 mock 上传：MMR 统计 + NLI 验证 + 引用统计真实可用"""
    up1 = _upload(client, "review_a.pdf", FAKE_PDF_REVIEW_A)
    up2 = _upload(client, "review_b.pdf", FAKE_PDF_REVIEW_B)
    data = _review(
        client,
        "Transformer 在视觉与语言任务中的应用",
        [up1["doc_id"], up2["doc_id"]],
        num_sections=2,
        verify_citations=True,
    )
    assert data["error"] == "", f"综述生成失败: {data['error']}"

    # --- MMR 统计：键齐全、类型正确，且 mmr_applied 与 calls 自洽 ---
    dedup = data["dedup_stats"]
    assert isinstance(dedup, dict)
    for key in (
        "mmr_applied", "calls", "input_summaries",
        "selected_summaries", "skipped", "reduce_top_k", "mmr_lambda",
    ):
        assert key in dedup, f"dedup_stats 缺字段 {key}"
    assert dedup["reduce_top_k"] == 8, "默认 REDUCE_TOP_K 应为 8"
    assert dedup["mmr_lambda"] == pytest.approx(0.7)
    assert data["mmr_applied"] == bool(dedup["calls"])
    assert dedup["mmr_applied"] == data["mmr_applied"]
    # 2 节各挂 1 篇文献 → summaries(1) <= REDUCE_TOP_K(8) 跳过 MMR，
    # 但跳过必须计数（skipped / input_summaries 非静默）
    assert dedup["skipped"] >= 1
    assert dedup["calls"] == 0
    assert dedup["input_summaries"] >= 1

    # --- NLI 引用验证：retriever 已传 → CitationVerifier 自动创建 → 不 skipped ---
    assert data["verification_skipped"] is False
    assert data["verification_error"] == ""
    assert data["verified"] is True, "retriever 已传时应完成真实验证"

    stats = data["citation_stats"]
    assert isinstance(stats, dict)
    assert stats.get("total", 0) > 0, "mock generator 每段应带引用"
    for key in ("entailment", "contradiction", "neutral", "total"):
        assert key in stats

    # 引用真实存在（mock generator 引用各 doc 首个 chunk），三色标签合法
    assert data["references"], "参考文献不应为空"
    for ref in data["references"]:
        assert ref["label"] in ("entailment", "contradiction", "neutral")
        assert ref["exists"] is True, "真实 chunk_id 应命中二次检索"


def test_review_mmr_executes_when_reduce_top_k_small(client, monkeypatch) -> None:
    """monkeypatch REDUCE_TOP_K=1：单章节 summaries(2) > 1 → MMR 真正执行"""
    from app.review.generator import MockReviewGenerator

    monkeypatch.setattr(MockReviewGenerator, "REDUCE_TOP_K", 1)

    up1 = _upload(client, "mmr_a.pdf", FAKE_PDF_MMR_A)
    up2 = _upload(client, "mmr_b.pdf", FAKE_PDF_MMR_B)
    # num_sections=1 → 该章节 related_doc_ids 含 2 篇文献 → 2 个 Map summaries
    data = _review(
        client,
        "多模态大模型综述",
        [up1["doc_id"], up2["doc_id"]],
        num_sections=1,
        verify_citations=True,
    )
    assert data["error"] == "", f"综述生成失败: {data['error']}"

    dedup = data["dedup_stats"]
    assert dedup["reduce_top_k"] == 1, "monkeypatch 应作用于本次请求的生成器"
    assert dedup["input_summaries"] >= 2
    assert dedup["calls"] >= 1, "summaries 超过 REDUCE_TOP_K 时 MMR 应真正执行"
    assert dedup["selected_summaries"] < dedup["input_summaries"], "MMR 应减少要点数"
    assert dedup["mmr_applied"] is True
    assert data["mmr_applied"] is True


# ---------------------------------------------------------------------------
# 3. P0-4 NLI：伪造引用判 contradiction
# ---------------------------------------------------------------------------


def test_review_fake_citation_marked_contradiction(client, monkeypatch) -> None:
    """段落里注入不存在的 chunk_id → NLI/二次检索判 contradiction"""
    from app.review.generator import MockReviewGenerator

    original_generate = MockReviewGenerator.generate
    fake_marker = "[doc_ghost, doc_ghost#9.9]"

    def _generate_with_fake_citation(self, topic, sections, doc_chunks):
        """在 mock generator 正常段落末尾追加一条伪造引用"""
        result = original_generate(self, topic, sections, doc_chunks)
        for paragraph in result.paragraphs:
            paragraph.text = (paragraph.text or "") + f" 伪造引用 {fake_marker}"
        return result

    monkeypatch.setattr(MockReviewGenerator, "generate", _generate_with_fake_citation)

    up = _upload(client, "fake_citation.pdf", FAKE_PDF_FAKE_CIT)
    data = _review(
        client,
        "引用真实性验证",
        [up["doc_id"]],
        num_sections=1,
        verify_citations=True,
    )
    assert data["error"] == "", f"综述生成失败: {data['error']}"
    assert data["verification_skipped"] is False

    stats = data["citation_stats"]
    assert stats.get("contradiction", 0) >= 1, (
        f"不存在的 chunk_id 应判 contradiction，实际 {stats}"
    )
    ghost_refs = [
        ref for ref in data["references"]
        if ref["chunk_id"] == "doc_ghost#9.9"
    ]
    assert ghost_refs, "伪造引用应出现在参考文献中（便于前端标红）"
    assert ghost_refs[0]["label"] == "contradiction"
    assert ghost_refs[0]["exists"] is False


# ---------------------------------------------------------------------------
# 4. verify_citations=False：不验证且不抛异常
# ---------------------------------------------------------------------------


def test_review_verify_citations_false(client) -> None:
    """verify_citations=False：verified=False、verification_skipped=False"""
    up = _upload(client, "no_verify.pdf", FAKE_PDF_REVIEW_A)
    data = _review(
        client,
        "关闭引用验证",
        [up["doc_id"]],
        num_sections=1,
        verify_citations=False,
    )
    assert data["error"] == "", f"综述生成失败: {data['error']}"
    assert data["verified"] is False
    assert data["verification_skipped"] is False
    assert data["verification_error"] == ""

    stats = data["citation_stats"]
    assert isinstance(stats, dict)
    assert stats.get("total", 0) == 0
    assert data["references"] == []

    # MMR 统计仍必须透出（不允许静默）
    dedup = data["dedup_stats"]
    assert isinstance(dedup, dict)
    for key in ("calls", "input_summaries", "skipped", "reduce_top_k"):
        assert key in dedup

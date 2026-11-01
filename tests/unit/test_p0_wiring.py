# -*- coding: utf-8 -*-
"""P0 接线契约测试（task-18 验收）：记忆池 → /ask、MMR/NLI → /review。

覆盖 5 组断言（全部 mock_mode=True，不触达真实 DashScope API）：
1) /upload(mock) → /ask：memory_entities>0、memory_used=True，且 use_memory=False 时不注入；
   memory_items 为 memory_entities 的兼容别名（同值）。
2) /review 默认（两篇 mock 文档、verify_citations=True）：citation_stats.total>0、
   verification_skipped=False、generator_mode/mmr_enabled 有值（mock 下 generator_mode="mock"）；
3) 无 doc 的 /review（未上传 doc_id / 空 doc_ids）：不抛异常，verification_skipped=True
   且 verification_skip_reason 非空；
4) monkeypatch app.review.pipeline.create_review_pipeline 捕获 kwargs：
   retriever/embedder 均非 None（证明 MMR/NLI 接线真正生效）；
5) monkeypatch REDUCE_TOP_K=1：mmr_applied=True 且 dedup_stats.calls>0（实际执行可观测）。

运行：
    pytest tests/unit/test_p0_wiring.py -v
"""

from __future__ import annotations

import base64

import pytest

from app.api import AskResponse, ReviewResponse, create_app

# 各用例用不同字节，避免 doc_id（文件 MD5）在用例间串号
PDF_ASK = b"%PDF-1.4 p0 wiring ask doc"
PDF_REVIEW_A = b"%PDF-1.4 p0 wiring review doc A"
PDF_REVIEW_B = b"%PDF-1.4 p0 wiring review doc B"
PDF_MMR_A = b"%PDF-1.4 p0 wiring mmr doc A"
PDF_MMR_B = b"%PDF-1.4 p0 wiring mmr doc B"

USER_ID = "u_p0_wiring"


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """mock 模式 FastAPI TestClient（显式 mock_mode=True，不依赖清空环境变量）"""
    app = create_app(mock_mode=True)
    if app is None:  # pragma: no cover - 环境缺 fastapi
        pytest.skip("fastapi 未安装，跳过 API 测试")
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename: str, raw: bytes) -> dict:
    """上传 base64 编码的假 PDF（mock 模式），返回响应 JSON"""
    resp = client.post("/upload", json={
        "filename": filename,
        "file_b64": base64.b64encode(raw).decode("ascii"),
        "user_id": USER_ID,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("error") == "", f"上传失败: {data}"
    return data


def _ask(client, paper_id: str, question: str, *, use_memory: bool = True) -> dict:
    resp = client.post("/ask", json={
        "paper_id": paper_id,
        "question": question,
        "user_id": USER_ID,
        "use_memory": use_memory,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _review(
    client,
    topic: str,
    doc_ids: list,
    *,
    num_sections: int = 2,
    verify_citations: bool = True,
) -> dict:
    resp = client.post("/review", json={
        "topic": topic,
        "doc_ids": doc_ids,
        "user_id": USER_ID,
        "num_sections": num_sections,
        "verify_citations": verify_citations,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 0. 响应 dataclass 契约字段默认值（向后兼容）
# ---------------------------------------------------------------------------


def test_response_contract_fields_default() -> None:
    ask = AskResponse()
    assert ask.memory_used is False
    assert ask.memory_entities == 0
    assert ask.memory_items == 0

    review = ReviewResponse()
    assert review.generator_mode == ""
    assert review.mmr_enabled is False
    assert review.verification_skipped is False
    assert review.verification_skip_reason == ""
    assert review.verification_error == ""


# ---------------------------------------------------------------------------
# 1. P0-2：/ask 注入记忆池实体（memory_entities）与 use_memory 开关
# ---------------------------------------------------------------------------


def test_ask_memory_entities_and_switch(client) -> None:
    doc_id = _upload(client, "wiring_ask.pdf", PDF_ASK)["doc_id"]

    on = _ask(client, doc_id, "本文用了什么数据集？", use_memory=True)
    assert on["error"] == "", f"问答失败: {on['error']}"
    assert on["memory_used"] is True, "命中实体时应注入记忆池"
    assert on["memory_entities"] > 0, "memory_entities 应大于 0"
    assert on["memory_items"] == on["memory_entities"], "兼容别名应与主字段同值"
    assert "记忆池已注入" in on["answer"], "mock 回答应体现记忆池生效"
    assert on["answer"].startswith("（记忆池已注入"), "实体摘要应按契约拼在 mock 答案前缀"

    off = _ask(client, doc_id, "本文用了什么数据集？", use_memory=False)
    assert off["error"] == ""
    assert off["memory_used"] is False, "use_memory=False 时不应注入记忆池"
    assert off["memory_entities"] == 0
    assert off["memory_items"] == 0
    assert "记忆池已注入" not in off["answer"]


# ---------------------------------------------------------------------------
# 2. P0-3/P0-4：/review 默认接线字段（citation_stats / generator_mode / mmr_enabled）
# ---------------------------------------------------------------------------


def test_review_default_wiring_fields(client) -> None:
    up1 = _upload(client, "wiring_review_a.pdf", PDF_REVIEW_A)
    up2 = _upload(client, "wiring_review_b.pdf", PDF_REVIEW_B)
    data = _review(
        client,
        "Transformer 在视觉与语言任务中的应用",
        [up1["doc_id"], up2["doc_id"]],
        num_sections=2,
        verify_citations=True,
    )
    assert data["error"] == "", f"综述生成失败: {data['error']}"

    # 引用事实验证真正执行：retriever 已传 → CitationVerifier 创建 → 不跳过
    assert data["verification_skipped"] is False
    assert data["verification_skip_reason"] == ""
    assert data["verification_error"] == ""
    stats = data["citation_stats"]
    assert isinstance(stats, dict) and stats.get("total", 0) > 0, f"引用统计异常: {stats}"
    for key in ("entailment", "contradiction", "neutral", "total"):
        assert key in stats

    # 生成器模式与 MMR 能力标记（mock 路径：generator_mode="mock"，mmr_enabled=False）
    assert data["generator_mode"] == "mock"
    assert data["mmr_enabled"] is False, "MockReviewGenerator 路径按契约 mmr_enabled=False"
    assert isinstance(data["mmr_applied"], bool)
    assert isinstance(data["dedup_stats"], dict)
    for key in ("calls", "input_summaries", "skipped", "reduce_top_k", "mmr_applied"):
        assert key in data["dedup_stats"], f"dedup_stats 缺字段 {key}"
    # mmr_applied 与实际 MMR 调用数自洽（默认候选数<=8，跳过也计数）
    assert data["mmr_applied"] == bool(data["dedup_stats"]["calls"])
    assert data["dedup_stats"]["skipped"] >= 1


# ---------------------------------------------------------------------------
# 3. 无 doc 的 /review：显式 skipped=True + reason 非空，不抛异常
# ---------------------------------------------------------------------------


def test_review_without_docs_marks_skipped(client) -> None:
    # 3a) doc_ids 指向未上传文献：run_review 路径（无 chunks → 不构造空 retriever）
    data = _review(
        client,
        "未上传文献的综述",
        ["doc_never_uploaded_p0_wiring"],
        num_sections=1,
        verify_citations=True,
    )
    assert data["error"] == "", f"无 doc 时不应抛异常/报错: {data['error']}"
    assert data["verification_skipped"] is True, "无可用 chunks 时应显式标记跳过"
    assert data["verification_skip_reason"], "跳过原因不能为空"
    assert "chunks" in data["verification_skip_reason"]
    assert data["verification_skip_reason"] == data["verification_error"]
    assert data["citation_stats"].get("total", 0) == 0
    assert data["verified"] is False
    assert data["generator_mode"] == "mock"
    assert data["mmr_enabled"] is False, "无 chunks 未传 embedder → mmr 未启用"
    assert isinstance(data["dedup_stats"], dict)

    # 3b) doc_ids 为空列表：路由提前返回，同样显式标记跳过（不 500）
    empty = _review(client, "空 doc_ids", [], num_sections=1, verify_citations=True)
    assert empty["error"], "空 doc_ids 应返回参数错误说明"
    assert empty["verification_skipped"] is True
    assert empty["verification_skip_reason"]


# ---------------------------------------------------------------------------
# 4. 单元级接线证明：monkeypatch create_review_pipeline 捕获 kwargs
# ---------------------------------------------------------------------------


def test_review_pipeline_receives_retriever_and_embedder(client, monkeypatch) -> None:
    from app.review import pipeline as pipeline_module

    captured: dict = {}
    real_create_review_pipeline = pipeline_module.create_review_pipeline

    def _spy_create_review_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return real_create_review_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "create_review_pipeline", _spy_create_review_pipeline
    )

    doc_id = _upload(client, "wiring_pipeline.pdf", PDF_REVIEW_A)["doc_id"]
    data = _review(client, "接线证明", [doc_id], num_sections=1, verify_citations=True)
    assert data["error"] == "", f"综述生成失败: {data['error']}"

    assert captured, "run_review 应调用 create_review_pipeline"
    assert captured.get("retriever") is not None, "retriever 必须传入（否则 NLI 被跳过）"
    assert captured.get("embedder") is not None, "embedder 必须传入（否则 MMR 被跳过）"
    assert captured.get("mode") == "mock", "mock 模式应显式传 mode='mock'（不依赖清空 env）"
    assert captured.get("verify_citations") is True
    # 与响应字段自洽：mock + embedder 已传，但 MockReviewGenerator 路径 mmr_enabled=False
    assert data["generator_mode"] == "mock"
    assert data["mmr_enabled"] is False
    assert data["verification_skipped"] is False
    assert data["citation_stats"].get("total", 0) > 0


# ---------------------------------------------------------------------------
# 5. MMR 实际执行可观测（monkeypatch REDUCE_TOP_K=1 触发去重）
# ---------------------------------------------------------------------------


def test_review_mmr_execution_observable(client, monkeypatch) -> None:
    from app.review.generator import MockReviewGenerator

    monkeypatch.setattr(MockReviewGenerator, "REDUCE_TOP_K", 1)

    up1 = _upload(client, "wiring_mmr_a.pdf", PDF_MMR_A)
    up2 = _upload(client, "wiring_mmr_b.pdf", PDF_MMR_B)
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
    assert dedup["calls"] >= 1, "候选数超过 REDUCE_TOP_K 时 MMR 应真正执行"
    assert dedup["mmr_applied"] is True
    assert data["mmr_applied"] is True, "mmr_applied 应反映实际执行（与 mmr_enabled 语义不同）"
    # 契约：MockReviewGenerator 路径 mmr_enabled 固定 False；实际执行看 mmr_applied
    assert data["generator_mode"] == "mock"
    assert data["mmr_enabled"] is False

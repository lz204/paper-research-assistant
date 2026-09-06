"""DashScope 真实 API 端到端验证（需要真实 Key）。

前置：DASHSCOPE_API_KEY 必须为真实 Key（环境变量或项目根 .env 均可）。
未设置 → 打印提示后 code=2 退出（CI 视为"跳过"，不失败）。

验证链路：
  1. LLM chat：调 Qwen-Plus，验证 ok/tokens/cost 合规 + QA 缓存命中
  2. QA RAG：用内置样例 + RAG 检索 + Qwen-Plus 生成
  3. 成本看板 + 结构化日志

（Qwen-VL-Max / Embedding v3 真实 PDF 解析暂不测，避免大成本）
运行：python tests/e2e/test_e2e_dashscope.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Key 检查前先 import app，触发 app._load_dotenv() 自动加载项目根 .env。
# 这样用户把 Key 写入 .env 即可，无需手动 set 环境变量。
import app  # noqa: F401  —  side-effect: load .env


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("=" * 70)
        print("⚠ 未配置 DASHSCOPE_API_KEY，跳过真实 DashScope E2E。")
        print("")
        print("请先设置真实 Key（阿里云 DashScope 控制台获取）：")
        print("  方式 A（推荐）：编辑项目根 .env 文件，写入：")
        print("      DASHSCOPE_API_KEY=sk-你的真实Key")
        print("  方式 B：命令行环境变量：")
        print("      （PowerShell）  $env:DASHSCOPE_API_KEY = 'sk-xxx'")
        print("      （bash/zsh）   export DASHSCOPE_API_KEY=sk-xxx")
        print("")
        print("填完后重新运行：python tests/e2e/test_e2e_dashscope.py")
        print("=" * 70)
        return 2  # CI 中 2 表示跳过

    # 有 Key，打印脱敏前缀（仅 sk-xxxxxxx... 前 6 后 4 位）
    masked = api_key[:6] + "***" + api_key[-4:]
    print("=" * 70)
    print(f"E2E DashScope 真实模式（Key={masked}）")
    print("=" * 70)

    total_t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Qwen-Plus chat
    # ------------------------------------------------------------------
    print("\n[Step 1] Qwen-Plus chat")
    from app.llm import create_llm_client, DashScopeLLMClient, QACache, LLMResponse, estimate_cost_yuan

    client = create_llm_client("dashscope")
    assert isinstance(client, DashScopeLLMClient), "DashScope 客户端应创建成功"
    assert client._ensure_client(), "DashScope client 初始化失败（检查 Key 是否正确）"

    q = "用 3 句话解释 Transformer 的 self-attention 机制。"
    t0 = time.time()
    resp = client.chat(q, system="你是深度学习领域的学术助教。")
    latency_ms = int((time.time() - t0) * 1000)
    print(f"  latency={latency_ms}ms")
    print(f"  model={resp.model}, tokens={resp.input_tokens}/{resp.output_tokens}")
    print(f"  cost_yuan={resp.cost_yuan:.4f}, ok={resp.ok}, partial={resp.partial}")
    if resp.ok and resp.text:
        print(f"  text preview: {resp.text[:100]}...")
    else:
        print(f"  text (empty/fail). error={resp.error or '<nil>'}")
    assert resp.ok, f"Qwen-Plus chat 失败: {resp.error or 'text 为空'}"
    assert resp.input_tokens > 0 and resp.output_tokens > 0, "tokens 为 0"
    assert resp.cost_yuan > 0, "成本应为正数"
    # 成本合理性：文档 6.2 节 Qwen-Plus 约 0.3 元/次，小问答应 < 0.05 元
    assert resp.cost_yuan < 0.05, f"单次小问答成本偏高: {resp.cost_yuan:.4f}元（参考 0.005-0.02）"
    # 估算函数一致性：与 DashScope 返回差距 < 10%
    est_cost = estimate_cost_yuan(resp.model, resp.input_tokens, resp.output_tokens)
    ratio = est_cost / max(resp.cost_yuan, 1e-9)
    assert 0.9 < ratio < 1.1, f"estimate_cost_yuan 与实际不一致（ratio={ratio:.2f}）"
    print(f"  ✓ estimate_cost_yuan 与 API 账单一致（ratio={ratio:.2f}）")

    # QA 缓存命中验证（文档 6.2 节，目标命中率约 30%）
    print("\n[Step 1b] QA 缓存命中（文档 6.2 节）")
    resp2 = client.chat(q, system="你是深度学习领域的学术助教。")
    assert resp2.from_cache is True, "同问题二次调用应命中缓存"
    assert resp2.cost_yuan == 0.0, "缓存命中不应计费"
    print(f"  ✓ from_cache={resp2.from_cache}, cost={resp2.cost_yuan}元")

    # ------------------------------------------------------------------
    # Step 2: RAG + Qwen-Plus（基于内置样例）
    # ------------------------------------------------------------------
    print("\n[Step 2] RAG + Qwen-Plus 问答")
    from app.rag import (
        HierarchicalChunker, FakeEmbedder, InMemoryVectorRetriever,
    )

    PAPER_1 = """# Attention Is All You Need
## Abstract
Transformer 是一个完全基于注意力机制的序列转导网络。我们的模型在
WMT 2014 English-to-German 任务上达到了 28.4 BLEU。
注意力公式：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V

## 1 Method
多头注意力：8 头，单头维度 d_k=64，总维度 d_model=512。
Adam optimizer：beta1=0.9，beta2=0.98，训练 100K 步。
在 ImageNet 分类任务上达到 85.6% top-5 准确率。
"""
    chunker = HierarchicalChunker(chunk_size=200, overlap=48)
    chunks = chunker.chunk(PAPER_1, doc_id="sample_e2e_p1")
    embedder = FakeEmbedder(dim=64)
    retriever = InMemoryVectorRetriever(embedder)
    retriever.add_chunks(chunks)

    q2 = "Transformer 在 WMT 2014 英德翻译任务上的 BLEU 分数是多少？"
    top = retriever.search(q2, top_k=3)
    print(f"  RAG 召回 {len(top)} 段")
    ctx = "\n".join(f"[{rr.chunk.chunk_id}] {rr.chunk.text}" for rr in top)[:2500]

    system2 = "你是学术问答助手。仅基于参考正文回答问题，并在关键结论处附 [chunk_id] 引用。"
    prompt2 = (
        f"参考正文：\n---\n{ctx}\n---\n\n问题：{q2}\n\n"
        f"要求：答案 50 字以内，附带 1 处 [chunk_id] 引用。"
    )
    resp_a = client.chat(prompt2, system=system2, temperature=0.1, max_tokens=256)
    print(f"  model={resp_a.model}, latency={resp_a.latency_ms}ms")
    print(f"  answer: {(resp_a.text or '')[:200]}")
    # 答案里应包含数字 28.4（不强制 100%，但打印供人工核对）
    has_284 = "28.4" in (resp_a.text or "") or "28,4" in (resp_a.text or "")
    has_chunk_id = any(rr.chunk.chunk_id in (resp_a.text or "") for rr in top)
    print(f"  含 28.4：{has_284}，含 chunk_id 引用：{has_chunk_id}")

    # ------------------------------------------------------------------
    # Step 3: 可观测性（DashScope → 成本追踪）
    # ------------------------------------------------------------------
    print("\n[Step 3] 成本追踪 + 看板（真实调用）")
    from app.observability import (
        InMemoryCostTracker, CostDashboard, CostRecord, create_logger,
    )
    from datetime import datetime

    tracker = InMemoryCostTracker()
    tracker.record(CostRecord(
        ts=datetime.now(), model=resp.model, step="e2e_ds.qwen_chat",
        input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
        cost_yuan=resp.cost_yuan, user_id="u_e2e_ds", paper_id="sample_e2e_p1",
    ))
    tracker.record(CostRecord(
        ts=datetime.now(), model=resp_a.model, step="e2e_ds.qa_rag",
        input_tokens=resp_a.input_tokens, output_tokens=resp_a.output_tokens,
        cost_yuan=resp_a.cost_yuan, user_id="u_e2e_ds", paper_id="sample_e2e_p1",
        from_cache=resp2.from_cache,
    ))
    dash = CostDashboard(tracker)
    board = dash.render_text()
    print(board)
    total_cost = tracker.total_cost()
    assert total_cost > 0, "总成本应 > 0"
    print(f"  总成本（真实 API 调用）：{total_cost:.4f}元（文档 6.2 节参考：首次单篇约 2.4 元，问答 0.3 元）")

    # 结构化日志
    log = create_logger("e2e_ds", trace_id="trace_e2e_ds_" + str(int(time.time())))
    log.info("DashScope E2E 通过", step="e2e_ds.finish",
             latency_ms=int((time.time() - total_t0) * 1000),
             status="ok", cost_yuan=total_cost)
    print(f"  ✓ 结构化日志 + 成本看板")

    total_ms = int((time.time() - total_t0) * 1000)
    print("\n" + "=" * 70)
    print(f"DashScope 真实模式 E2E 通过 ✓  总耗时 {total_ms}ms  "
          f"真实成本 {total_cost:.4f}元")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

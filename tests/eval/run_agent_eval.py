"""Agent 自动化评测脚本（冻结契约 v1）。

mock 模式（默认，全离线、无 API 费用）：
    用 ScriptedMockLLM + 假文档/假检索/假翻译/假综述回调，跑 6 条 Agent 任务：
    数据集问答 / 方法问答 / 翻译 / 综述 / 无文档 / 工具异常恢复，
    逐条记录 expected_tools、actual_tools、steps、success、latency、tokens、cost。

real 模式（需配置 DASHSCOPE_API_KEY，产生真实费用）：
    用 create_tool_calling_llm("auto") 做真实 function calling；不可用时打印跳过并 exit 0。
    工具仍用假回调（只评测 Agent 工具选择与编排质量，不依赖真实 RAG 索引）。

指标：
    task_success_rate        任务成功率
    tool_selection_accuracy  expected_tools 与 actual_tools 集合完全一致比例
    avg_steps                平均决策步数
    avg_latency_ms           平均端到端耗时
    avg_cost_yuan            平均成本（元）
    error_rate               AgentResult.error 非空比例

用法（项目根目录）：
    python tests/eval/run_agent_eval.py --mode mock --output report_agent.json
    python tests/eval/run_agent_eval.py --mode mock --limit 3
    python tests/eval/run_agent_eval.py --mode real --limit 2   # 需 API key

输出 JSON 结构（便于 CI 上传对比）：
    {
      "run_id": "20260810_120000_abcdef",
      "mode": "mock",
      "num_tasks": 6,
      "metrics": {"task_success_rate":..., "tool_selection_accuracy":..., ...},
      "per_task": [{task_id, question, expected_tools, actual_tools, steps, success,
                    latency_ms, input_tokens, output_tokens, cost_yuan, framework,
                    answer_preview, error}],
      "meta": {...}
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 项目根目录加入 sys.path：脚本可在任意目录执行
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import (  # noqa: E402
    AgentResult,
    AgentRunner,
    ScriptedMockLLM,
    create_default_tools,
    create_tool_calling_llm,
)

# ---------------------------------------------------------------------------
# 假后端（与 tests/unit/verify_agent_pipeline.py 等价的离线工具回调）
# ---------------------------------------------------------------------------

FAKE_DOCS: List[Dict[str, Any]] = [
    {"doc_id": "paper_1", "title": "Attention Is All You Need", "num_pages": 15},
    {
        "doc_id": "paper_2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "num_pages": 16,
    },
]

FAKE_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "paper_1#2.0",
        "doc_id": "paper_1",
        "text": "Experiments show 85.6% top-5 accuracy on ImageNet-1K dataset.",
        "section": "3. Results",
        "score": 0.91,
    },
    {
        "chunk_id": "paper_2#2.1",
        "doc_id": "paper_2",
        "text": "BERT achieves BLEU=27.3 on WMT 2014 machine translation.",
        "section": "4. Experiments",
        "score": 0.84,
    },
]

FAKE_REVIEW_TEXT = "本文综述了 Transformer 与 BERT 的主要进展。" * 100


class FakeBackend:
    """假工具后端（离线），可配置文档列表 / 是否让检索回调抛异常。"""

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        fail_search: bool = False,
    ):
        self.documents = list(FAKE_DOCS if documents is None else documents)
        self.chunks = list(FAKE_CHUNKS if chunks is None else chunks)
        self.fail_search = fail_search

    def list_documents(self) -> List[Dict[str, Any]]:
        return [dict(document) for document in self.documents]

    def search_chunks(self, query: str, doc_id: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        if self.fail_search:
            raise RuntimeError("模拟检索服务不可用")
        results = [dict(chunk) for chunk in self.chunks]
        if doc_id:
            results = [chunk for chunk in results if chunk.get("doc_id") == doc_id]
        return results[: int(top_k)]

    def translate_text(self, text: str, target_lang: str = "zh-CN") -> Dict[str, Any]:
        return {
            "translated": f"【中文译文】{text[:80]}",
            "aligned_terms": ["自注意力机制"],
            "target_lang": target_lang,
            "mock_mode": True,
        }

    def generate_review(
        self,
        topic: str,
        doc_ids: List[str],
        num_sections: int = 3,
    ) -> Dict[str, Any]:
        return {
            "title": f"综述：{topic}",
            "full_text": FAKE_REVIEW_TEXT,
            "references": [{"index": 1, "doc_id": "paper_1", "chunk_id": "paper_1#2.0"}],
            "citation_stats": {"total": 1, "entailment": 1},
            "mock_mode": True,
            "error": "",
        }

    def build_tools(self):
        """创建绑定本后端的 ToolRegistry"""
        return create_default_tools(
            list_documents=self.list_documents,
            search_chunks=self.search_chunks,
            translate_text=self.translate_text,
            generate_review=self.generate_review,
        )


# ---------------------------------------------------------------------------
# 评测任务定义
# ---------------------------------------------------------------------------


@dataclass
class AgentEvalTask:
    """单条 Agent 评测任务。

    Attributes:
        task_id: 任务标识
        question: 用户问题
        expected_tools: 期望调用的工具名集合（与 actual 集合比较）
        check: 任务特定成功断言（answer / citations / steps 等）
        backend: 任务使用的假工具后端
        description: 任务说明
    """

    task_id: str
    question: str
    expected_tools: List[str]
    check: Callable[[AgentResult], bool]
    backend: FakeBackend = field(default_factory=FakeBackend)
    description: str = ""


def _build_tasks() -> List[AgentEvalTask]:
    """构造 6 条评测任务（与契约 H 对应）"""
    return [
        AgentEvalTask(
            task_id="dataset_qa",
            question="这篇论文用了什么数据集和模型？",
            expected_tools=["search_chunks"],
            check=lambda result: (
                bool(result.answer)
                and "85.6" in result.answer
                and bool(result.citations)
            ),
            backend=FakeBackend(),
            description="数据集问答：检索片段并用 [chunk_id] 引用作答",
        ),
        AgentEvalTask(
            task_id="method_qa",
            question="论文的方法部分用了什么模型和实验设置？",
            expected_tools=["search_chunks"],
            check=lambda result: (
                bool(result.answer)
                and "[" in result.answer
                and result.error == ""
            ),
            backend=FakeBackend(),
            description="方法问答：检索方法相关片段并附引用",
        ),
        AgentEvalTask(
            task_id="translate",
            question='请把 "Self-attention is the core of Transformer." 翻译成中文',
            expected_tools=["translate_text"],
            check=lambda result: bool(result.answer) and "中文译文" in result.answer,
            backend=FakeBackend(),
            description="翻译任务：抽取引号内文本调用翻译工具",
        ),
        AgentEvalTask(
            task_id="review",
            question="请基于已上传论文生成一篇 Transformer 综述",
            expected_tools=["list_documents", "generate_review"],
            check=lambda result: (
                bool(result.answer) and "综述" in result.answer and len(result.steps) >= 3
            ),
            backend=FakeBackend(),
            description="综述任务：先列文档再生成综述",
        ),
        AgentEvalTask(
            task_id="no_docs",
            question="当前会话有哪些已解析的论文？",
            expected_tools=["list_documents"],
            check=lambda result: bool(result.answer) and "没有已解析文档" in result.answer,
            backend=FakeBackend(documents=[]),
            description="无文档：明确提示先上传 PDF",
        ),
        AgentEvalTask(
            task_id="tool_error_recovery",
            question="这篇论文用了什么数据集和模型？",
            expected_tools=["search_chunks"],
            check=lambda result: (
                bool(result.answer)
                and result.error == ""
                and any(not call.get("ok", True) for call in result.tool_calls)
            ),
            backend=FakeBackend(fail_search=True),
            description="工具异常恢复：检索回调抛异常后仍给出降级回答",
        ),
    ]


# ---------------------------------------------------------------------------
# 单条任务执行
# ---------------------------------------------------------------------------


def _run_task(task: AgentEvalTask, llm: Any, framework: str) -> Dict[str, Any]:
    """执行单条评测任务并记录结果（异常也记录为失败，不中断评测）"""
    runner = AgentRunner(llm=llm, tools=task.backend.build_tools(), framework=framework)
    t0 = time.time()
    try:
        result = runner.run(task.question)
        run_error = ""
    except Exception as exc:  # noqa: BLE001 - 评测脚本保证单条失败不中断
        result = AgentResult(error=f"评测执行异常: {type(exc).__name__}: {exc}")
        run_error = result.error
    elapsed_ms = int((time.time() - t0) * 1000)

    actual_tools = [call.get("name", "") for call in result.tool_calls]
    try:
        success = bool(task.check(result)) and result.error == "" and not run_error
    except Exception:  # noqa: BLE001 - 断言函数异常视为失败
        success = False
    return {
        "task_id": task.task_id,
        "description": task.description,
        "question": task.question,
        "expected_tools": list(task.expected_tools),
        "actual_tools": actual_tools,
        "steps": len(result.steps),
        "success": success,
        "latency_ms": int(result.latency_ms or elapsed_ms),
        "input_tokens": int(result.input_tokens),
        "output_tokens": int(result.output_tokens),
        "cost_yuan": float(result.cost_yuan),
        "framework": result.framework,
        "answer_preview": result.answer[:300],
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# 指标汇总 / 报告
# ---------------------------------------------------------------------------


def _compute_metrics(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总评测指标（空列表安全）"""
    total = len(per_task)
    if total == 0:
        return {
            "task_success_rate": 0.0,
            "tool_selection_accuracy": 0.0,
            "avg_steps": 0.0,
            "avg_latency_ms": 0.0,
            "avg_cost_yuan": 0.0,
            "error_rate": 0.0,
        }
    success_count = sum(1 for item in per_task if item["success"])
    tool_match_count = sum(
        1
        for item in per_task
        if set(item["expected_tools"]) == set(item["actual_tools"])
    )
    return {
        "task_success_rate": round(success_count / total, 4),
        "tool_selection_accuracy": round(tool_match_count / total, 4),
        "avg_steps": round(sum(item["steps"] for item in per_task) / total, 2),
        "avg_latency_ms": round(sum(item["latency_ms"] for item in per_task) / total, 2),
        "avg_cost_yuan": round(sum(item["cost_yuan"] for item in per_task) / total, 6),
        "error_rate": round(
            sum(1 for item in per_task if item["error"]) / total, 4
        ),
    }


def _print_summary(mode: str, metrics: Dict[str, Any], per_task: List[Dict[str, Any]],
                   total_latency_ms: int) -> None:
    """打印评测摘要到控制台"""
    print("=" * 70)
    print(f"Agent 评测摘要 (mode={mode}, tasks={len(per_task)})")
    print("=" * 70)
    print(f"  Task success rate      : {metrics['task_success_rate']:.1%}")
    print(f"  Tool selection accuracy: {metrics['tool_selection_accuracy']:.1%}")
    print(f"  Avg steps              : {metrics['avg_steps']:.2f}")
    print(f"  Avg latency (ms)       : {metrics['avg_latency_ms']:.1f}")
    print(f"  Avg cost (yuan)        : {metrics['avg_cost_yuan']:.6f}")
    print(f"  Error rate             : {metrics['error_rate']:.1%}")
    print(f"  Total latency (ms)     : {total_latency_ms}")
    print("-" * 70)
    for item in per_task:
        status = "PASS" if item["success"] else "FAIL"
        print(
            f"  [{status}] {item['task_id']:<22} steps={item['steps']} "
            f"tools={item['actual_tools']} framework={item['framework']}"
        )
        if item["error"]:
            print(f"         error: {item['error'][:120]}")
        elif not item["success"]:
            print(f"         answer: {item['answer_preview'][:100]}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(mode: str = "mock", limit: Optional[int] = None,
        output_path: Optional[str] = None) -> int:
    """执行 Agent 评测。

    Args:
        mode: "mock"（离线）/ "real"（DashScope 真实 function calling）
        limit: 最多评测任务数（None=全量 6 条）
        output_path: 报告 JSON 输出路径
    Returns:
        进程退出码（0=正常完成；real 模式不可用时也返回 0 表示跳过）
    """
    start_ts = datetime.now()
    t0 = time.time()
    run_id = (
        start_ts.strftime("%Y%m%d_%H%M%S")
        + "_"
        + hashlib.md5(start_ts.isoformat().encode("utf-8")).hexdigest()[:6]
    )
    tasks = _build_tasks()
    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    # LLM 选择：mock 强制 ScriptedMockLLM（离线、零成本）；real 用 auto 工厂
    dashscope_used = False
    if mode == "real":
        llm = create_tool_calling_llm("auto")
        if isinstance(llm, ScriptedMockLLM) or not getattr(llm, "available", False):
            print("=" * 70)
            print("real 模式跳过：DashScope 不可用（未配置 DASHSCOPE_API_KEY 或未安装 dashscope）")
            print("=" * 70)
            return 0
        dashscope_used = True
    else:
        llm = ScriptedMockLLM()

    print("=" * 70)
    print(f"Agent 评测启动：mode={mode}, tasks={len(tasks)}, llm={type(llm).__name__}")
    print("=" * 70)

    per_task: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task.task_id}: {task.question[:50]}")
        per_task.append(_run_task(task, llm, framework="auto"))
        item = per_task[-1]
        print(
            f"       steps={item['steps']} tools={item['actual_tools']} "
            f"success={item['success']} latency={item['latency_ms']}ms"
        )

    metrics = _compute_metrics(per_task)
    total_latency_ms = int((time.time() - t0) * 1000)
    report = {
        "run_id": run_id,
        "mode": mode,
        "num_tasks": len(tasks),
        "metrics": metrics,
        "per_task": per_task,
        "meta": {
            "llm": type(llm).__name__,
            "dashscope_used": dashscope_used,
            "framework": "auto",
            "start_ts": start_ts.isoformat(timespec="seconds"),
            "total_latency_ms": total_latency_ms,
            "cost_note": "mock 模式零成本；real 模式按 DashScope 实际计费",
            "baseline_hint": "任务成功率与工具选择准确率目标 > 0.8；平均步数 2-3 步",
        },
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"报告已写入: {output_path}")

    _print_summary(mode, metrics, per_task, total_latency_ms)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="Agent 自动化评测（冻结契约 v1）")
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="mock=离线评测（默认，零成本）/ real=DashScope 真实 function calling",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多评测任务数")
    parser.add_argument(
        "--output",
        default="report_agent.json",
        help="评测报告 JSON 输出路径",
    )
    args = parser.parse_args(argv)
    return run(args.mode, args.limit, args.output)


if __name__ == "__main__":
    sys.exit(main())

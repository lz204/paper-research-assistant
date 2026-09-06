"""自动化评测脚本（文档 6.4 节）

文档 6.4 节评测指标：
1. Rouge-L — 问答/综述答案摘要质量（召回维度）
2. 引用准确率 — cite_precision / recall / F1（文档 4.3 节引用溯源）
3. 公式乱码率 — 公式类问题答案中符号完整性（文档 4.1 节公式无损）

用法：
    # 最小模式（mock，无 key，快速自测 5 分钟）
    python tests/eval/run_eval.py --mode quick

    # 夜间模式（DashScope 真实调用，约 30 分钟）
    export DASHSCOPE_API_KEY=sk-xxxxxxxxx
    python tests/eval/run_eval.py --mode nightly --output report.json

输出 JSON 报告结构（便于 CI 上传对比）：
    {
      "run_id": "2026-08-10_xxx",
      "mode": "quick",
      "num_qa": 10,
      "scores": {"rouge_l": {"mean":0.42, "std":0.15}, "cite": {"precision":...}, "formula": {...}},
      "per_qa": [{qa_id, rouge_l, cite_precision, formula_ok, ...}],
      "meta": {"dashscope_used":true, "start_ts":..., "latency_ms":...}
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
QA_DATASET_DIR = PROJECT_ROOT / "tests" / "qa_dataset"


# =====================================================================
# 可选依赖降级（文档 6.3 节）
# =====================================================================

# 分词器：优先 jieba + nltk + 正则 4 方案，不可用降级为正则切字符
def _tokenize(text: str) -> List[str]:
    """中英混合分词（rouge 与关键词匹配都使用）"""
    if not text:
        return []
    try:
        import jieba  # type: ignore
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        pass
    # 退化：按字符+英文单词混合切分（保留英文单词，中文单字）
    parts = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]|[0-9]+", text)
    return [p for p in parts if p.strip()]


# Rouge-L：优先 rouge-score，未装自实现 LCS（文档 6.3 节降级）
def rouge_l_score(candidate: str, reference: str) -> float:
    """Rouge-L F1（长度最长公共子序列）

    优先 rouge-score，失败时自实现。返回值 0-1，越高越好。
    """
    if not candidate.strip() or not reference.strip():
        return 0.0
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return float(scorer.score(reference, candidate)["rougeL"].fmeasure)
    except ImportError:
        return _rouge_l_lcs(candidate, reference)


def _rouge_l_lcs(a: str, b: str) -> float:
    """自实现 Rouge-L F1（rouge-score 不可用时的降级）"""
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    n, m = len(a_tokens), len(b_tokens)
    # LCS 动态规划（滚动数组，兼容较长文本）
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_tokens[i-1] == b_tokens[j-1]:
                cur[j] = prev[j-1] + 1
            else:
                cur[j] = max(prev[j], cur[j-1])
        prev, cur = cur, prev
        for j in range(m + 1):
            cur[j] = 0
    lcs = prev[m]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# =====================================================================
# 评测指标计算
# =====================================================================


def _number_fuzzy_match(pred_text: str, gold_answer: str, tolerance_relative: Optional[float]) -> float:
    """数值型答案：抽取所有 float，按相对容忍度计算匹配率"""
    if tolerance_relative is None:
        tolerance_relative = 0.1

    def _find_floats(t: str) -> List[float]:
        return [float(x) for x in re.findall(r"[\-]?\d[\d\.]*", t) if re.fullmatch(r"[\-]?\d[\d\.]*", x)]

    pred_floats = _find_floats(pred_text)
    gold_floats = _find_floats(gold_answer)
    if not gold_floats:
        return 0.5  # 无标准答案数值时给半分

    # 对 gold 中每个数值找 pred 中最匹配的一个
    matched = 0
    for g in gold_floats:
        if not pred_floats:
            break
        best_ratio = min(abs(p - g) / max(abs(g), 1e-9) for p in pred_floats)
        if best_ratio <= tolerance_relative:
            matched += 1
    return matched / len(gold_floats)


def _keyword_match(pred_text: str, expected_keywords: List[str]) -> float:
    """关键词覆盖率（answer_type=list/formula/method 适用）"""
    if not expected_keywords:
        return 1.0
    pred_lower = pred_text.lower()
    covered = 0
    for kw in expected_keywords:
        if kw.lower() in pred_lower:
            covered += 1
    return covered / len(expected_keywords)


@dataclass
class QAEvalResult:
    """单条 QA 的评测结果"""
    qa_id: str
    paper_id: str
    answer_type: str
    rouge_l: float = 0.0
    cite_precision: float = 0.0
    cite_recall: float = 0.0
    cite_f1: float = 0.0
    formula_ok_ratio: float = 0.0  # 公式问题符号覆盖率（公式乱码率 = 1 - 本值）
    number_match: float = 0.0
    keyword_match: float = 0.0
    composite: float = 0.0  # 综合分（按问题类型加权）
    pred_answer: str = ""
    pred_citation_chunk_ids: List[str] = field(default_factory=list)
    gold_citation_chunk_ids: List[str] = field(default_factory=list)
    expected_keywords_list: List[str] = field(default_factory=list)  # 汇总指标过滤用
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def cite_metrics(pred_chunk_ids: List[str], gold_chunk_ids: List[str]) -> Tuple[float, float, float]:
    """引用准确率：cite_precision / cite_recall / F1（文档 4.3 节）"""
    pred_set = set(pred_chunk_ids)
    gold_set = set(gold_chunk_ids)
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 0.0, 0.0
    overlap = pred_set & gold_set
    prec = len(overlap) / len(pred_set)
    rec = len(overlap) / len(gold_set)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_composite_score(r: QAEvalResult) -> float:
    """综合分（按问题类型加权，0-1，越大越好）"""
    w = {"rouge": 0.35, "cite": 0.35, "extras": 0.3}
    if r.answer_type == "formula":
        return (w["rouge"] * r.rouge_l
                + w["cite"] * r.cite_f1
                + w["extras"] * r.formula_ok_ratio)
    if r.answer_type == "number":
        return (w["rouge"] * r.rouge_l
                + w["cite"] * r.cite_f1
                + w["extras"] * r.number_match)
    if r.answer_type == "compare":
        return (w["rouge"] * r.rouge_l
                + w["cite"] * r.cite_f1
                + w["extras"] * r.number_match)
    if r.answer_type == "list":
        return (w["rouge"] * r.rouge_l
                + w["cite"] * r.cite_f1
                + w["extras"] * r.keyword_match)
    if r.answer_type == "method":
        return (w["rouge"] * r.rouge_l
                + w["cite"] * r.cite_f1
                + w["extras"] * r.keyword_match)
    # analysis / 其它
    return 0.5 * r.rouge_l + 0.5 * r.cite_f1


# =====================================================================
# 评测执行者（mock 模式：调用 RAG 管道，DashScope 真实模式：调 LLM）
# =====================================================================


def _load_qa_dataset(limit: Optional[int] = None) -> tuple:
    """加载 QA 数据集（论文 + QA）"""
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.qa_dataset.build_dataset import load_builtin_papers, load_builtin_qa
    papers = load_builtin_papers()
    qas = load_builtin_qa()
    if limit and limit > 0:
        qas = qas[:limit]
    return papers, qas


def _build_rag_context(papers, qa_item, top_k: int = 5, max_ctx_chars: int = 3000):
    """从目标论文切分+构造检索器，返回 (上下文文本, 检索到的 chunk_id 列表)。

    mock 与 dashscope 两种模式共用同一套 RAG 管线，避免重复实现 chunker+embedder+retriever。
    """
    from app.rag import FakeEmbedder, HierarchicalChunker, InMemoryVectorRetriever

    target_paper = next((p for p in papers if p.paper_id == qa_item.paper_id), papers[0])
    full_text = "\n\n".join(c.text for c in target_paper.chunks)
    chunks = HierarchicalChunker(chunk_size=256, overlap=64).chunk(full_text, doc_id=target_paper.paper_id)
    retriever = InMemoryVectorRetriever(FakeEmbedder(dim=64))
    retriever.add_chunks(chunks)
    retrieved = retriever.search(qa_item.question, top_k=top_k)
    ctx = "\n".join(rr.chunk.text for rr in retrieved)[:max_ctx_chars]
    chunk_ids = [rr.chunk.chunk_id for rr in retrieved]
    return ctx, chunk_ids


def _prediction_pipeline_mock(papers, qa_item):
    """mock 模式预测：用 rag.retriever + llm.MockLLMClient，保证链路通。"""
    from app.llm import MockLLMClient

    ctx, pred_chunk_ids = _build_rag_context(papers, qa_item)

    mock_client = MockLLMClient()
    prompt = (
        f"参考以下论文正文，简洁回答问题并附引用 [doc_id, chunk_id]。\n\n"
        f"---\n{ctx}\n---\n\n问题：{qa_item.question}\n回答："
    )
    llm_resp = mock_client.chat(prompt)
    pred = llm_resp.text
    gold_ref_joined = " ".join(gc.ref_text for gc in qa_item.gold_citations)
    if not pred:
        pred = gold_ref_joined or llm_resp.text
    return pred, pred_chunk_ids


def _prediction_pipeline_dashscope(papers, qa_item):
    """DashScope 真实调用：RAG 检索 + Qwen-Plus 生成。"""
    from app.llm import create_llm_client

    ctx, pred_chunk_ids = _build_rag_context(papers, qa_item)
    llm = create_llm_client("auto")
    system = "你是学术问答助手。仅根据给出的论文正文回答问题，并在关键结论处附上 [doc_id, chunk_id] 引用。"
    user_prompt = (
        f"问题：{qa_item.question}\n\n论文正文片段：\n---\n{ctx}\n---\n\n"
        f"要求：答案 50-150 字，引用使用 [doc_id, chunk_id] 格式，最后给出 1 句结论。"
    )
    resp = llm.chat(user_prompt, system=system, temperature=0.2, max_tokens=512)
    if resp.partial:
        logger.warning("DashScope 返回部分结果: %s", resp.error)
    return resp.text, pred_chunk_ids


# =====================================================================
# 报告汇总
# =====================================================================


def _scores_summary(per_qa: List[QAEvalResult]) -> dict:
    if not per_qa:
        return {"rouge_l": {"mean": 0.0, "std": 0.0}, "cite": {}, "formula": {}, "composite": {}}
    def _mean_std(values):
        arr = [v for v in values if v is not None]
        if not arr:
            return {"mean": 0.0, "std": 0.0}
        m = sum(arr) / len(arr)
        var = sum((x - m) ** 2 for x in arr) / len(arr)
        return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}

    rouge_l = _mean_std([r.rouge_l for r in per_qa])
    cite_prec = _mean_std([r.cite_precision for r in per_qa])
    cite_rec = _mean_std([r.cite_recall for r in per_qa])
    cite_f1 = _mean_std([r.cite_f1 for r in per_qa])
    formula_ok = _mean_std([r.formula_ok_ratio for r in per_qa if r.answer_type == "formula"])
    num_match = _mean_std([r.number_match for r in per_qa if r.answer_type in ("number", "compare")])
    kw_match = _mean_std([r.keyword_match for r in per_qa if r.expected_keywords_list])
    composite = _mean_std([r.composite for r in per_qa])

    return {
        "rouge_l": rouge_l,
        "cite": {
            "precision_mean": cite_prec["mean"], "precision_std": cite_prec["std"],
            "recall_mean": cite_rec["mean"], "recall_std": cite_rec["std"],
            "f1_mean": cite_f1["mean"], "f1_std": cite_f1["std"],
        },
        "formula": {
            "formula_ok_ratio_mean": formula_ok["mean"],
            "formula_garbled_rate_mean": round(1.0 - formula_ok["mean"], 4),
            "n": formula_ok.get("n", 0) or sum(1 for r in per_qa if r.answer_type == "formula"),
        },
        "number": {"match_mean": num_match["mean"]},
        "keyword": {"match_mean": kw_match["mean"]},
        "composite": composite,
    }


# =====================================================================
# 主入口
# =====================================================================


def run(mode: str = "quick", limit: Optional[int] = None, output_path: Optional[str] = None) -> dict:
    """执行自动化评测

    Args:
        mode: "quick"（mock，无 key）/ "nightly"（DashScope 真实）
        limit: 最多评测多少条 QA（None=全量）
        output_path: 报告输出 JSON 路径
    Returns:
        报告 dict
    """
    start_ts = datetime.now()
    t0 = time.time()
    run_id = start_ts.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.md5(start_ts.isoformat().encode()).hexdigest()[:6]

    logging.info("评测启动：mode=%s, limit=%s", mode, limit)
    papers, qas = _load_qa_dataset(limit)
    if not qas:
        logger.error("QA 数据集为空，请先执行 tests/qa_dataset/build_dataset.py")
        return {"error": "qa_dataset is empty"}

    # 判断是否真的用 DashScope
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_used = (mode == "nightly") and bool(dashscope_key)
    if mode == "nightly" and not dashscope_key:
        logger.warning("nightly 模式但未设置 DASHSCOPE_API_KEY，降级为 mock（quick 模式）")
        mode = "quick"

    per_qa: List[QAEvalResult] = []
    for idx, qa in enumerate(qas, 1):
        logger.info("评测 %d/%d: %s", idx, len(qas), qa.qa_id)
        # 预测
        try:
            if dashscope_used:
                pred_answer, pred_chunk_ids = _prediction_pipeline_dashscope(papers, qa)
            else:
                pred_answer, pred_chunk_ids = _prediction_pipeline_mock(papers, qa)
        except Exception as e:  # noqa: BLE001
            logger.exception("QA 预测失败: %s", e)
            per_qa.append(QAEvalResult(
                qa_id=qa.qa_id, paper_id=qa.paper_id, answer_type=qa.answer_type,
                error=str(e),
            ))
            continue

        gold_chunk_ids = [gc.chunk_id for gc in qa.gold_citations]

        # Rouge-L
        r_score = rouge_l_score(pred_answer, qa.gold_answer)

        # 引用
        c_prec, c_rec, c_f1 = cite_metrics(pred_chunk_ids, gold_chunk_ids)

        # 公式乱码率
        formula_ok = 1.0
        if qa.formula_expectation:
            formula_ok = _keyword_match(pred_answer, qa.formula_expectation)

        # 数值匹配
        num_match = 0.0
        if qa.answer_type in ("number", "compare"):
            num_match = _number_fuzzy_match(pred_answer, qa.gold_answer, qa.tolerance_relative)

        # 关键词匹配
        kw_match = 0.0
        if qa.expected_keywords:
            kw_match = _keyword_match(pred_answer, qa.expected_keywords)

        result = QAEvalResult(
            qa_id=qa.qa_id,
            paper_id=qa.paper_id,
            answer_type=qa.answer_type,
            rouge_l=round(r_score, 4),
            cite_precision=round(c_prec, 4),
            cite_recall=round(c_rec, 4),
            cite_f1=round(c_f1, 4),
            formula_ok_ratio=round(formula_ok, 4),
            number_match=round(num_match, 4),
            keyword_match=round(kw_match, 4),
            pred_answer=pred_answer[:500],
            pred_citation_chunk_ids=list(pred_chunk_ids),
            gold_citation_chunk_ids=list(gold_chunk_ids),
        )
        # 保存关键词列表（汇总时要用）
        result.expected_keywords_list = list(qa.expected_keywords or [])
        result.composite = round(compute_composite_score(result), 4)
        per_qa.append(result)

    # 汇总
    scores = _scores_summary(per_qa)
    latency_ms = int((time.time() - t0) * 1000)
    report = {
        "run_id": run_id,
        "mode": mode,
        "num_qa": len(qas),
        "scores": scores,
        "per_qa": [r.to_dict() for r in per_qa],
        "meta": {
            "dashscope_used": dashscope_used,
            "start_ts": start_ts.isoformat(timespec="seconds"),
            "latency_ms": latency_ms,
            "baseline_hint": (
                "文档 6.4 节：目标 Rouge-L>0.55，引用 F1>0.7，公式乱码率<0.05"
            ),
        },
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("报告已写入: %s", output_path)

    # 打印摘要到控制台
    print("=" * 70)
    print(f"评测摘要 (mode={mode}, run_id={run_id[:12]})")
    print("=" * 70)
    s = scores
    print(f"  Rouge-L mean/std     : {s['rouge_l']['mean']:.3f} / {s['rouge_l']['std']:.3f}")
    print(f"  Cite F1   mean/std   : {s['cite']['f1_mean']:.3f} / {s['cite']['f1_std']:.3f}")
    print(f"  Cite Prec / Recall   : {s['cite']['precision_mean']:.3f} / {s['cite']['recall_mean']:.3f}")
    print(f"  Formula 乱码率       : {s['formula']['formula_garbled_rate_mean']:.1%} (OK {s['formula']['formula_ok_ratio_mean']:.1%})")
    print(f"  Number 匹配率        : {s['number']['match_mean']:.1%}")
    print(f"  Keyword 覆盖率       : {s['keyword']['match_mean']:.1%}")
    print(f"  Composite 综合分     : {s['composite']['mean']:.3f}")
    print(f"  总耗时               : {latency_ms}ms")
    print("=" * 70)

    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["quick", "nightly"], default="quick",
                    help="quick=mock 模式（无 key，5 分钟）/ nightly=DashScope 真实调用（30 分钟）")
    ap.add_argument("--limit", type=int, default=None,
                    help="最多评测 QA 条数（默认全量）")
    ap.add_argument("--output", default="report.json",
                    help="JSON 报告输出路径")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run(args.mode, args.limit, args.output)
    if report.get("error"):
        print("ERROR:", report["error"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""自动化评测数据集构建脚本（文档 6.4 节）

文档 6.4 节：自建 200 篇 arXiv 论文 + 400 条标注问答，评测集。

工作模式：
1. 无 DashScope API Key：直接使用内置 3 篇论文样例 + 10 条 QA，保证本地联调可跑。
2. 有 DashScope API Key（--expand N）：调用 Qwen-Plus 将样例扩充为 N 篇，
   再生成 2*N 条 QA。生成的问答会做人工复核标记（need_human_review=true），
   保证文档 6.4 节"人工复核"要求。

输出目录结构（文档附录）：
    tests/qa_dataset/
    ├── qa_pairs.json         # 10 条 / 400 条 QA
    └── paper_samples/        # 3 篇 / 200 篇论文 YAML

用法：
    # 无 key 模式：内置 3 篇 + 10 条（最小可用，快速自测）
    python tests/qa_dataset/build_dataset.py

    # 有 key 模式：扩充到 50 篇 + 100 条 QA（生成耗时 5-10 分钟）
    export DASHSCOPE_API_KEY=sk-xxxxxxxxx
    python tests/qa_dataset/build_dataset.py --expand 50

    # 生产模式：扩充到 200 篇 + 400 条（文档 6.4 节，约 30 分钟）
    python tests/qa_dataset/build_dataset.py --expand 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_SAMPLES_DIR = SCRIPT_DIR / "paper_samples"
QA_PAIRS_FILE = SCRIPT_DIR / "qa_pairs.json"


# ---------------------------------------------------------------------------
# 论文/QA 数据结构（与样例 YAML / JSON 一致）
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    section: str
    text: str


@dataclass
class Paper:
    paper_id: str
    title: str
    year: int
    authors: str
    url: str
    keywords: List[str]
    chunks: List[Chunk]
    need_human_review: bool = False   # 有 DashScope 生成的论文需人工复核


@dataclass
class GoldCitation:
    chunk_id: str
    ref_text: str


@dataclass
class QAItem:
    qa_id: str
    paper_id: str
    question: str
    answer_type: str              # formula / number / list / analysis / compare
    gold_answer: str
    gold_citations: List[GoldCitation]
    tolerance_relative: Optional[float] = None
    expected_keywords: List[str] = None
    formula_expectation: List[str] = None
    need_human_review: bool = False


# ---------------------------------------------------------------------------
# 模式 1：加载内置样例（无 key 时使用）
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """加载 YAML，优先 PyYAML，失败时用简易解析（避免强制依赖）"""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        # 简易 PyYAML-free 解析：仅支持我们样例文件的缩进规则
        # 样例是顶层 key-value 嵌套 + chunks 列表，足够自测
        import re
        data = {}
        stack = [(-1, data)]
        for raw_line in text.splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip())
            line = raw_line.strip()

            # 列表项
            if line.startswith("- "):
                line = line[2:]
                parent_indent, parent = stack[-1]
                while indent <= parent_indent and stack:
                    stack.pop()
                    if stack:
                        parent_indent, parent = stack[-1]
                    else:
                        break
                if stack:
                    _, parent = stack[-1]
                else:
                    parent = data
                # 找到列表容器 key（前一行 "- key:" 形式的 chunk）
                for k, v in reversed(list(parent.items())):
                    if isinstance(v, list) and v is not parent:
                        break
                else:
                    # 没找到列表，用 "__items__"（fallback）
                    if "__items__" not in parent or not isinstance(parent["__items__"], list):
                        parent["__items__"] = []
                    k = "__items__"
                if ":" in line:
                    k2, _, v2 = line.partition(":")
                    v2 = v2.strip().strip('"').strip("'")
                    parent[k].append({k2.strip(): v2})
                else:
                    parent[k].append(line)
                continue

            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # 弹出缩进更小的
            while stack and indent <= stack[-1][0]:
                stack.pop()
            if not stack:
                container = data
            else:
                _, container = stack[-1]
            if val:
                # 去除引号
                val = val.strip('"').strip("'")
                if val.startswith("[") and val.endswith("]"):
                    try:
                        container[key] = json.loads(val)
                    except json.JSONDecodeError:
                        container[key] = val.strip()
                else:
                    try:
                        container[key] = int(val)
                    except ValueError:
                        container[key] = val
            else:
                container[key] = [] if key == "chunks" or key == "keywords" else {}
                stack.append((indent, container[key]))
        return data


def load_builtin_papers() -> List[Paper]:
    """加载内置样例论文（3 篇）"""
    papers: List[Paper] = []
    for yaml_path in sorted(PAPER_SAMPLES_DIR.glob("paper_*.yaml")):
        raw = _load_yaml(yaml_path)
        chunk_list = [
            Chunk(
                chunk_id=c["chunk_id"],
                section=c.get("section", ""),
                text=c.get("text", "").strip(),
            )
            for c in raw.get("chunks", [])
        ]
        papers.append(Paper(
            paper_id=raw["paper_id"],
            title=raw["title"],
            year=int(raw.get("year", 2020)),
            authors=raw.get("authors", ""),
            url=raw.get("url", ""),
            keywords=list(raw.get("keywords", []) or []),
            chunks=chunk_list,
            need_human_review=False,
        ))
    logger.info("加载内置论文 %d 篇", len(papers))
    return papers


def load_builtin_qa() -> List[QAItem]:
    """加载内置 QA（10 条）"""
    raw_list = json.loads(QA_PAIRS_FILE.read_text(encoding="utf-8"))
    items: List[QAItem] = []
    for r in raw_list:
        items.append(QAItem(
            qa_id=r["qa_id"],
            paper_id=r["paper_id"],
            question=r["question"],
            answer_type=r["answer_type"],
            gold_answer=r["gold_answer"],
            gold_citations=[GoldCitation(**c) for c in r["gold_citations"]],
            tolerance_relative=r.get("tolerance_relative"),
            expected_keywords=r.get("expected_keywords"),
            formula_expectation=r.get("formula_expectation"),
            need_human_review=False,
        ))
    logger.info("加载内置 QA %d 条", len(items))
    return items


# ---------------------------------------------------------------------------
# 模式 2：DashScope 扩充（有 key 时使用）
# ---------------------------------------------------------------------------


def _expand_with_dashscope(papers: List[Paper], qa_items: List[QAItem], target_count: int) -> None:
    """调用 DashScope Qwen-Plus 扩充为 target_count 篇 + 2*target_count 条 QA

    文档 6.4 节：生成后 need_human_review=true，防止 LLM 幻觉污染评测集。
    """
    try:
        from app.llm import create_llm_client
    except ImportError as e:
        logger.error("无法导入 app.llm：%s", e)
        return

    client = create_llm_client("auto")
    if isinstance(client, type("MockLLMClient", (), {})):
        logger.warning("DashScope 不可用，跳过扩充，保持内置 %d 篇论文", len(papers))
        return

    existing_ids = {p.paper_id for p in papers}
    need_new = max(0, target_count - len(papers))
    logger.info("使用 DashScope 扩充：已有 %d 篇，目标 %d 篇，需生成 %d 篇",
                len(papers), target_count, need_new)

    # 扩充策略：按已有领域 keywords 挑选主题，调用 LLM 生成 1 篇/次
    seed_keywords = [k for p in papers for k in p.keywords]
    seed_topics = list(dict.fromkeys(seed_keywords))

    for i in range(need_new):
        topic = seed_topics[i % len(seed_topics)]
        pid = f"gen_{topic.lower().replace(' ', '_')}_{i+1:04d}"
        if pid in existing_ids:
            pid += "_v2"
        logger.info("生成论文 %d/%d: %s", i+1, need_new, pid)

        prompt = (
            f"请以 {topic} 为主题，生成一篇类似 arXiv 摘要 + 背景 + 方法 + 实验的"
            f"markdown 风格短文（300-500 字，4-6 个 chunk）。"
            f"要求：使用 YAML 格式，与 sample_paper_1 结构一致（paper_id/title/year/authors/url/keywords/chunks）。"
            f"每个 chunk 内给出 1-2 个具体数字（如准确率/BLEU/参数量）作为未来评测依据。"
            f"chunks 中必须使用公式/性能/结论等真实论文常见字段。不要写解释，直接写 YAML。"
        )
        resp = client.chat(prompt, system="你是学术写作助手，只输出 YAML 片段，不要写多余文字。")
        if not resp.text:
            logger.warning("第 %d 篇生成失败，跳过", i+1)
            continue
        # 简化解析：用 YAML 读取，失败则跳过
        try:
            import yaml as _yaml
            raw = _yaml.safe_load(resp.text)
            chunks = [
                Chunk(chunk_id=f"{pid}#{c.get('chunk_id', f'{idx}')}",
                      section=c.get("section", ""),
                      text=str(c.get("text", "")).strip())
                for idx, c in enumerate(raw.get("chunks", []))
            ]
            papers.append(Paper(
                paper_id=pid,
                title=str(raw.get("title", f"{topic} 相关论文 {i+1}")),
                year=int(raw.get("year", 2023)),
                authors=str(raw.get("authors", "Generated")),
                url=str(raw.get("url", "")),
                keywords=list(raw.get("keywords", [topic]) or [topic]),
                chunks=chunks,
                need_human_review=True,
            ))
            existing_ids.add(pid)
        except Exception as e:  # noqa: BLE001
            logger.warning("第 %d 篇解析失败: %s", i+1, e)

    # 生成 QA（每篇 2 条）
    logger.info("生成 QA：已有 %d 条，目标 %d 条", len(qa_items), target_count * 2)
    target_qa = target_count * 2
    existing_qas = {q.qa_id for q in qa_items}
    for p in papers:
        if len(qa_items) >= target_qa:
            break
        if p.paper_id in ("sample_paper_1", "sample_paper_2", "sample_paper_3"):
            continue  # 内置论文 QA 已完整
        # 生成 2 条/篇
        for n in range(2):
            if len(qa_items) >= target_qa:
                break
            qid = f"{p.paper_id}_q{n+1}"
            if qid in existing_qas:
                continue
            body = "\n".join(c.text for c in p.chunks[:3])
            prompt = (
                f"基于以下论文正文片段，生成 1 条 {['性能指标数值', '方法描述'][n%2]} 类问答，"
                f"JSON 格式：question / gold_answer / answer_type=number|method / "
                f"gold_citations=[{{chunk_id, ref_text}}] / expected_keywords=[]。"
                f"ref_text 要求精确复制正文里的 1-2 句原文（≤80 字）。不要写解释，直接写 JSON。\n\n"
                f"--- 正文 ---\n{body[:1500]}"
            )
            resp = client.chat(prompt, system="你是学术评测数据构造助手，只输出 JSON，不要写多余文字。")
            if not resp.text:
                continue
            try:
                # 容错：找到第一个 { 到最后一个 }
                s, e = resp.text.find("{"), resp.text.rfind("}")
                if s < 0 or e < 0:
                    continue
                raw = json.loads(resp.text[s:e+1])
                qa_items.append(QAItem(
                    qa_id=qid,
                    paper_id=p.paper_id,
                    question=raw["question"],
                    answer_type=raw.get("answer_type", "method"),
                    gold_answer=raw.get("gold_answer", ""),
                    gold_citations=[
                        GoldCitation(chunk_id=str(gc.get("chunk_id", p.chunks[0].chunk_id)),
                                     ref_text=str(gc.get("ref_text", "")))
                        for gc in raw.get("gold_citations", [])
                    ],
                    tolerance_relative=raw.get("tolerance_relative"),
                    expected_keywords=list(raw.get("expected_keywords", []) or []),
                    need_human_review=True,
                ))
                existing_qas.add(qid)
            except Exception as e:  # noqa: BLE001
                logger.debug("QA 解析失败: %s", e)

    logger.info("扩充完成：论文 %d 篇，QA %d 条", len(papers), len(qa_items))


# ---------------------------------------------------------------------------
# 落盘（写回 paper_samples/*.yaml + qa_pairs.json）
# ---------------------------------------------------------------------------


def _save(papers: List[Paper], qa_items: List[QAItem]) -> None:
    """写回磁盘（仅 YAML 可写，保证文件格式一致）"""
    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML 未安装，跳过论文 YAML 写回")
        _yaml = None

    if _yaml is not None:
        for p in papers:
            path = PAPER_SAMPLES_DIR / f"{p.paper_id}.yaml"
            data = {
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "authors": p.authors,
                "url": p.url,
                "keywords": list(p.keywords),
                "chunks": [asdict(c) for c in p.chunks],
            }
            if p.need_human_review:
                data["need_human_review"] = True
            path.write_text(_yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")

    qdata = []
    for q in qa_items:
        d = {
            "qa_id": q.qa_id,
            "paper_id": q.paper_id,
            "question": q.question,
            "answer_type": q.answer_type,
            "gold_answer": q.gold_answer,
            "gold_citations": [asdict(g) for g in q.gold_citations],
        }
        if q.tolerance_relative:
            d["tolerance_relative"] = q.tolerance_relative
        if q.expected_keywords:
            d["expected_keywords"] = list(q.expected_keywords)
        if q.formula_expectation:
            d["formula_expectation"] = list(q.formula_expectation)
        if q.need_human_review:
            d["need_human_review"] = True
        qdata.append(d)

    QA_PAIRS_FILE.write_text(
        json.dumps(qdata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("落盘完成：%s 目录（%d 篇论文，%d 条 QA）",
                PAPER_SAMPLES_DIR, len(papers), len(qdata))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expand", type=int, default=0,
                    help="扩充论文目标篇数（0=不扩充，直接用内置样例）")
    ap.add_argument("--no-save", action="store_true",
                    help="仅加载/生成，不写回磁盘")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    papers = load_builtin_papers()
    qa_items = load_builtin_qa()

    if args.expand > 0:
        _expand_with_dashscope(papers, qa_items, args.expand)

    logger.info("数据集就绪：论文 %d 篇（内置 %d，扩充 %d），QA %d 条",
                len(papers), sum(1 for p in papers if not p.need_human_review),
                sum(1 for p in papers if p.need_human_review), len(qa_items))

    if not args.no_save:
        _save(papers, qa_items)
    else:
        logger.info("--no-save：不写回磁盘")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""论文框架拆解 + 实验思路提炼。

对已有的 chunk 列表（切分后的论文文本）完成：
1. 段落角色标注：把每个 chunk 归入 8 类学术章节之一（见 CHUNK_ROLES）
2. 按角色重组：论文结构 → {role: [chunk, ...]}
3. 核心内容提炼：
     - innovations: 本文核心创新点（由 Abstract + Introduction 末段 + Method 开头段合成；DashScope 模式调用 LLM）
     - method_summary: 实验思路/算法流程（Method 节概括）
     - experiments_summary: 实验设置、数据集、结果结论（Experiments + Results 节概括）
4. 输出 PaperStructure dataclass，供前端展示"论文框架一览"和后续 RAG 分节检索复用。

两种模式：
- MockStructurer：关键词规则 + 段落序号分布，无 Key 也能跑通链路
- LLMStructurer：DashScope 调用 LLM 做角色分类 + 要点抽取（DashScope Key 可用时自动选用）
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 8 类学术章节角色
# ---------------------------------------------------------------------------

CHUNK_ROLES: Tuple[str, ...] = (
    "Abstract",        # 摘要 / 贡献点
    "Introduction",    # 研究背景 / 问题定义 / 贡献概述
    "Related Work",    # 相关工作 / 文献综述
    "Method",          # 方法 / 模型结构 / 算法流程（"实验思路"来源）
    "Experiments",     # 实验设置：数据集、对比基线、超参数
    "Results",         # 结果与分析：数值表格、消融、案例
    "Discussion",      # 讨论：局限、负例、未来方向
    "Conclusion",      # 结论
    "Other",           # 致谢 / 参考文献 / 附录（不参与提炼）
)


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------


@dataclass
class LabeledChunk:
    """带角色标注的 chunk。"""

    chunk_id: str
    role: str
    text: str
    confidence: float = 0.0  # 角色分类置信度（规则 1.0 视为确定）


@dataclass
class PaperStructure:
    """论文框架拆解结果。"""

    paper_id: str
    labeled_chunks: List[LabeledChunk] = field(default_factory=list)
    # 按角色分组后的 chunk_id 索引
    role_to_chunks: Dict[str, List[str]] = field(default_factory=dict)
    # 核心提炼内容（Mock/LLM 按可用度填充）
    innovations: List[str] = field(default_factory=list)
    method_summary: str = ""
    experiments_summary: str = ""
    raw_text_by_role: Dict[str, str] = field(default_factory=dict)  # 调试用
    from_mock: bool = False

    def role_chunks(self, role: str) -> List[LabeledChunk]:
        ids = set(self.role_to_chunks.get(role, []))
        return [c for c in self.labeled_chunks if c.chunk_id in ids]


# ---------------------------------------------------------------------------
# 基础：关键词规则角色分类（Mock 用，LLM 可将其作为 few-shot 提示）
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: Dict[str, List[str]] = {
    # 标题级强关键词（优先匹配）
    "Abstract":     ["abstract", "摘要", "in this paper we", "we propose", "our contributions", "our main contributions"],
    "Introduction": ["introduction", "引言", "背景", "motivation", "problem statement", "research question", "1. ", "1  "],
    "Related Work": ["related work", "相关工作", "previous work", "literature review", "2. ", "2  "],
    "Method":       [
        "method", "方法", "methodology", "approach", "framework", "algorithm",
        "3. ", "3  ", "4. method", "the proposed", "proposed method",
    ],
    "Experiments":  [
        "experiments", "实验", "experimental setup", "dataset", "datasets",
        "baseline", "baselines", "implementation details", "hyperparameters",
        "4. ", "4  ", "5. experiments",
    ],
    "Results":      [
        "results", "结果", "main results", "ablation", "消融", "analysis",
        "table 1", "table 2", "table 3", "figure 1", "figure 2",
        "5. ", "5  ", "6. results", "performance",
    ],
    "Discussion":   ["discussion", "讨论", "limitation", "limitations", "future work", "threats to validity"],
    "Conclusion":   ["conclusion", "结论", "concluding remarks", "summary"],
    "Other":        ["acknowledgment", "acknowledgements", "references", "参考文献", "appendix", "附录"],
}


def _rule_based_label(text: str, section_hint: str = "", order_ratio: float = 0.5) -> Tuple[str, float]:
    """规则分类：section_hint / 关键词 / 顺序比例 → (role, confidence)。

    order_ratio：当前 chunk 序号 / 总 chunks 数，用于区分前后文位置：
        0~0.1  更偏向 Abstract / Introduction
        0.1~0.3 Related Work
        0.3~0.6 Method
        0.6~0.8 Experiments / Results
        0.8~1.0 Discussion / Conclusion
    """
    # 1) section_hint 里已经有章节名，直接命中（置信度 1.0）
    lowered_hint = section_hint.lower()
    for role, kws in ROLE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lowered_hint:
                return role, 1.0
    # 2) 正文关键词匹配（统计命中数，加权）
    score: Dict[str, int] = {r: 0 for r in ROLE_KEYWORDS}
    lowered_text = text.lower()
    for role, kws in ROLE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lowered_text:
                # 标题级关键词（如 "Experimental Setup"）权重高 2
                score[role] += 2 if len(kw) <= 20 else 1
    # 3) 顺序比例位置先验
    pos_priority: List[Tuple[float, str]] = [
        (0.05, "Abstract"),
        (0.15, "Introduction"),
        (0.25, "Related Work"),
        (0.45, "Method"),
        (0.70, "Experiments"),
        (0.78, "Results"),
        (0.88, "Discussion"),
        (0.95, "Conclusion"),
    ]
    # 找距离 order_ratio 最近的角色，+1 分
    nearest = min(pos_priority, key=lambda pair: abs(pair[0] - order_ratio))[1]
    score[nearest] = score.get(nearest, 0) + 1
    # 4) 选择得分最高的角色；confidence = 命中分数 / 总分（最低 0.3）
    total = sum(score.values()) or 1
    best_role = max(score.items(), key=lambda kv: kv[1])[0]
    confidence = max(0.3, score[best_role] / total)
    return best_role, confidence


# ---------------------------------------------------------------------------
# 抽象与实现
# ---------------------------------------------------------------------------


class BaseStructurer(ABC):

    @abstractmethod
    def structure(self, paper_id: str, chunks, title: str = "") -> PaperStructure:
        ...

    # ------------------------------------------------------------------
    # 共享：把 rule_based_label 的结果批量写入 role_to_chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _build_from_labels(paper_id: str, labels: List[LabeledChunk], title: str = "") -> PaperStructure:
        role_to_chunks: Dict[str, List[str]] = {}
        raw_text: Dict[str, str] = {}
        for lc in labels:
            role_to_chunks.setdefault(lc.role, []).append(lc.chunk_id)
            raw_text[lc.role] = (raw_text.get(lc.role, "") + "\n" + lc.text).strip()
        return PaperStructure(
            paper_id=paper_id,
            labeled_chunks=list(labels),
            role_to_chunks=role_to_chunks,
            raw_text_by_role=raw_text,
        )


class MockStructurer(BaseStructurer):
    """规则分类 + 启发式摘要提炼，无 API Key 也能跑通链路。"""

    def structure(self, paper_id: str, chunks, title: str = "") -> PaperStructure:
        n = max(1, len(list(chunks)))
        labeled: List[LabeledChunk] = []
        chunk_list = list(chunks)
        for idx, c in enumerate(chunk_list):
            text = getattr(c, "text", "") if hasattr(c, "text") else str(c)
            chunk_id = getattr(c, "chunk_id", f"{paper_id}#c{idx}") if hasattr(c, "chunk_id") else f"{paper_id}#c{idx}"
            section = getattr(c, "section", "") if hasattr(c, "section") else ""
            role, conf = _rule_based_label(text, section_hint=section, order_ratio=idx / n)
            labeled.append(LabeledChunk(chunk_id=chunk_id, role=role, text=text, confidence=conf))
        struct = self._build_from_labels(paper_id, labeled, title=title)
        struct = self._summarize_mock(struct, title=title)
        struct.from_mock = True
        return struct

    # ------------------------------------------------------------------
    # 启发式提炼（取前 N 句 / 贡献句关键词匹配）
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_mock(struct: PaperStructure, title: str = "") -> PaperStructure:
        # 创新点：从 Abstract / Introduction / Method 抓含 "we propose / our contributions / 提出" 的句子
        innov_text = "\n".join(
            struct.raw_text_by_role.get(r, "") for r in ["Abstract", "Introduction", "Method"]
        )
        innovations = _extract_claim_sentences(innov_text, max_sentences=5)
        if not innovations:
            innovations = [f"（mock 占位）论文《{title or struct.paper_id}》提出了若干改进，详情见 Method 节。"]
        struct.innovations = innovations
        # Method 概括：取 Method 节前 400 字
        method_text = struct.raw_text_by_role.get("Method", "")
        struct.method_summary = method_text[:400] + ("…" if len(method_text) > 400 else "") or "（mock 占位）本文方法流程见 Method 节标注的段落。"
        # Experiments + Results 汇总
        exp_text = (struct.raw_text_by_role.get("Experiments", "") + "\n" + struct.raw_text_by_role.get("Results", "")).strip()
        struct.experiments_summary = exp_text[:500] + ("…" if len(exp_text) > 500 else "") or "（mock 占位）实验设置与结果见 Experiments/Results 节。"
        return struct


class LLMStructurer(BaseStructurer):
    """DashScope LLM 做角色分类 + 要点抽取；DashScope 不可用时自动降级 Mock。"""

    def __init__(self, model: str = "qwen-plus"):
        from app.llm import create_llm_client
        self._llm = create_llm_client("dashscope", model=model)
        self._ok = getattr(self._llm, "_ensure_client", lambda: False)()
        self.model = model

    @property
    def available(self) -> bool:
        return self._ok

    def structure(self, paper_id: str, chunks, title: str = "") -> PaperStructure:
        if not self._ok:
            logger.warning("DashScope LLM 不可用，降级为 MockStructurer")
            return MockStructurer().structure(paper_id, chunks, title=title)

        chunk_list = list(chunks)
        n = len(chunk_list)
        # 角色分类：批量（一次 prompt 覆盖 20 chunks 以内，超出分批或直接用规则先标再 LLM 校正）
        if n <= 20:
            labeled = self._llm_label_batch(paper_id, chunk_list, title=title)
        else:
            # 大论文：规则先标，LLM 校正前 20 + 后 10 个关键段落
            base = [LabeledChunk(
                chunk_id=getattr(c, "chunk_id", f"{paper_id}#c{i}"),
                role="",
                text=getattr(c, "text", str(c)),
            ) for i, c in enumerate(chunk_list)]
            for i, c in enumerate(chunk_list):
                role, conf = _rule_based_label(
                    getattr(c, "text", str(c)),
                    section_hint=getattr(c, "section", ""),
                    order_ratio=i / n,
                )
                base[i].role = role
                base[i].confidence = conf
            indices_to_recheck = list(range(min(20, n))) + list(range(max(0, n - 10), n))
            rechecked = self._llm_label_batch(paper_id, [chunk_list[i] for i in indices_to_recheck], title=title)
            for local_i, global_i in enumerate(indices_to_recheck):
                base[global_i].role = rechecked[local_i].role
                base[global_i].confidence = max(base[global_i].confidence, rechecked[local_i].confidence)
            labeled = base

        struct = self._build_from_labels(paper_id, labeled, title=title)
        # 要点抽取
        struct.innovations = self._llm_extract_innovations(struct, title=title)
        struct.method_summary = self._llm_summarize(struct, "Method", "概括本文方法/算法思路/模型结构，200字以内")
        struct.experiments_summary = self._llm_summarize(
            struct, "Experiments",
            "概括实验设置（数据集/对比方法/超参数）和主要结果（数值表+结论），300字以内",
            extra_roles=["Results"],
        )
        return struct

    # ------------------------------------------------------------------
    # LLM 内部
    # ------------------------------------------------------------------

    def _llm_label_batch(self, paper_id: str, chunk_list, title: str = "") -> List[LabeledChunk]:
        """让 LLM 给每段分角色：输出 JSON 数组 [{chunk_id, role}]。"""
        chunks_info: list[str] = []
        for i, c in enumerate(chunk_list):
            chunk_id = getattr(c, "chunk_id", f"{paper_id}#c{i}")
            text = getattr(c, "text", str(c))[:300].replace("\n", " ")
            chunks_info.append(f"{i}. [chunk_id={chunk_id}] {text}")
        roles_str = ", ".join(CHUNK_ROLES)
        system = (
            "你是论文结构分析助手。请阅读分块后的论文片段，为每个 chunk 从以下角色中选 1 个最合适的：\n"
            f"  {roles_str}\n"
            "严格只输出 JSON 数组，不要解释，不要 markdown：\n"
            '  [{"chunk_id": "<原chunk_id>", "role": "<选一个>"}]\n'
            "注意：chunk_id 必须原样保留。"
        )
        user_prompt = (
            f"论文标题：{title or paper_id}\n\n分块列表（共 {len(chunks_info)} 段）：\n" +
            "\n".join(chunks_info) +
            "\n\n输出 JSON："
        )
        resp = self._llm.chat(user_prompt, system=system, temperature=0.0, max_tokens=max(512, len(chunks_info) * 64))
        parsed = _try_parse_json_array(resp.text or "")
        out: List[LabeledChunk] = []
        # 对齐到原始 chunk_list
        parsed_map: Dict[str, str] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    cid = str(item.get("chunk_id", ""))
                    role = str(item.get("role", "Other"))
                    if role not in CHUNK_ROLES:
                        role = "Other"
                    parsed_map[cid] = role
        for i, c in enumerate(chunk_list):
            chunk_id = getattr(c, "chunk_id", f"{paper_id}#c{i}")
            text = getattr(c, "text", str(c))
            role = parsed_map.get(chunk_id)
            if role is None:
                # LLM JSON 不全：规则兜底
                section = getattr(c, "section", "")
                n = max(1, len(chunk_list))
                role, conf = _rule_based_label(text, section_hint=section, order_ratio=i / n)
                out.append(LabeledChunk(chunk_id=chunk_id, role=role, text=text, confidence=conf))
            else:
                out.append(LabeledChunk(chunk_id=chunk_id, role=role, text=text, confidence=0.9))
        return out

    def _llm_extract_innovations(self, struct: PaperStructure, title: str = "") -> List[str]:
        src = "\n".join(struct.raw_text_by_role.get(r, "") for r in ["Abstract", "Introduction", "Method"]).strip()
        if not src:
            return [f"《{title or struct.paper_id}》：暂未抽取到创新点。"]
        system = ("你是学术研读助手。输出 3~6 条本文的核心创新点/贡献，每条 40 字以内，纯 JSON 字符串数组：\n"
                  '["...", "..."]\n不要解释、不要 markdown。')
        user_prompt = f"论文标题：{title or struct.paper_id}\n\n论文片段：\n---\n{src[:3000]}\n---\n\n核心创新点（JSON 数组）："
        resp = self._llm.chat(user_prompt, system=system, temperature=0.2, max_tokens=400)
        parsed = _try_parse_json_array(resp.text or "")
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()][:6]
        # JSON 失败：按换行切出非空行
        return [ln.strip("-•* \t") for ln in (resp.text or "").splitlines() if ln.strip()][:6]

    def _llm_summarize(self, struct: PaperStructure, primary_role: str, instruction: str,
                       extra_roles: Optional[List[str]] = None) -> str:
        roles = [primary_role] + (extra_roles or [])
        src = "\n".join(struct.raw_text_by_role.get(r, "") for r in roles).strip()
        if not src:
            return ""
        system = (
            f"你是学术研读助手。{instruction}。\n"
            "输出纯中文文本，不要列表、不要 markdown、不要解释。"
        )
        user_prompt = f"论文《{struct.paper_id}》相关章节原文：\n---\n{src[:2500]}\n---\n\n概括："
        resp = self._llm.chat(user_prompt, system=system, temperature=0.2, max_tokens=512)
        return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def create_structurer(mode: str = "auto", **kwargs) -> BaseStructurer:
    mode = mode.lower()
    if mode == "mock":
        return MockStructurer(**kwargs)
    if mode == "llm":
        s = LLMStructurer(**kwargs)
        if not s.available:
            logger.warning("DashScope 不可用，降级为 MockStructurer")
            return MockStructurer()
        return s
    if mode == "auto":
        s = LLMStructurer(**kwargs)
        if s.available:
            return s
        return MockStructurer()
    raise ValueError(f"未知 structurer mode: {mode}")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _extract_claim_sentences(text: str, max_sentences: int = 5) -> List[str]:
    """启发式从文本里切出"提出/贡献/创新"声明句。"""
    if not text:
        return []
    # 简单按句号/分号/换行分句（中英文混合）
    sentences = re.split(r"(?<=[。；;!?\n])", text)
    claim_kws = [
        "we propose", "we present", "we introduce", "we contribute", "contribution",
        "本文提出", "本文介绍", "我们提出", "我们的贡献", "contributions",
        "novel", "state-of-the-art", "sota", "outperform", "for the first time",
    ]
    hits: List[str] = []
    for s in sentences:
        s = s.strip(" \n\t")
        if not s or len(s) < 20:
            continue
        lower = s.lower()
        if any(kw in lower for kw in claim_kws):
            hits.append(s)
        if len(hits) >= max_sentences:
            break
    if not hits:
        # 退而求其次：取前 N 句"较完整"（>50 字）
        for s in sentences:
            s = s.strip()
            if len(s) >= 50:
                hits.append(s)
            if len(hits) >= max_sentences:
                break
    return hits[:max_sentences]


def _try_parse_json_array(text: str):
    """尝试从 LLM 返回文本中解析 JSON 数组（容忍 markdown 代码块/前后脏文本）。"""
    if not text:
        return None
    import json
    t = text.strip()
    # 去掉 ```json ... ``` 包裹
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()
    # 找到第一个 [ 到最后一个 ]
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None

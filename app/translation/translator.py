"""多语种文献翻译模块。

模块组成：
- TermGlossary：专业术语词库（英文术语 → 标准中文译法 + 学科域 + 别名），翻译前后做替换校准
- TranslationUnit：单 chunk 翻译结果（含原文、译文、对齐到的术语）
- TranslatedPaper：单篇翻译完成的 PaperSchema（保持与原 chunk 顺序一致，便于后续检索）
- DashScopeTranslator：调 Qwen-Plus 翻译（系统提示约束术语使用标准译法），无 Key/缺包时降级为 MockTranslator
- MockTranslator：模板化"翻译"，保证无 Key 链路通
- create_translator：工厂函数
- translate_folder：批量翻译目录下的 YAML/PDF（PDF 需要 parser 先解析）
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 术语词库
# ---------------------------------------------------------------------------


@dataclass
class TermEntry:
    """单条术语条目。"""

    term_en: str
    term_zh: str
    domain: str = ""          # 学科域（CS/Physics/Bio/...），用于冲突仲裁
    aliases: List[str] = field(default_factory=list)  # 别名/缩写

    def all_forms(self) -> List[str]:
        return [self.term_en, *self.aliases]


class TermGlossary:
    """专业术语词库：英文术语 → 标准中文译法。

    翻译时的作用：
      1. 前置（翻译前）：把术语替换为 ``__TERM_{idx}__`` 占位符，避免 LLM 误译。
      2. 后置（翻译后）：把占位符还原为标准中文译法。
      3. 对 LLM 返回文本做"译法后处理"：若 LLM 用了本词库中术语的其他译法，强制替换为标准译法。
    """

    def __init__(self, entries: Optional[Iterable[TermEntry]] = None):
        self._entries: List[TermEntry] = []
        # form_lower -> entry
        self._form_to_entry: Dict[str, TermEntry] = {}
        if entries:
            for e in entries:
                self.add(e)
        # 如果空，自动载入内置理工科基础词库（保证默认开箱可用）
        if not self._entries:
            for e in self._builtin_entries():
                self.add(e)

    # ------------------------------------------------------------------
    # 增删查
    # ------------------------------------------------------------------

    def add(self, entry: TermEntry) -> None:
        if entry in self._entries:
            return
        self._entries.append(entry)
        for form in entry.all_forms():
            if not form:
                continue
            self._form_to_entry[form.lower()] = entry

    def entries(self) -> List[TermEntry]:
        return list(self._entries)

    def find_en(self, text: str) -> Optional[TermEntry]:
        return self._form_to_entry.get(text.strip().lower())

    # ------------------------------------------------------------------
    # 核心：翻译前后校准
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Tuple[str, Dict[str, str]]:
        """把文本中命中的术语替换为占位符，避免 LLM 误译。

        Returns:
            (processed_text, placeholder_map)  placeholder_map={placeholder: term_zh}
        """
        placeholder_map: Dict[str, str] = {}
        processed = text
        # 长术语优先匹配（避免 "self-attention" 被 "attention" 先替换）
        sorted_entries = sorted(
            self._entries,
            key=lambda e: max(len(f) for f in e.all_forms()),
            reverse=True,
        )
        for idx, entry in enumerate(sorted_entries):
            placeholder = f"__TERM_{idx}__"
            mapped_this_entry = False
            for form in entry.all_forms():
                if not form:
                    continue
                new_text = _replace_case_insensitive(processed, form, placeholder)
                if new_text != processed:
                    processed = new_text
                    mapped_this_entry = True
            if mapped_this_entry:
                placeholder_map[placeholder] = entry.term_zh
        return processed, placeholder_map

    def postprocess(self, translated_text: str, placeholder_map: Dict[str, str]) -> Tuple[str, List[str]]:
        """把占位符还原为标准中文译法，并做 LLM 非标准译法覆盖。

        Returns:
            (final_text, aligned_terms)  aligned_terms：本句翻译中校准过的术语列表
        """
        result = translated_text
        aligned: List[str] = []
        for placeholder, zh in placeholder_map.items():
            if placeholder in result:
                result = result.replace(placeholder, zh)
                if zh not in aligned:
                    aligned.append(zh)
        # 译法后处理覆盖：若 LLM 用了其他译法，强制替换为标准译法
        for entry in self._entries:
            non_standard = _guess_nonstandard_zh(entry)
            for bad in non_standard:
                if bad and bad != entry.term_zh and bad in result:
                    result = result.replace(bad, entry.term_zh)
                    if entry.term_zh not in aligned:
                        aligned.append(entry.term_zh)
        # 去"术语+重复后缀"：占位符还原后可能出现
        #   "自注意力机制机制是..." / "Transformer 架构架构核心..." 类叠词
        # 这里去掉紧跟在 zh 之后重复出现的 zh 的最后一个中文词
        result = _deduplicate_suffix(result, placeholder_map.values())
        return result, aligned

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = [
            {
                "term_en": e.term_en,
                "term_zh": e.term_zh,
                "domain": e.domain,
                "aliases": e.aliases,
            }
            for e in self._entries
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("术语词库已保存: %s (%d 条)", path, len(data))

    def load(self, path: str) -> None:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        for item in data:
            self.add(TermEntry(
                term_en=item["term_en"],
                term_zh=item["term_zh"],
                domain=item.get("domain", ""),
                aliases=list(item.get("aliases", [])),
            ))
        logger.info("术语词库已加载: %s (%d 条)", path, len(data))

    # ------------------------------------------------------------------
    # 内置理工科基础词库
    # ------------------------------------------------------------------

    @staticmethod
    def _builtin_entries() -> List[TermEntry]:
        return [
            # 通用 AI / 机器学习
            TermEntry("Transformer", "Transformer 架构", "CS", ["transformer model", "Transformer model"]),
            TermEntry("self-attention", "自注意力机制", "CS", ["self attention", "Self-Attention"]),
            TermEntry("multi-head attention", "多头注意力机制", "CS", ["Multi-Head Attention", "multi head attention"]),
            TermEntry("scaled dot-product attention", "缩放点积注意力", "CS", []),
            TermEntry("positional encoding", "位置编码", "CS", ["position encoding", "positional embedding"]),
            TermEntry("residual connection", "残差连接", "CS", ["skip connection", "残差网络"]),
            TermEntry("layer normalization", "层归一化", "CS", ["LayerNorm", "layer norm"]),
            TermEntry("feed-forward network", "前馈神经网络", "CS", ["FFN", "position-wise feed-forward"]),
            TermEntry("BERT", "BERT 预训练模型", "CS", []),
            TermEntry("GPT", "GPT 生成式预训练模型", "CS", []),
            TermEntry("T5", "T5 文本到文本预训练模型", "CS", []),
            TermEntry("fine-tuning", "微调", "CS", ["fine tuning", "Fine-tuning"]),
            TermEntry("pre-training", "预训练", "CS", ["pre training", "Pre-training"]),
            TermEntry("prompt tuning", "提示调优", "CS", ["prompt-tuning", "soft prompt"]),
            TermEntry("instruction tuning", "指令调优", "CS", ["instruction-tuning"]),
            TermEntry("tokenization", "分词", "CS", ["tokenization", "tokenize"]),
            TermEntry("sentencepiece", "SentencePiece 子词分词器", "CS", ["BPE", "byte-pair encoding"]),
            TermEntry("embedding", "词嵌入", "CS", ["embeddings"]),
            TermEntry("recall", "召回率", "CS", []),
            TermEntry("precision", "准确率", "CS", []),
            TermEntry("F1 score", "F1 分数", "CS", ["F1"]),
            TermEntry("BLEU", "BLEU 分数", "CS", []),
            TermEntry("ROUGE", "ROUGE 评测指标", "CS", []),
            TermEntry("METEOR", "METEOR 评测指标", "CS", []),
            TermEntry("Perplexity", "困惑度", "CS", ["PPL"]),
            TermEntry("convolutional neural network", "卷积神经网络", "CS", ["CNN"]),
            TermEntry("recurrent neural network", "循环神经网络", "CS", ["RNN"]),
            TermEntry("Long Short-Term Memory", "长短期记忆网络", "CS", ["LSTM"]),
            TermEntry("Gated Recurrent Unit", "门控循环单元", "CS", ["GRU"]),
            TermEntry("graph neural network", "图神经网络", "CS", ["GNN"]),
            TermEntry("contrastive learning", "对比学习", "CS", []),
            TermEntry("contrastive loss", "对比损失", "CS", ["InfoNCE"]),
            TermEntry("knowledge distillation", "知识蒸馏", "CS", []),
            TermEntry("reinforcement learning", "强化学习", "CS", ["RL"]),
            TermEntry("Reinforcement Learning from Human Feedback", "基于人类反馈的强化学习", "CS", ["RLHF"]),
            TermEntry("language model", "语言模型", "CS", ["LM"]),
            TermEntry("large language model", "大语言模型", "CS", ["LLM"]),
            TermEntry("Mixture of Experts", "专家混合模型", "CS", ["MoE"]),
            TermEntry("Retrieval-Augmented Generation", "检索增强生成", "CS", ["RAG"]),
            TermEntry("Chain-of-Thought", "思维链", "CS", ["CoT"]),
            # 计算机视觉
            TermEntry("Vision Transformer", "视觉 Transformer", "CS", ["ViT", "Vision Transformer"]),
            TermEntry("Swin Transformer", "Swin Transformer 模型", "CS", ["Swin"]),
            TermEntry("Convolutional Neural Vision", "卷积视觉", "CS", ["CNN backbone"]),
            TermEntry("object detection", "目标检测", "CS", []),
            TermEntry("semantic segmentation", "语义分割", "CS", []),
            TermEntry("instance segmentation", "实例分割", "CS", []),
            TermEntry("Intersection over Union", "交并比", "CS", ["IoU"]),
            TermEntry("mean Average Precision", "平均精度均值", "CS", ["mAP"]),
            # 通用学术
            TermEntry("ablation study", "消融实验", "General", ["ablation", "ablation experiments"]),
            TermEntry("baseline", "基线方法", "General", []),
            TermEntry("state-of-the-art", "当前最优方法", "General", ["SOTA", "state of the art"]),
            TermEntry("dataset", "数据集", "General", ["datasets"]),
            TermEntry("benchmark", "基准测试集", "General", []),
            TermEntry("hyperparameter", "超参数", "General", ["hyperparameters"]),
            TermEntry("learning rate", "学习率", "General", ["LR"]),
            TermEntry("batch size", "批大小", "General", []),
            TermEntry("epoch", "训练轮次", "General", ["epochs"]),
            TermEntry("gradient descent", "梯度下降", "General", ["SGD", "stochastic gradient descent"]),
            TermEntry("backpropagation", "反向传播", "General", ["backprop"]),
            TermEntry("dropout", "丢弃正则化", "General", []),
            TermEntry("weight decay", "权重衰减", "General", []),
            TermEntry("early stopping", "早停策略", "General", []),
            TermEntry("cross validation", "交叉验证", "General", []),
            TermEntry("training curve", "训练曲线", "General", []),
            TermEntry("convergence", "收敛性", "General", ["converge"]),
        ]


# ---------------------------------------------------------------------------
# 翻译数据结构
# ---------------------------------------------------------------------------


@dataclass
class TranslationUnit:
    """单 chunk 翻译结果。"""

    chunk_id: str
    source_text: str
    target_text: str
    target_lang: str = "zh-CN"
    aligned_terms: List[str] = field(default_factory=list)  # 本句被校准过的术语
    latency_ms: int = 0
    from_mock: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.target_text)


@dataclass
class TranslatedPaper:
    """翻译完成的论文。

    保持与原始 chunks 顺序一致，chunk_id 不变，
    这样切分/向量化/检索流程可以直接复用。
    """

    paper_id: str
    title_zh: str
    abstract_zh: str = ""
    units: List[TranslationUnit] = field(default_factory=list)
    term_glossary_used: List[str] = field(default_factory=list)  # 用到的术语
    total_latency_ms: int = 0
    mock_mode: bool = False

    def to_chunk_list(self, original_doc_id: Optional[str] = None):
        """把翻译 units 转换为 Chunk 列表（方便直接入 rag 索引）。

        导入放内部避免循环依赖。
        """
        from app.rag.chunker import Chunk
        chunks: List[Chunk] = []
        for idx, u in enumerate(self.units):
            doc_id = original_doc_id or self.paper_id
            chunks.append(Chunk(
                chunk_id=u.chunk_id,
                text=u.target_text,
                token_count=max(1, len(u.target_text) // 3),
                doc_id=doc_id,
                section=f"翻译#{idx:03d}",
                section_idx=0,
                chunk_idx=idx,
                overlap_prev=0,
                metadata={"lang": self.units[0].target_lang if self.units else "zh-CN",
                          "source": "translated",
                          "aligned_terms": u.aligned_terms},
            ))
        return chunks


# ---------------------------------------------------------------------------
# 翻译器抽象与实现
# ---------------------------------------------------------------------------


class BaseTranslator(ABC):
    """翻译器抽象基类。"""

    def __init__(self, glossary: Optional[TermGlossary] = None):
        self.glossary = glossary or TermGlossary()

    @abstractmethod
    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "zh-CN") -> TranslationUnit:
        """翻译单段文本。"""
        ...

    # ------------------------------------------------------------------
    # 高阶：翻译单篇 / 批量
    # ------------------------------------------------------------------

    def translate_paper(self, paper_id: str, chunks, title_en: str = "",
                        abstract_en: str = "", source_lang: str = "en",
                        target_lang: str = "zh-CN") -> TranslatedPaper:
        """按 chunk 翻译一整篇论文。

        chunks：任意具有 chunk_id / text 属性的对象列表
                （app.rag.chunker.Chunk 或 tests.qa_dataset.Paper.chunks）
        """
        import time
        t0 = time.time()
        units: List[TranslationUnit] = []
        used_terms: List[str] = []
        for c in chunks:
            text = getattr(c, "text", "") if hasattr(c, "text") else str(c)
            chunk_id = getattr(c, "chunk_id", f"{paper_id}#chunk_{len(units)}") if hasattr(c, "chunk_id") else f"{paper_id}#chunk_{len(units)}"
            u = self.translate_text(text, source_lang=source_lang, target_lang=target_lang)
            u.chunk_id = chunk_id
            units.append(u)
            for t in u.aligned_terms:
                if t not in used_terms:
                    used_terms.append(t)
        title_zh = self.translate_text(title_en or paper_id, source_lang=source_lang, target_lang=target_lang).target_text if title_en else paper_id
        abstract_zh = self.translate_text(abstract_en, source_lang=source_lang, target_lang=target_lang).target_text if abstract_en else ""
        total_ms = int((time.time() - t0) * 1000)
        return TranslatedPaper(
            paper_id=paper_id,
            title_zh=title_zh,
            abstract_zh=abstract_zh,
            units=units,
            term_glossary_used=used_terms,
            total_latency_ms=total_ms,
            mock_mode=self.__class__.__name__ == "MockTranslator",
        )

    def translate_folder(self, folder: str, source_lang: str = "en",
                         target_lang: str = "zh-CN",
                         suffix: str = ".yaml") -> List[TranslatedPaper]:
        """批量翻译目录下的 YAML 论文（qa_dataset 格式）。

        返回翻译后的 TranslatedPaper 列表（不写回源文件，避免覆盖）。
        """
        # 延迟导入避免循环依赖
        from dataclasses import dataclass as _dc
        from tests.qa_dataset.build_dataset import _load_yaml
        p = Path(folder)
        if not p.exists():
            logger.warning("translate_folder 目录不存在: %s", folder)
            return []
        yaml_files = sorted(p.glob(f"*{suffix}")) if suffix else sorted([f for f in p.iterdir() if f.suffix == ".yaml"])
        results: List[TranslatedPaper] = []

        @_dc
        class _ChunkLike:
            chunk_id: str
            text: str
            section: str = ""

        for f in yaml_files:
            raw = _load_yaml(f)
            paper_id = raw.get("paper_id", f.stem)
            title = raw.get("title", f.stem)
            chunk_dicts = raw.get("chunks", []) or []
            chunks = [
                _ChunkLike(
                    chunk_id=c.get("chunk_id") or f"{paper_id}#chunk_{i}",
                    text=c.get("text", ""),
                    section=c.get("section", ""),
                )
                for i, c in enumerate(chunk_dicts)
            ]
            abstract_en = (chunks[0].text if chunks else "")[:500]
            results.append(self.translate_paper(
                paper_id, chunks, title_en=title,
                abstract_en=abstract_en, source_lang=source_lang, target_lang=target_lang,
            ))
        return results


class DashScopeTranslator(BaseTranslator):
    """DashScope Qwen-Plus 翻译器：术语预处理→LLM 翻译→译法后处理。"""

    def __init__(self, glossary: Optional[TermGlossary] = None, model: str = "qwen-plus"):
        super().__init__(glossary)
        self.model = model
        from app.llm import create_llm_client
        # auto：若 Key 不存在自动降级 Mock，但这里通过 _ensure_client 显式判断
        self._llm = create_llm_client("dashscope", model=model)
        self._ok = getattr(self._llm, "_ensure_client", lambda: False)()

    @property
    def available(self) -> bool:
        return self._ok

    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "zh-CN") -> TranslationUnit:
        import time
        t0 = time.time()
        chunk_id = ""
        if not text.strip():
            return TranslationUnit(chunk_id=chunk_id, source_text=text, target_text="",
                                   target_lang=target_lang, aligned_terms=[], latency_ms=0)

        # 1) 预处理：术语→占位符
        processed, placeholder_map = self.glossary.preprocess(text)

        # 2) LLM 翻译：系统提示里把占位符 <-> 标准译法的映射也带上，减少 LLM 误译
        system = (
            f"你是学术论文专业翻译助手，擅长把 {_lang_name(source_lang)} 翻译为 {_lang_name(target_lang)}。\n"
            f"翻译规则：\n"
            f"1) 输出纯译文，不解释、不附加原句、不加引号。\n"
            f"2) 文中形如 __TERM_X__ 的占位符表示「专业术语」，必须原样保留在译文中，不要翻译或改名。\n"
            f"3) 理工科术语优先采用中文计算机界/物理学界通用译法。\n"
            f"4) 公式/数字/缩写（如 BLEU、28.4、WMT 2014）原样保留。\n"
        )
        if placeholder_map:
            mapping_lines = "\n".join(f"  - {k} 对应标准中文译法：{v}" for k, v in placeholder_map.items())
            system += f"\n当前占位符表：\n{mapping_lines}\n"

        user_prompt = (
            f"请翻译以下论文片段为 {_lang_name(target_lang)}，保留所有 __TERM_X__ 占位符：\n\n"
            f"---\n{processed}\n---\n\n译文："
        )
        resp = self._llm.chat(user_prompt, system=system, temperature=0.1, max_tokens=max(128, len(processed) * 2))
        raw = resp.text or ""

        # 3) 后处理：占位符还原 + 非标准译法覆盖
        final, aligned = self.glossary.postprocess(raw, placeholder_map)
        latency = int((time.time() - t0) * 1000)
        if not final:
            final = raw or text
        return TranslationUnit(
            chunk_id=chunk_id, source_text=text, target_text=final,
            target_lang=target_lang, aligned_terms=aligned, latency_ms=latency,
        )


class MockTranslator(BaseTranslator):
    """无 Key 降级翻译：把英文关键词替换为中文模板，保证链路可联调。"""

    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "zh-CN") -> TranslationUnit:
        import time
        t0 = time.time()
        chunk_id = ""
        if not text.strip():
            return TranslationUnit(chunk_id=chunk_id, source_text=text, target_text="",
                                   target_lang=target_lang, aligned_terms=[], latency_ms=0)
        processed, placeholder_map = self.glossary.preprocess(text)
        # Mock：用内置小映射翻译常见词 + 占位符还原
        zh = _mock_en_to_zh(processed)
        final, aligned = self.glossary.postprocess(zh, placeholder_map)
        # 如果 mock 小词典完全没替换，就拼一个前缀提示"（翻译-mock 占位）"
        if final == processed and not aligned:
            final = "（翻译-mock 模式）" + text[:80]
        latency = int((time.time() - t0) * 1000)
        return TranslationUnit(
            chunk_id=chunk_id, source_text=text, target_text=final,
            target_lang=target_lang, aligned_terms=aligned, latency_ms=latency,
            from_mock=True,
        )


def create_translator(mode: str = "auto", glossary: Optional[TermGlossary] = None, **kwargs) -> BaseTranslator:
    """工厂函数：按模式创建翻译器。"""
    mode = mode.lower()
    if mode == "mock":
        return MockTranslator(glossary=glossary, **kwargs)
    if mode == "dashscope":
        t = DashScopeTranslator(glossary=glossary, **kwargs)
        if not t.available:
            logger.warning("DashScope 翻译器不可用，降级为 Mock")
            return MockTranslator(glossary=glossary)
        return t
    if mode == "auto":
        t = DashScopeTranslator(glossary=glossary, **kwargs)
        if t.available:
            return t
        logger.info("auto 模式：DashScope 不可用，降级为 MockTranslator")
        return MockTranslator(glossary=glossary)
    raise ValueError(f"未知 translator mode: {mode}")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _replace_case_insensitive(s: str, old: str, new: str) -> str:
    """大小写不敏感地把 old 替换为 new（old 为英语单词时按边界匹配避免误替换）。"""
    import re
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, s)


def _guess_nonstandard_zh(entry: TermEntry) -> List[str]:
    """启发式构造常见"非标准译法"，用于译法后处理覆盖。"""
    bad: List[str] = []
    zh = entry.term_zh
    if "机制" in zh:
        bad.append(zh.replace("机制", ""))
        bad.append(zh.replace("机制", "方法"))
    if "架构" in zh:
        bad.append(zh.replace("架构", "模型"))
    if "神经网络" in zh:
        bad.append(zh.replace("神经网络", "网络"))
    # 去掉后缀修饰词（"预训练模型"→"模型" 之类）不是我们想要的，去掉
    return [b for b in bad if b.strip() and b != zh]


def _lang_name(code: str) -> str:
    return {
        "en": "英文", "en-US": "英文", "en-GB": "英文",
        "zh-CN": "中文（简体）", "zh": "中文（简体）", "zh-TW": "中文（繁体）",
        "ja": "日文", "de": "德文", "fr": "法文", "ko": "韩文", "ru": "俄文",
        "ar": "阿拉伯文", "es": "西班牙文", "it": "意大利文",
    }.get(code, f"语言 {code}")


def _mock_en_to_zh(text: str) -> str:
    """最小 mock 翻译字典，用于无 Key 时链路演示。"""
    pairs = [
        ("Transformer", "Transformer 架构"),
        ("Self-Attention", "自注意力机制"),
        ("self-attention", "自注意力机制"),
        ("Attention", "注意力"),
        ("attention", "注意力"),
        ("Neural Machine Translation", "神经机器翻译"),
        ("machine translation", "机器翻译"),
        ("Machine Translation", "机器翻译"),
        ("WMT 2014", "WMT 2014"),
        ("English-to-German", "英到德"),
        ("English-to-French", "英到法"),
        ("BLEU", "BLEU 分数"),
        ("training", "训练"),
        ("inference", "推理"),
        ("dataset", "数据集"),
        ("model", "模型"),
        ("results show", "实验结果表明"),
        ("we propose", "本文提出"),
        ("we introduce", "本文介绍"),
        ("the paper", "该论文"),
        ("this paper", "本文"),
        ("Our approach", "我们的方法"),
        ("our approach", "我们的方法"),
        ("achieves", "取得了"),
        ("outperforms", "优于"),
        ("state of the art", "当前最优方法"),
        ("State-of-the-Art", "当前最优方法"),
        ("SOTA", "SOTA"),
        ("on par with", "与…相当"),
        ("experimental", "实验"),
        ("results", "结果"),
        ("section", "章节"),
        ("related work", "相关工作"),
        ("Related Work", "相关工作"),
        ("Method", "方法"),
        ("method", "方法"),
        ("Experiments", "实验"),
        ("experiments", "实验"),
        ("Conclusion", "结论"),
        ("conclusion", "结论"),
        ("Introduction", "引言"),
        ("introduction", "引言"),
        ("Abstract", "摘要"),
        ("abstract", "摘要"),
    ]
    result = text
    for en, zh in pairs:
        result = _replace_case_insensitive(result, en, zh)
    return result


# 常见两字词后缀（中文名词后缀），用于还原后去除紧接重复
_REPEAT_SUFFIXES = (
    "机制", "架构", "模型", "方法", "网络", "损失", "分数",
    "指标", "编码", "连接", "归一化", "数据", "轮次", "策略",
    "下降", "传播", "衰减", "正则化", "验证", "曲线", "性",
    "集",
)


def _deduplicate_suffix(text: str, term_translations) -> str:
    """去除形如"自注意力机制机制是..."的叠词后缀重复。

    仅当 zh 的末尾恰好以 ``_REPEAT_SUFFIXES`` 之一结尾，且紧随其后
    在正文中再次出现同一后缀词本身（非 zh 内）时才删除重复。
    """
    for zh in term_translations:
        if not zh:
            continue
        suffix = None
        for s in _REPEAT_SUFFIXES:
            if zh.endswith(s) and len(zh) > len(s):
                suffix = s
                break
        if suffix is None:
            continue
        # 查找 "zh+suffix+..." 模式
        dup = zh + suffix
        if dup in text:
            # 仅替换首次出现（避免误改句子其他位置）
            text = text.replace(dup, zh, 1)
    return text

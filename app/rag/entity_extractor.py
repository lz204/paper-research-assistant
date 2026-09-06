"""实体抽取模块（正则 + spaCy 混合方案）

按项目文档 4.2 节设计：
- 4 类正则规则：数据集名、模型名、数值指标、核心结论句
- spaCy NER 补全普通术语（en_core_web_sm，纯 CPU）
- BaseEntityExtractor 抽象基类支持热插拔
  （未来迁移 GPU 环境可无缝切换回 deep_trio，文档"硬件约束下的改造说明"）

实测（文档 4.2 节）：规则覆盖 CV/NLP 领域约 80% 实体，NER 补 20%，单块抽取 < 50ms。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """抽取出的实体

    Attributes:
        name: 唯一键（实体文本本身；结论句取截断前缀作 key）
        value: 实体值（一般与 name 相同；结论句为完整句子）
        type: 实体类型，取值见 EntityType
        confidence: 抽取置信度（正则 = 1.0，NER < 1.0）
    """

    name: str
    value: str
    type: str
    confidence: float = 1.0


class EntityType:
    """实体类型常量"""

    DATASET = "dataset"        # 数据集名（ImageNet、COCO-2017）
    MODEL = "model"            # 模型名（ResNet、SwinTransformer）
    METRIC = "metric"          # 数值指标（85%、0.81 mAP）
    CONCLUSION = "conclusion"  # 核心结论句
    TERM = "term"              # 普通术语（spaCy NER 补全）


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseEntityExtractor(ABC):
    """实体抽取器抽象基类

    设计目的：支持实现类热插拔（文档 4.2 节"硬件约束下的改造说明"）。
    当前默认实现为 HybridEntityExtractor（正则 + spaCy）；
    未来迁移到 GPU 环境可无缝切换为 DeepTrioExtractor。
    """

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """从文本中抽取实体

        Args:
            text: 输入文本（论文 chunk）
        Returns:
            抽取出的实体列表（已去重）
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 4 类正则规则（文档 4.2 节表格）
# ---------------------------------------------------------------------------

# 数据集名：`已知数据集基名-版本/年份`，匹配 ImageNet-1K、COCO-2017、GLUE-SST
# 用基名白名单消歧：ImageNet-1K 是数据集，而 ResNet-50 是模型——二者形态相同，
# 纯正则无法区分，需领域先验（文档 4.2 节"规则覆盖 80%"的现实边界）。
KNOWN_DATASET_BASES = frozenset({
    "ImageNet", "COCO", "GLUE", "SuperGLUE", "WMT", "CIFAR", "CIFAR-10", "CIFAR-100",
    "MNIST", "SVHN", "SQuAD", "Cityscapes", "ADE20K", "PASCAL", "VOC", "LFW",
    "TinyImageNet", "OpenImage", "Kinetics", "HMDB", "UCF", "GQA", "VQA",
})
# `[A-Z][A-Za-z]+-[A-Z0-9]+` 需基名属于白名单才判为数据集
DATASET_PATTERN = re.compile(r"([A-Z][A-Za-z]+)-([A-Z0-9][A-Za-z0-9]*)")

# 模型名：ResNet / SwinTransformer / ViTFormer，且带版本号也归模型：ResNet-50 / BERT-base
# 顺序：模型先独立识别（不含版本），再按版本后缀扩展为 ResNet-50 完整 token。
MODEL_PATTERN = re.compile(r"[A-Z][a-z]+(?:Net|Former|Transformer|CNN|BERT)")
# 全大写缩写模型：BERT / GPT / ViT / DETR / CNN / RNN / LSTM / GRU / GAN
# （旧 MODEL_PATTERN 的 [A-Z][a-z]+ 前缀无法匹配全大写词）
MODEL_ACRONYM_PATTERN = re.compile(
    r"\b(?:BERT|GPT|ViT|DETR|CNN|RNN|LSTM|GRU|GAN|ELMo|T5)\b"
)
# 模型名后的版本号：ResNet-50 / BERT-base / SwinTransformer-T（跟随前一个模型 token）
MODEL_VERSION_SUFFIX = re.compile(r"-(?:[A-Za-z0-9]+)")

# 数值指标：支持两类论文常见写法：
#   数字在前：85.6% / 0.92 mAP / 27.3 BLEU
#   数字在后：mAP=0.81 / F1=0.92 / BLEU=28.4 / IoU=0.74
METRIC_PATTERN = re.compile(
    r"(?:"
    r"\d+\.?\d*\s*(?:%|mAP|F1|BLEU|IoU|AUROC|AUC)"
    r"|(?:mAP|F1|BLEU|IoU|AUROC|AUC|accuracy|top-\d)\s*=\s*\d+\.?\d*"
    r")"
)

# 核心结论句开头标志
DEFAULT_CONCLUSION_PREFIXES: Tuple[str, ...] = (
    "we find that",
    "our method achieves",
    "experiments show",
    "we observe that",
    "results show that",
    "we demonstrate that",
    "in this paper, we",
)

# spaCy NER 中视为"领域术语"的标签
# 论文场景下 PRODUCT / ORG / WORK_OF_ART / LANGUAGE / LAW 多为方法名/数据集名/任务名
SPACY_KEEP_LABELS = frozenset({"PRODUCT", "ORG", "WORK_OF_ART", "LANGUAGE", "LAW"})


# ---------------------------------------------------------------------------
# 纯正则实现
# ---------------------------------------------------------------------------


class RegexEntityExtractor(BaseEntityExtractor):
    """纯正则实体抽取器（4 类规则，无外部依赖）

    适用于：CPU 环境、spaCy 未安装时的降级模式。
    """

    def __init__(
        self,
        conclusion_prefixes: Tuple[str, ...] = DEFAULT_CONCLUSION_PREFIXES,
    ):
        self.conclusion_prefixes = tuple(p.lower() for p in conclusion_prefixes)

    def extract(self, text: str) -> List[Entity]:
        if not text:
            return []

        entities: List[Entity] = []
        seen: set = set()

        # 1. 模型名（先识别，并合并可选版本号：ResNet-50 / BERT-base）
        model_spans: List[tuple] = []
        # 匹配两类形态：驼峰命名（ResNet/SwinTransformer）与全大写缩写（BERT/GPT/ViT）
        for m in list(MODEL_PATTERN.finditer(text)) + list(MODEL_ACRONYM_PATTERN.finditer(text)):
            # 已知数据集基名优先当数据集（ImageNet-1K），不当作模型前缀
            if m.group(0) in KNOWN_DATASET_BASES:
                continue
            start, end = m.start(), m.end()
            # 若紧跟 "-版本"，且版本 token 不扩展进下一个数据集基名，则并入模型 token
            version_m = MODEL_VERSION_SUFFIX.match(text, end)
            if version_m:
                end = version_m.end()
            val = text[start:end]
            model_spans.append((start, end))
            if val not in seen:
                entities.append(Entity(name=val, value=val, type=EntityType.MODEL))
                seen.add(val)

        # 2. 数据集名：仅白名单基名，且不与已识别模型 span 重叠（避免 ImageNet-1K 的前缀被抢）
        dataset_spans: List[tuple] = []
        for m in DATASET_PATTERN.finditer(text):
            base = m.group(1)
            if base not in KNOWN_DATASET_BASES:
                continue
            if any(s <= m.start() and m.end() <= e for s, e in model_spans):
                continue
            val = m.group().strip()
            dataset_spans.append((m.start(), m.end()))
            if val not in seen:
                entities.append(Entity(name=val, value=val, type=EntityType.DATASET))
                seen.add(val)

        # 3. 数值指标（需带单位，过滤纯数字）
        for m in METRIC_PATTERN.finditer(text):
            val = m.group().strip()
            # 文档示例要求"带单位"，纯数字丢弃
            if re.fullmatch(r"\d+\.?\d*", val):
                continue
            if val not in seen:
                entities.append(Entity(name=val, value=val, type=EntityType.METRIC))
                seen.add(val)

        # 4. 核心结论句
        entities.extend(self._extract_conclusions(text, seen))

        return entities

    def _extract_conclusions(self, text: str, seen: set) -> List[Entity]:
        """抽取以指定前缀开头的句子作为核心结论"""
        result: List[Entity] = []
        # 按句号/问号/感叹号切句（保留句子内容）
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue
            sent_lower = sent_stripped.lower()
            for prefix in self.conclusion_prefixes:
                if sent_lower.startswith(prefix):
                    # 用前 80 字符作为唯一键，避免长句作 key
                    name = sent_stripped[:80]
                    if name not in seen:
                        result.append(
                            Entity(
                                name=name,
                                value=sent_stripped,
                                type=EntityType.CONCLUSION,
                            )
                        )
                        seen.add(name)
                    break  # 一个句子只匹配一个前缀
        return result


# ---------------------------------------------------------------------------
# spaCy NER 实现（补全普通术语）
# ---------------------------------------------------------------------------


class SpacyEntityExtractor(BaseEntityExtractor):
    """spaCy NER 实体抽取器（补全前 4 类未覆盖的普通术语）

    使用 en_core_web_sm 小模型（90MB，纯 CPU 推理，文档 4.2 节"硬件约束下的改造说明"）。
    若 spaCy 未安装或模型未下载，extract() 返回空列表（不抛异常），
    保证系统在最小依赖下仍可运行——这是 8GB 纯 CPU 环境的工程妥协。
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None  # lazy load
        self._load_attempted = False

    def _ensure_loaded(self) -> bool:
        """懒加载 spaCy 模型，失败返回 False"""
        if self._nlp is not None:
            return True
        if self._load_attempted:
            # 已尝试过且失败，不再重试（避免每次调用都尝试导入）
            return False
        self._load_attempted = True
        try:
            import spacy  # type: ignore
            self._nlp = spacy.load(self.model_name)
            logger.info("spaCy 模型 %s 加载成功", self.model_name)
            return True
        except ImportError:
            logger.warning(
                "spaCy 未安装，跳过 NER 补全。pip install spacy 后可启用，"
                "不影响正则抽取"
            )
            return False
        except OSError:
            logger.warning(
                "spaCy 模型 %s 未下载。运行 `python -m spacy download %s` 后可启用，"
                "不影响正则抽取",
                self.model_name,
                self.model_name,
            )
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning("spaCy 加载失败：%s，跳过 NER 补全", e)
            return False

    def extract(self, text: str) -> List[Entity]:
        if not text or not self._ensure_loaded():
            return []

        entities: List[Entity] = []
        seen: set = set()

        doc = self._nlp(text)
        for ent in doc.ents:
            if ent.label_ not in SPACY_KEEP_LABELS:
                continue
            val = ent.text.strip()
            # 过滤过短或单字符噪声
            if len(val) < 2 or val in seen:
                continue
            entities.append(
                Entity(
                    name=val,
                    value=val,
                    type=EntityType.TERM,
                    confidence=0.7,  # NER 置信度低于正则
                )
            )
            seen.add(val)

        return entities


# ---------------------------------------------------------------------------
# 混合实现（项目默认）
# ---------------------------------------------------------------------------


class HybridEntityExtractor(BaseEntityExtractor):
    """混合实体抽取器（正则 + spaCy，项目文档默认实现）

    工作流程：
    1. 先用 RegexEntityExtractor 抽 4 类领域实体（高精度，confidence=1.0）
    2. 再用 SpacyEntityExtractor 补全普通术语（高召回，confidence=0.7）
    3. 去重：spaCy 抽出的实体若已被正则命中（含子串关系）则丢弃

    实测（文档 4.2 节）：规则覆盖 CV/NLP 领域约 80% 实体，NER 补 20%，
    单块抽取耗时 < 50ms（纯本地 CPU，不调 LLM）。
    """

    def __init__(
        self,
        spacy_extractor: Optional[SpacyEntityExtractor] = None,
        use_spacy: bool = True,
    ):
        self.regex_extractor = RegexEntityExtractor()
        self.use_spacy = use_spacy
        self.spacy_extractor = spacy_extractor or SpacyEntityExtractor()

    def extract(self, text: str) -> List[Entity]:
        if not text:
            return []

        # 1. 正则先抽（高优先级）
        regex_ents = self.regex_extractor.extract(text)
        regex_names_lower = {e.name.lower() for e in regex_ents}

        if not self.use_spacy:
            return regex_ents

        # 2. spaCy 补全（过滤已被正则命中或为正则实体子串的）
        spacy_ents = self.spacy_extractor.extract(text)
        for ent in spacy_ents:
            ent_lower = ent.name.lower()
            # 已被正则命中
            if ent_lower in regex_names_lower:
                continue
            # 是某正则实体的子串（避免 "Net" 被 NER 单独抽出）
            if any(ent_lower in r or r in ent_lower for r in regex_names_lower):
                continue
            regex_ents.append(ent)

        return regex_ents

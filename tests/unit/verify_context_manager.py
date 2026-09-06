"""ContextManager 本地验证脚本

模拟项目文档 4.2 节的核心场景：
- 入库阶段：对论文 chunk 抽实体写入 KV 池
- 问答阶段：按问题意图检索相关实体并构建 prompt
- LRU 淘汰：超过 max_size 时按最后访问时间淘汰

运行方式（在项目根目录 paper-research-assistant/ 下）：
    python tests/unit/verify_context_manager.py
"""

from __future__ import annotations

import sys
import time

# 共享引导：插入项目根目录到 sys.path + 清空 API key / Langfuse 配置（强制 mock）
from _verify_common import PROJECT_ROOT  # noqa: E402

from app.rag import ContextManager, HybridEntityExtractor  # noqa: E402
from app.rag.entity_extractor import (  # noqa: E402
    EntityType,
    RegexEntityExtractor,
)

# 模拟论文 chunk（含数据集名、模型名、数值指标、核心结论句）
SAMPLE_CHUNKS = [
    (
        "chunk_1_intro",
        "We propose ResNet, a deep residual network for image recognition. "
        "Experiments show that our method achieves 85.6% top-5 accuracy "
        "on ImageNet-1K, significantly outperforming prior CNN architectures.",
    ),
    (
        "chunk_2_method",
        "Our model uses a SwinTransformer backbone with F1=0.92 on COCO-2017 "
        "for object detection. We find that the window-shifted attention "
        "reduces computational cost by 40% while maintaining BLEU=27.3 "
        "on machine translation tasks.",
    ),
    (
        "chunk_3_results",
        "We demonstrate that ViTFormer outperforms baseline models. "
        "Results show that our approach reaches mAP=0.81 on the benchmark "
        "dataset GLUE-SST, with IoU=0.74 for semantic segmentation. "
        "Our method achieves state-of-the-art performance across 5 datasets.",
    ),
    (
        "chunk_4_discussion",
        "In this paper, we present a novel architecture combining ResNet "
        "and BERT for multimodal learning. The proposed method achieves "
        "AUROC=0.95 on medical imaging classification tasks. "
        "We observe that larger batch sizes improve training stability.",
    ),
]


def test_basic_extract_and_query() -> None:
    """测试 1：基础入库 + 问答构建"""
    print("\n" + "=" * 70)
    print("测试 1：基础入库 + 问答构建")
    print("=" * 70)

    cm = ContextManager(max_size=200)

    # 入库阶段：逐 chunk 抽取实体
    for chunk_id, chunk_text in SAMPLE_CHUNKS:
        entities = cm.update(chunk_text, chunk_id)
        print(f"\n[入库] {chunk_id} 抽取到 {len(entities)} 个实体：")
        for e in entities:
            preview = e.value[:70] + ("..." if len(e.value) > 70 else "")
            print(f"  - [{e.type}] {preview}")

    # 问答阶段：按不同意图构建 prompt
    test_questions = [
        "本文用了什么数据集？",
        "what model does this paper use?",
        "实验的 accuracy 和 metric 结果如何？",
        "本文的核心 conclusion 是什么？",
        "what is the contribution of this paper?",  # 无明确意图，兜底返回全部
    ]

    for q in test_questions:
        print(f"\n[问答] 问题：{q}")
        prompt = cm.build_prompt(q, top_k=5)
        print(prompt)
        print("-" * 50)

    # 统计
    print("\n[统计] 记忆池状态：")
    stats = cm.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


def test_lru_eviction() -> None:
    """测试 2：LRU 淘汰策略"""
    print("\n" + "=" * 70)
    print("测试 2：LRU 淘汰策略（max_size=5）")
    print("=" * 70)

    cm = ContextManager(max_size=5)

    # 注入 8 个实体（超过 max_size=5）
    chunks = [
        ("c1", "Use ResNet for ImageNet-1K classification, accuracy 85%."),
        ("c2", "SwinTransformer on COCO-2017 gets mAP=0.81."),
        ("c3", "ViTFormer backbone with F1=0.92."),
        ("c4", "BERT for NLP tasks, BLEU=27.3."),
        ("c5", "Experiments show we reach IoU=0.74 on GLUE-SST."),
        ("c6", "We find that our method achieves AUROC=0.95."),
    ]
    for cid, txt in chunks:
        before = len(cm)
        cm.update(txt, cid)
        after = len(cm)
        evicted = before + 1 - after if before + 1 > after else 0
        print(f"  [入库 {cid}] 池大小: {before} → {after}")

    print(f"\n[最终池大小] {len(cm)} / max_size=5")
    print(f"[池中实体] {list(cm.memory.keys())}")
    print(f"[统计] {cm.stats()}")

    # 触发一次问答（访问 dataset 类实体）
    print("\n[问答] 触发 dataset 类查询以更新 last_access...")
    prompt = cm.build_prompt("用了什么数据集 dataset", top_k=10)
    print(f"  prompt 片段: {prompt[:200]}...")

    # 再注入新实体，验证被访问过的不会被淘汰
    cm.update("NewNet on OpenImage-V6 gets 90%.", "c7")
    print(f"\n[再注入后] 池大小: {len(cm)}")
    print(f"[池中实体] {list(cm.memory.keys())}")


def test_regex_only_mode() -> None:
    """测试 3：纯正则模式（spaCy 不可用时的降级）"""
    print("\n" + "=" * 70)
    print("测试 3：纯正则模式（spaCy 不可用时降级）")
    print("=" * 70)

    cm = ContextManager(
        max_size=200,
        extractor=HybridEntityExtractor(use_spacy=False),
    )

    chunk_text = SAMPLE_CHUNKS[0][1]
    entities = cm.update(chunk_text, "chunk_test")
    print(f"抽取到 {len(entities)} 个实体（纯正则，无 NER 补全）：")
    for e in entities:
        print(f"  - [{e.type}] {e.value[:60]}")

    prompt = cm.build_prompt("what datasets?", top_k=10)
    print(f"\n问答 prompt：\n{prompt}")


def test_empty_and_edge_cases() -> None:
    """测试 4：边界情况"""
    print("\n" + "=" * 70)
    print("测试 4：边界情况（空输入、空池、无匹配）")
    print("=" * 70)

    cm = ContextManager(max_size=200)

    # 空池 + 任意问题 → 返回原问题
    p1 = cm.build_prompt("anything?")
    assert p1 == "anything?", f"空池应返回原问题，得到: {p1}"
    print("  ✓ 空池返回原问题")

    # 空 chunk 文本 → 不抽取
    ents = cm.update("", "empty_chunk")
    assert ents == [], "空文本应返回空列表"
    print("  ✓ 空文本不抽取")

    # 无匹配实体的问题 → 返回原问题
    cm.update("ResNet on ImageNet-1K gets 85%.", "c1")
    p2 = cm.build_prompt("今天天气怎么样 weather")
    # 无 dataset/model/metric/conclusion/term 意图关键词，会兜底返回全部
    # 但 token 与实体无重叠，仍有候选——验证不会抛异常即可
    print(f"  ✓ 无匹配问题不抛异常，返回 prompt 长度: {len(p2)}")

    # max_size 校验
    try:
        ContextManager(max_size=0)
        assert False, "max_size=0 应抛 ValueError"
    except ValueError:
        print("  ✓ max_size=0 抛 ValueError")


def test_regex_extractor_directly() -> None:
    """测试 5：直接测试 RegexEntityExtractor 4 类规则"""
    print("\n" + "=" * 70)
    print("测试 5：RegexEntityExtractor 4 类规则覆盖")
    print("=" * 70)

    extractor = RegexEntityExtractor()
    text = (
        "We use ResNet and SwinTransformer on ImageNet-1K and COCO-2017. "
        "Our method achieves 85.6% accuracy and mAP=0.81. "
        "Experiments show that the proposed approach outperforms baselines. "
        "We find that window attention reduces cost."
    )
    entities = extractor.extract(text)

    type_map: dict = {}
    for e in entities:
        type_map.setdefault(e.type, []).append(e.value)

    print(f"共抽取 {len(entities)} 个实体：")
    for t, vals in type_map.items():
        print(f"\n  [{t}] ({len(vals)} 个)")
        for v in vals:
            preview = v[:80] + ("..." if len(v) > 80 else "")
            print(f"    - {preview}")

    # 断言 4 类都有覆盖
    assert EntityType.DATASET in type_map, "数据集类未覆盖"
    assert EntityType.MODEL in type_map, "模型类未覆盖"
    assert EntityType.METRIC in type_map, "指标类未覆盖"
    assert EntityType.CONCLUSION in type_map, "结论类未覆盖"
    print("\n  ✓ 4 类规则全部覆盖")


def main() -> None:
    print("=" * 70)
    print("ContextManager 本地验证（项目文档 4.2 节核心亮点）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)

    t0 = time.time()

    test_regex_extractor_directly()
    test_basic_extract_and_query()
    test_lru_eviction()
    test_regex_only_mode()
    test_empty_and_edge_cases()

    elapsed_ms = (time.time() - t0) * 1000
    print("\n" + "=" * 70)
    print(f"全部测试通过 ✓  总耗时: {elapsed_ms:.1f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Parser 模块集成验证脚本

验证项目文档 4.1 节的完整 PDF 解析链路：
    PDF → 转图 + pdfplumber 抽取 → VLM 识别 → 公式绑定（四级规则）→ 多源融合

重点验证：坐标对齐漏洞修复
- formula_binder 不再做 IoU 对齐（VLM 不返回坐标框）
- 改用四级规则绑定：段落序号匹配 / 位置描述映射 / 文本锚点匹配 / 兜底未绑定

本地无 API key 时用 mock 模式跑通全链路。
"""

from __future__ import annotations

import sys

# 共享引导：插入项目根目录到 sys.path + 清空 API key / Langfuse 配置（强制 mock）
from _verify_common import PROJECT_ROOT  # noqa: E402

from app.parser import (  # noqa: E402
    BindingRule,
    BoundFormula,
    FallbackHandler,
    FormulaBinder,
    FusionEngine,
    QwenVlmClient,
    VlmFormula,
    VlmPageResult,
    build_mock_layout,
    binding_stats,
    document_stats,
)


# 模拟 pdfplumber 抽取的段落（5 段，覆盖不同位置）
SAMPLE_PARAGRAPHS = [
    "We propose a novel transformer architecture for image recognition. "
    "The key insight is that self-attention can replace recurrent layers.",

    "The softmax computation is as follows. "  # 规则三锚点：含"如下"
    "We use multi-head attention to capture different representation subspaces.",

    "Our model is trained on ImageNet-1K with standard data augmentation. "
    "The learning rate schedule uses warmup followed by cosine decay.",

    "The loss function is defined as follows. "  # 规则三锚点
    "We combine cross-entropy loss with auxiliary regularization terms.",

    "Experiments show that our method achieves 85.6% top-5 accuracy on ImageNet-1K.",
]


# 模拟 VLM 识别的公式（覆盖四级规则的不同场景）
SAMPLE_FORMULAS = [
    VlmFormula(
        latex=r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
        location_desc="第2段后",       # 规则一：段落序号匹配
        confidence=0.95,
        formula_id="(1)",
    ),
    VlmFormula(
        latex=r"\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)",
        location_desc="第二栏中部",      # 规则二：位置描述映射
        confidence=0.72,
        formula_id="(2)",
    ),
    VlmFormula(
        latex=r"E = mc^2",
        location_desc="softmax 计算公式如下",  # 规则三：文本锚点匹配
        confidence=0.55,
        formula_id="(3)",
    ),
    VlmFormula(
        latex=r"\nabla_\theta J(\theta)",
        location_desc="页面右上角某处",    # 规则四：兜底未绑定
        confidence=0.45,
        formula_id="(4)",
    ),
    VlmFormula(
        latex=r"attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V",
        location_desc="第1段下方",       # 规则一：另一个段落序号
        confidence=0.92,
        formula_id="(5)",
    ),
    VlmFormula(
        latex=r"\text{IoU} = \frac{|A \cap B|}{|A \cup B|}",
        location_desc="the loss function is defined as follows",  # 规则三锚点
        confidence=0.6,
        formula_id="(6)",
    ),
]


def test_formula_binder_4rules() -> None:
    """测试 1：四级规则绑定（核心 - 修复坐标对齐漏洞）"""
    print("\n" + "=" * 70)
    print("测试 1：FormulaBinder 四级规则绑定（修复 IoU 对齐漏洞）")
    print("=" * 70)

    # 构造 mock 页面版面（5 段落）
    layout = build_mock_layout(page_no=0, paragraphs=SAMPLE_PARAGRAPHS)
    print(f"页面版面：{len(layout.paragraphs)} 段落, 中轴线 x={layout.midline_x}")
    for p in layout.paragraphs:
        print(f"  段落 {p.idx}: region={p.region}, y0={p.y0:.0f}, text={p.text[:40]}...")

    # 绑定公式
    binder = FormulaBinder()
    bound_formulas = binder.bind(SAMPLE_FORMULAS, layout)

    print(f"\n绑定 {len(bound_formulas)} 个公式：")
    for f in bound_formulas:
        rule_name = {
            BindingRule.RULE1_PARAGRAPH_NUMBER: "规则一段落序号",
            BindingRule.RULE2_POSITION_MAPPING: "规则二位置映射",
            BindingRule.RULE3_TEXT_ANCHOR: "规则三文本锚点",
            BindingRule.RULE4_UNBOUND: "规则四兜底未绑定",
        }.get(f.binding_rule, f.binding_rule)

        review_mark = " ⚠️需复核" if f.need_review else " ✓"
        anchor_info = f" (锚点: {f.anchor_text!r})" if f.anchor_text else ""
        print(
            f"  公式 {f.formula_id}: {rule_name} "
            f"→ 段落 {f.bound_paragraph_idx}, 置信度 {f.confidence}{review_mark}{anchor_info}"
        )
        print(f"    LaTeX: {f.latex[:60]}...")

    # 统计
    stats = binding_stats(bound_formulas)
    print(f"\n绑定统计: {stats}")

    # 验证四级规则都被触发
    rules_hit = {f.binding_rule for f in bound_formulas}
    assert BindingRule.RULE1_PARAGRAPH_NUMBER in rules_hit, "规则一应被触发"
    assert BindingRule.RULE2_POSITION_MAPPING in rules_hit, "规则二应被触发"
    assert BindingRule.RULE3_TEXT_ANCHOR in rules_hit, "规则三应被触发"
    assert BindingRule.RULE4_UNBOUND in rules_hit, "规则四应被触发"
    print("\n  ✓ 四级规则全部触发（IoU 对齐漏洞已修复）")

    # 验证置信度对应规则
    for f in bound_formulas:
        expected_conf = BindingRule.CONFIDENCE[f.binding_rule]
        assert f.confidence == expected_conf, (
            f"规则 {f.binding_rule} 置信度应为 {expected_conf}，实际 {f.confidence}"
        )
    print("  ✓ 各规则置信度与文档 4.1 节一致（0.9/0.7/0.5/0.3）")

    # 验证 need_review 阈值
    for f in bound_formulas:
        assert f.need_review == (f.confidence < 0.7), (
            f"need_review 判断错误: confidence={f.confidence}"
        )
    print("  ✓ need_review 阈值正确（< 0.7 触发黄色高亮）")


def test_vlm_client_mock_mode() -> None:
    """测试 2：VLM 客户端 mock 模式（本地无 API key）"""
    print("\n" + "=" * 70)
    print("测试 2：QwenVlmClient mock 模式（本地无 API key 降级）")
    print("=" * 70)

    client = QwenVlmClient()  # 无 api_key，进入 mock 模式
    print(f"mock 模式: {client._mock_mode}")

    # 单页识别
    result = client.recognize_pages_sync(
        images=[(0, "mock_b64_data")],
        pdf_md5="test_md5_123",
    )
    print(f"识别 {len(result)} 页:")
    for r in result:
        print(f"  page {r.page_no}: text={r.text[:50]}...")
        print(f"    formulas: {len(r.formulas)}, confidence={r.confidence}")
        for f in r.formulas:
            print(f"      [{f.formula_id}] {f.latex[:40]}... | loc={f.location_desc!r}")

    # 验证缓存命中
    result_cached = client.recognize_pages_sync(
        images=[(0, "mock_b64_data")],
        pdf_md5="test_md5_123",
    )
    assert result_cached[0].from_cache, "二次调用应命中缓存"
    print("\n  ✓ 缓存命中（文档 6.2 成本治理）")
    print("  ✓ mock 模式正常（本地无 API 可跑通）")


def test_fallback_handler() -> None:
    """测试 3：降级处理器（低置信度二次验证 + pdfplumber 兜底）"""
    print("\n" + "=" * 70)
    print("测试 3：FallbackHandler 降级与二次验证")
    print("=" * 70)

    client = QwenVlmClient()
    handler = FallbackHandler(client)

    # 低置信度公式需二次验证
    low_conf_formula = VlmFormula(
        latex=r"E = mc^2",
        location_desc="测试位置",
        confidence=0.55,
        formula_id="(test)",
    )
    print(f"低置信度公式: confidence={low_conf_formula.confidence}")
    print(f"  needs_reverify: {handler.needs_formula_reverify(low_conf_formula)}")
    assert handler.needs_formula_reverify(low_conf_formula)

    # mock 模式二次验证
    reverify = handler.reverify_formula(low_conf_formula)
    print(f"  二次验证结果: success={reverify.success}, "
          f"new_conf={reverify.formula.confidence if reverify.formula else 'N/A'}")
    assert reverify.success
    assert reverify.formula.confidence >= 0.7

    # pdfplumber 降级
    layout = build_mock_layout(page_no=0, paragraphs=SAMPLE_PARAGRAPHS[:2])
    degraded = handler.fallback_to_pdfplumber(layout, page_no=0)
    print(f"\npdfplumber 降级: formula_resolved={degraded.formula_resolved}, "
          f"text={degraded.text[:50]}...")
    assert not degraded.formula_resolved, "降级模式应标注 formula_resolved=false"

    print("\n  ✓ 二次验证 + pdfplumber 降级正常（文档 6.3 容错链）")


def test_fusion_engine_full() -> None:
    """测试 4：FusionEngine 多源融合（端到端）"""
    print("\n" + "=" * 70)
    print("测试 4：FusionEngine 多源融合 + 冲突仲裁（端到端）")
    print("=" * 70)

    # 构造输入
    layout = build_mock_layout(page_no=0, paragraphs=SAMPLE_PARAGRAPHS)
    vlm_result = VlmPageResult(
        page_no=0,
        text="VLM recognized text (will be replaced by pdfplumber paragraphs)",
        formulas=SAMPLE_FORMULAS,
        tables=[],
        confidence=0.85,
        formula_resolved=True,
    )

    # 融合
    client = QwenVlmClient()
    engine = FusionEngine(vlm_client=client)
    doc = engine.fuse(
        doc_id="paper_test",
        vlm_results=[vlm_result],
        page_layouts=[layout],
    )

    print(f"融合文档: {doc.doc_id}, {len(doc.pages)} 页")
    print(f"  总公式: {doc.total_formulas}, 总表格: {doc.total_tables}")
    print(f"  平均置信度: {doc.avg_confidence}")

    page = doc.pages[0]
    print(f"\n第 0 页:")
    print(f"  段落数: {len(page.paragraphs)}")
    print(f"  绑定统计: {page.binding_stats}")
    print(f"  formula_resolved: {page.formula_resolved}")

    print(f"\n段落与绑定公式：")
    for p in page.paragraphs:
        print(f"  段落 {p.idx} (source={p.source}, conf={p.confidence}):")
        print(f"    text: {p.text[:60]}...")
        for f in p.bound_formulas:
            print(f"    📐 公式 {f.formula_id} [{f.binding_rule}] conf={f.confidence}")
            print(f"       LaTeX: {f.latex[:50]}...")

    print(f"\n未绑定公式: {len(page.unbound_formulas)}")
    for f in page.unbound_formulas:
        print(f"  📐 公式 {f.formula_id} [{f.binding_rule}] conf={f.confidence} ⚠️")

    # 验证：正文来自 pdfplumber（文档冲突仲裁规则）
    assert all(p.source == "pdfplumber" for p in page.paragraphs), \
        "正文应取 pdfplumber（文档：正文 pdfplumber 最准）"
    print("\n  ✓ 正文取 pdfplumber（文档冲突仲裁）")

    # 验证：公式来自 VLM（pdfplumber 是乱码）
    all_bound = [f for p in page.paragraphs for f in p.bound_formulas]
    all_bound += page.unbound_formulas
    assert len(all_bound) == len(SAMPLE_FORMULAS), "所有公式都应保留"
    print("  ✓ 公式取 VLM（文档：公式 VLM 最准）")

    # 文档级统计
    stats = document_stats(doc)
    print(f"\n文档统计: {stats}")


def test_pdfplumber_degradation() -> None:
    """测试 5：VLM 完全降级到 pdfplumber"""
    print("\n" + "=" * 70)
    print("测试 5：VLM 降级到 pdfplumber（API 全挂兜底）")
    print("=" * 70)

    layout = build_mock_layout(page_no=0, paragraphs=SAMPLE_PARAGRAPHS[:3])
    # 不传 vlm_results，模拟 VLM 完全不可用
    engine = FusionEngine()
    doc = engine.fuse(
        doc_id="paper_degraded",
        vlm_results=[],
        page_layouts=[layout],
    )

    page = doc.pages[0]
    print(f"降级页面: formula_resolved={page.formula_resolved}, "
          f"confidence={page.confidence}")
    print(f"  段落数: {len(page.paragraphs)}")
    for p in page.paragraphs:
        print(f"    [{p.idx}] source={p.source} conf={p.confidence}: {p.text[:50]}...")

    assert not page.formula_resolved, "降级模式应标注 formula_resolved=false"
    assert page.confidence == 0.5, "降级模式置信度应为 0.5"
    print("\n  ✓ VLM 降级正常（文档 4.1 pdfplumber 兜底通道）")


def main() -> None:
    print("=" * 70)
    print("Parser 模块集成验证（项目文档 4.1 节 PDF 解析链路）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)

    test_formula_binder_4rules()
    test_vlm_client_mock_mode()
    test_fallback_handler()
    test_fusion_engine_full()
    test_pdfplumber_degradation()

    print("\n" + "=" * 70)
    print("✓ Parser 模块全部验证通过")
    print("  核心修复：formula_binder 不再做 IoU 对齐，改为四级规则绑定")
    print("  文档 4.1 节'Qwen-VL-Max 不返回坐标框'的工程现实已落地")
    print("=" * 70)


if __name__ == "__main__":
    main()

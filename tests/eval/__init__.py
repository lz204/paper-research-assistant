"""自动化评测模块（文档 6.4 节：没有数据的优化都是耍流氓）

文档 6.4 节评测流水线：
    数据集 (200 篇 arXiv + 400 条 QA)
        → 跑问答，记录 {answer / citations}
        → Rouge-L（摘要质量）
        → 引用准确率（cite_precision / recall / F1）
        → 公式乱码率（公式类问题的符号完整性）
        → 输出对比报告 JSON（与基线对比，文档 6.4 节"基线"）

可选依赖降级（文档 6.3 节）：
- rouge-score 未装 → 简易 LCS 版 Rouge-L
- nltk 未装 → 按汉字/英文词元分词（兼容中英混合）
- pytest 未装 → 纯脚本 + assert
"""

__all__: list = []

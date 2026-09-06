"""自动化评测 QA 数据集（文档 6.4 节）

文档 6.4 节：自建 200 篇 arXiv 论文（多语覆盖）+ 400 条标注问答，
标注内容：answer、引用原文位置、公式正确性（人工）。

实际生产环境 200 篇 + 400 条由人工标注或 LLM 生成+人工复核，
这里提供最小可用的内置样例（3 篇论文，10 条 QA），
保证 `tests/eval/run_eval.py` 无 key 也能跑通。

有真实 DashScope API key 时，可运行 `python tests/qa_dataset/build_dataset.py --expand 50`
调用 DashScope 扩充为 50 篇，降低人工标注成本。
"""

# 占位，无函数/类需导出
__all__: list = []

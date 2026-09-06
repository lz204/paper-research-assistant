"""verify_*.py 集成验证脚本的共享引导模块。

各 verify 脚本顶部的样板代码（插入项目根目录 sys.path + 清空 API key）高度重复，
统一收敛到本模块：

1. 项目根目录加入 sys.path：让脚本在任意目录执行时都能 ``import app.*``；
2. 在 import app 之前清空 DashScope / Langfuse 环境变量：保证 mock 模式（与
   tests/conftest.py 的 pytest 隔离策略一致），避免本地 .env 的真实 key 让脚本
   误走真实 API（真实调用由 tests/e2e/ 与 verify_completion_modules.py 负责）；
3. 统一 mock 联调常用的两篇论文语料（SAMPLE_PAPER_1 / SAMPLE_PAPER_2）及其
   分块 / 文献信息构建函数，供 verify_infrastructure 与 verify_review_pipeline
   复用，消除两个脚本间的重复样板。

注意：verify_completion_modules.py 设计为"有 Key 时自动切真实模式"，刻意不引入
本模块，以免环境变量清空破坏其真实链路验证。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目根目录：本文件位于 tests/unit/，上溯两级即为 paper-research-assistant/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _k in ("DASHSCOPE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
    os.environ.pop(_k, None)
    os.environ[_k] = ""


# ---------------------------------------------------------------------------
# mock 联调共享语料：两篇论文全文（含章节标题、数据集名、模型名、指标）
# 语料同时服务于 verify_infrastructure（主流程编排）与
# verify_review_pipeline（综述链路），保证两条验证链对同一份数据。
# ---------------------------------------------------------------------------
SAMPLE_PAPER_1 = """\
# Attention Is All You Need

## 1. Introduction

We propose ResNet, a deep residual network for image recognition.
Experiments show that our method achieves 85.6% top-5 accuracy on ImageNet-1K,
significantly outperforming prior CNN architectures.

## 2. Method

Our model uses a SwinTransformer backbone with F1=0.92 on COCO-2017 for object
detection. We find that the window-shifted attention reduces computational cost
by 40% while maintaining BLEU=27.3 on machine translation tasks.

## 3. Results

Results show that our approach reaches mAP=0.81 on the benchmark dataset
GLUE-SST, with IoU=0.74 for semantic segmentation.
"""

SAMPLE_PAPER_2 = """\
# BERT: Pre-training of Deep Bidirectional Transformers

## 1. Introduction

We introduce BERT, a new language representation model designed to pre-train
deep bidirectional representations from unlabeled text.

## 2. Method

Our model uses a Transformer encoder architecture. We find that the masked
language modeling objective improves downstream task performance.

## 3. Results

Experiments show that our method achieves state-of-the-art performance on
GLUE-SST with accuracy 92.7%. The model is evaluated on ImageNet-1K for
multimodal understanding.
"""


def build_doc_chunks() -> dict:
    """分块两篇论文，返回 {doc_id: [Chunk, ...]}"""
    from app.rag import HierarchicalChunker  # noqa: E402（延迟导入，避免引导期加载重依赖）

    chunker = HierarchicalChunker(chunk_size=256, overlap=64)
    chunks_1 = chunker.chunk(SAMPLE_PAPER_1, doc_id="paper_1")
    chunks_2 = chunker.chunk(SAMPLE_PAPER_2, doc_id="paper_2")
    return {"paper_1": chunks_1, "paper_2": chunks_2}


def build_doc_infos() -> list:
    """构造文献信息列表（用于大纲规划 / 综述主流程）"""
    return [
        {"doc_id": "paper_1", "title": "Attention Is All You Need"},
        {"doc_id": "paper_2", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
    ]

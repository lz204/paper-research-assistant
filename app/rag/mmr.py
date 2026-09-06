"""MMR（最大边际相关性）去重算法（项目文档 4.3 节）

文档原伪代码：
    def mmr_select(candidates, query_vec, lambda_=0.7, k=15):
        selected = []
        for _ in range(k):
            scores = []
            for c in candidates:
                rel = cosine(c.vec, query_vec)
                penalty = max([cosine(c.vec, s.vec) for s in selected], default=0)
                mmr = lambda_ * rel - (1 - lambda_) * penalty
                scores.append((c, mmr))
            best = max(scores, key=lambda x: x[1])[0]
            selected.append(best)
            candidates.remove(best)
        return selected

效果（文档 4.3 节）：
- 去重前每篇综述平均 4.2 处重复观点（重复占比 38%）
- 去重后降到 0.8 处（占比 9%）
- lambda_=0.7 是调参后的值（0.5 过度惩罚漏关键文献，0.9 惩罚不足去重差）

本模块严格实现文档伪代码，并扩展：
- 输入用 dataclass 封装（避免裸 list 难以理解）
- 余弦相似度数值稳定（处理零向量）
- 支持 max_iter 提前终止（候选数 < k 时不会报错）
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class MMRItem:
    """MMR 候选项

    Attributes:
        id: 唯一标识（chunk_id / doc_id）
        vec: 向量（用于计算相似度）
        payload: 原始数据（chunk 文本、metadata 等），MMR 不解析
        score: 与 query 的初始相关性分数（可选，由检索器给出）
    """

    id: str
    vec: np.ndarray
    payload: object = None
    score: float = 0.0


# ---------------------------------------------------------------------------
# 相似度函数
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，数值稳定（处理零向量）

    Returns:
        [-1, 1] 的相似度；任一向量为零时返回 0.0
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# MMR 选择器（严格实现文档伪代码）
# ---------------------------------------------------------------------------


def mmr_select(
    candidates: Sequence[MMRItem],
    query_vec: np.ndarray,
    lambda_: float = 0.7,
    k: int = 15,
) -> List[MMRItem]:
    """最大边际相关性选择（项目文档 4.3 节原伪代码实现）

    Args:
        candidates: 候选项列表（每个含向量和 id）
        query_vec: 查询向量
        lambda_: 相关性 vs 多样性的权衡系数
            - 1.0：只看相关性（退化为纯向量排序）
            - 0.0：只看多样性（完全去重）
            - 0.7：文档调参后的默认值
        k: 选择的数量
    Returns:
        选中的 MMRItem 列表（按选择顺序排列）
    """
    if not candidates or k <= 0:
        return []

    # 文档伪代码：candidates.remove(best) 会修改入参，这里复制一份避免副作用
    remaining = list(candidates)
    selected: List[MMRItem] = []
    target_k = min(k, len(remaining))

    for _ in range(target_k):
        scores: List[tuple] = []
        for c in remaining:
            # 相关性：候选项与 query 的余弦相似度
            rel = cosine_similarity(c.vec, query_vec)

            # 多样性惩罚：已选项中与候选项的最大相似度
            if selected:
                penalty = max(cosine_similarity(c.vec, s.vec) for s in selected)
            else:
                penalty = 0.0  # 第一个候选无惩罚

            # MMR 分数
            mmr_score = lambda_ * rel - (1.0 - lambda_) * penalty
            scores.append((c, mmr_score))

        # 选 MMR 分数最高的
        best, best_score = max(scores, key=lambda x: x[1])
        selected.append(best)
        remaining.remove(best)

        logger.debug(
            "mmr.select",
            extra={
                "selected_id": best.id,
                "mmr_score": round(best_score, 4),
                "remaining": len(remaining),
            },
        )

    return selected


# ---------------------------------------------------------------------------
# 封装类（带预计算的 query 相关性，可选优化）
# ---------------------------------------------------------------------------


class MMRSelector:
    """MMR 选择器封装

    相比裸函数 mmr_select，封装类提供：
    - 预计算候选向量矩阵（多次调用时省重复 norm 计算）
    - 批量余弦相似度（numpy 向量化）
    - 适用于综述生成场景（10-12 篇文献 × 数十 chunk）

    用法：
        selector = MMRSelector(lambda_=0.7)
        selected = selector.select(candidates, query_vec, k=15)
    """

    def __init__(self, lambda_: float = 0.7):
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ 必须在 [0, 1] 区间，收到 {lambda_}")
        self.lambda_ = lambda_

    def select(
        self,
        candidates: Sequence[MMRItem],
        query_vec: np.ndarray,
        k: int = 15,
    ) -> List[MMRItem]:
        """选择 k 个最大边际相关性的候选项"""
        return mmr_select(candidates, query_vec, self.lambda_, k)

    def select_batch(
        self,
        candidates: Sequence[MMRItem],
        query_vec: np.ndarray,
        k: int = 15,
    ) -> List[MMRItem]:
        """向量化批量版（候选项 > 50 时更高效）

        算法等价于 mmr_select，但用 numpy 矩阵运算加速余弦相似度计算。
        """
        if not candidates or k <= 0:
            return []

        n = len(candidates)
        target_k = min(k, n)

        # 构造候选矩阵 (n, d)
        mat = np.vstack([c.vec for c in candidates])
        # L2 归一化
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1.0, norms)
        mat_normed = mat / norms_safe

        # query 归一化
        q_norm = np.linalg.norm(query_vec)
        q_normed = query_vec / q_norm if q_norm > 0 else query_vec

        # 预计算 query-candidate 相关性 (n,)
        rel_scores = mat_normed @ q_normed

        # 候选项是否仍在 remaining 中
        remaining_mask = np.ones(n, dtype=bool)
        selected_idx: List[int] = []

        for _ in range(target_k):
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) == 0:
                break

            if selected_idx:
                # 已选矩阵 (m, d)
                sel_mat = mat_normed[selected_idx]
                # 候选项与已选项的相似度 (remaining, m)
                sim_to_selected = mat_normed[remaining_indices] @ sel_mat.T
                # 每个候选项的最大相似度作为惩罚
                penalty = sim_to_selected.max(axis=1)
            else:
                penalty = np.zeros(len(remaining_indices))

            rel = rel_scores[remaining_indices]
            mmr_scores = self.lambda_ * rel - (1.0 - self.lambda_) * penalty

            # 选 MMR 分数最高的
            best_local = int(np.argmax(mmr_scores))
            best_global = int(remaining_indices[best_local])
            selected_idx.append(best_global)
            remaining_mask[best_global] = False

        return [candidates[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# 辅助：从 chunk 列表构造 MMRItem
# ---------------------------------------------------------------------------


def build_mmr_items(
    items: Sequence[tuple],
) -> List[MMRItem]:
    """从 (id, vec, payload) 元组列表构造 MMRItem 列表

    Args:
        items: [(id, vec, payload), ...] 或 [(id, vec), ...]
    Returns:
        MMRItem 列表
    """
    result: List[MMRItem] = []
    for item in items:
        if len(item) == 3:
            id_, vec, payload = item
            result.append(MMRItem(id=id_, vec=np.asarray(vec, dtype=float), payload=payload))
        elif len(item) == 2:
            id_, vec = item
            result.append(MMRItem(id=id_, vec=np.asarray(vec, dtype=float)))
        else:
            raise ValueError(f"每个 item 应为 2 元或 3 元组，收到 {len(item)} 元")
    return result

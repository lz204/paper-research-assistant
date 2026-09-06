"""API 成本追踪 + 看板模块（项目文档 6.1 / 6.2 节）

文档 6.1 节"API 成本看板"要求：
- 每次 API 调用记录 model / input_tokens / output_tokens / cost
- 按天聚合写入 PG
- Streamlit 管理页展示每日成本趋势 + 按论文/用户分摊

文档 6.2 节成本表（实测数据）：
- PDF 解析（Qwen-VL-Max，50 页首次）：约 2.0 元
- Embedding（DashScope v3）：约 0.04 元
- LLM 问答 + 综述（Qwen-Plus）：约 0.3 元
- 首次处理单篇约 2.4 元，二次打开因缓存命中降至 0.3 元以下

设计：
- CostRecord dataclass：单次 API 调用成本记录
- BaseCostTracker ABC：支持内存 / PG / 文件热插拔
- InMemoryCostTracker：内存版兜底（本地联调，无 PG 依赖）
- CostDashboard：聚合统计（按天/模型/用户/论文分摊）
- create_cost_tracker：工厂函数
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class CostRecord:
    """单次 API 调用成本记录（文档 6.1 节）

    Attributes:
        ts: 调用时间戳
        date: 日期（用于按天聚合）
        model: 模型名（如 qwen-plus / qwen-vl-max / text-embedding-v3）
        step: 调用步骤（如 rag.retrieve / review.generate / parse.vlm）
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        cost_yuan: 成本（元）
        trace_id: 关联 trace_id（与日志/链路埋点串联）
        user_id: 用户 ID（用于按用户分摊）
        paper_id: 论文 ID（用于按论文分摊）
        from_cache: 是否缓存命中（命中不计费）
    """

    ts: datetime
    model: str
    step: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    trace_id: str = ""
    user_id: str = ""
    paper_id: str = ""
    from_cache: bool = False

    @property
    def date_str(self) -> str:
        """YYYY-MM-DD 格式日期"""
        return self.ts.strftime("%Y-%m-%d")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseCostTracker(ABC):
    """成本追踪抽象基类

    设计目的：支持内存 / PG / 文件 / Redis 等多种存储后端热插拔。
    """

    @abstractmethod
    def record(self, record: CostRecord) -> None:
        """记录单次 API 调用成本"""
        ...

    @abstractmethod
    def get_records(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        model: Optional[str] = None,
        step: Optional[str] = None,
    ) -> List[CostRecord]:
        """查询成本记录"""
        ...

    def summary(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """按天聚合成本（文档 6.1 节"按天聚合"）

        Returns:
            {date_str: {"cost": float, "input_tokens": int, "output_tokens": int, "calls": int}}
        """
        records = self.get_records(date_from=date_from, date_to=date_to)
        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        for r in records:
            d = r.date_str
            agg[d]["cost"] += r.cost_yuan
            agg[d]["input_tokens"] += r.input_tokens
            agg[d]["output_tokens"] += r.output_tokens
            agg[d]["calls"] += 1
        return dict(agg)

    def total_cost(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> float:
        """累计成本（元）"""
        records = self.get_records(date_from=date_from, date_to=date_to)
        return sum(r.cost_yuan for r in records)


# ---------------------------------------------------------------------------
# 内存版成本追踪器（本地联调，无 PG 依赖）
# ---------------------------------------------------------------------------


class InMemoryCostTracker(BaseCostTracker):
    """内存版成本追踪器

    所有记录保存在内存中，适用于：
    - 本地开发联调
    - 单元测试
    - 单机演示

    生产环境应替换为 PgCostTracker（写入 PG，文档 6.1 节"写入 PG"）。
    """

    def __init__(self, capacity: int = 10000):
        self._records: List[CostRecord] = []
        self._capacity = capacity
        self._lock = threading.Lock()

    def record(self, record: CostRecord) -> None:
        with self._lock:
            if len(self._records) >= self._capacity:
                # 简单清理：删一半最旧的
                self._records = self._records[len(self._records) // 2:]
            self._records.append(record)
            logger.debug(
                "成本记录: model=%s, step=%s, cost=%.4f元, tokens=%d/%d",
                record.model, record.step, record.cost_yuan,
                record.input_tokens, record.output_tokens,
            )

    def get_records(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        model: Optional[str] = None,
        step: Optional[str] = None,
    ) -> List[CostRecord]:
        with self._lock:
            records = list(self._records)

        result: List[CostRecord] = []
        for r in records:
            if date_from and r.date_str < date_from:
                continue
            if date_to and r.date_str > date_to:
                continue
            if user_id and r.user_id != user_id:
                continue
            if paper_id and r.paper_id != paper_id:
                continue
            if model and r.model != model:
                continue
            if step and r.step != step:
                continue
            result.append(r)
        return result

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# 成本看板（文档 6.1 节"Streamlit 管理页展示"）
# ---------------------------------------------------------------------------


class CostDashboard:
    """成本看板聚合统计（文档 6.1 节）

    用法：
        tracker = InMemoryCostTracker()
        dashboard = CostDashboard(tracker)
        daily = dashboard.daily_summary()
        by_model = dashboard.cost_by_model()
        by_user = dashboard.cost_by_user()
        by_paper = dashboard.cost_by_paper()
    """

    def __init__(self, tracker: BaseCostTracker):
        self.tracker = tracker

    def daily_summary(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """按天汇总成本（文档 6.1 节"每日成本趋势"）"""
        return self.tracker.summary(date_from=date_from, date_to=date_to)

    def cost_by_model(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """按模型汇总成本（文档 6.1 节"按论文/用户分摊"基础）"""
        records = self.tracker.get_records(date_from=date_from, date_to=date_to)
        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        for r in records:
            agg[r.model]["cost"] += r.cost_yuan
            agg[r.model]["calls"] += 1
            agg[r.model]["input_tokens"] += r.input_tokens
            agg[r.model]["output_tokens"] += r.output_tokens
        return dict(agg)

    def cost_by_user(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """按用户分摊成本"""
        records = self.tracker.get_records(date_from=date_from, date_to=date_to)
        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0}
        )
        for r in records:
            uid = r.user_id or "(anonymous)"
            agg[uid]["cost"] += r.cost_yuan
            agg[uid]["calls"] += 1
        return dict(agg)

    def cost_by_paper(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """按论文分摊成本"""
        records = self.tracker.get_records(date_from=date_from, date_to=date_to)
        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0}
        )
        for r in records:
            pid = r.paper_id or "(no_paper)"
            agg[pid]["cost"] += r.cost_yuan
            agg[pid]["calls"] += 1
        return dict(agg)

    def cache_hit_rate(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, float]:
        """缓存命中率统计（文档 6.2 节）"""
        records = self.tracker.get_records(date_from=date_from, date_to=date_to)
        if not records:
            return {"rate": 0.0, "hits": 0, "total": 0}
        hits = sum(1 for r in records if r.from_cache)
        return {
            "rate": hits / len(records),
            "hits": hits,
            "total": len(records),
        }

    def render_text(self) -> str:
        """渲染为文本看板（Streamlit 管理页 / CLI 展示用）"""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("成本看板（文档 6.1 / 6.2 节）")
        lines.append("=" * 60)

        # 按天
        lines.append("\n[按天聚合]")
        daily = self.daily_summary()
        if not daily:
            lines.append("  (无数据)")
        for d, stat in sorted(daily.items()):
            lines.append(
                f"  {d}: {stat['cost']:.4f}元 | "
                f"调用 {int(stat['calls'])} 次 | "
                f"tokens {int(stat['input_tokens'])}/{int(stat['output_tokens'])}"
            )

        # 按模型
        lines.append("\n[按模型聚合]")
        by_model = self.cost_by_model()
        if not by_model:
            lines.append("  (无数据)")
        for m, stat in sorted(by_model.items()):
            lines.append(
                f"  {m}: {stat['cost']:.4f}元 | 调用 {int(stat['calls'])} 次"
            )

        # 缓存命中率
        lines.append("\n[缓存命中]")
        rate = self.cache_hit_rate()
        lines.append(
            f"  命中率: {rate['rate']:.1%} ({int(rate['hits'])}/{int(rate['total'])})"
        )

        # 累计
        lines.append(f"\n[累计成本] {self.tracker.total_cost():.4f} 元")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 全局单例 + 便捷接口
# ---------------------------------------------------------------------------


_global_tracker: Optional[BaseCostTracker] = None
_global_lock = threading.Lock()


def get_cost_tracker() -> BaseCostTracker:
    """获取全局成本追踪器（单例，默认 InMemoryCostTracker）"""
    global _global_tracker
    with _global_lock:
        if _global_tracker is None:
            _global_tracker = InMemoryCostTracker()
        return _global_tracker


def set_cost_tracker(tracker: BaseCostTracker) -> None:
    """替换全局成本追踪器（生产环境可换成 PgCostTracker）"""
    global _global_tracker
    with _global_lock:
        _global_tracker = tracker


def record_cost(
    model: str,
    step: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_yuan: float = 0.0,
    trace_id: str = "",
    user_id: str = "",
    paper_id: str = "",
    from_cache: bool = False,
) -> None:
    """便捷函数：记录单次 API 调用成本（文档 6.1 节）

    Args:
        model: 模型名（如 qwen-plus）
        step: 调用步骤（如 review.generate）
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        cost_yuan: 成本（元）
        trace_id: 关联 trace_id
        user_id: 用户 ID
        paper_id: 论文 ID
        from_cache: 是否缓存命中
    """
    tracker = get_cost_tracker()
    tracker.record(
        CostRecord(
            ts=datetime.now(),
            model=model,
            step=step,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_yuan=cost_yuan,
            trace_id=trace_id,
            user_id=user_id,
            paper_id=paper_id,
            from_cache=from_cache,
        )
    )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_cost_tracker(mode: str = "auto", **kwargs) -> BaseCostTracker:
    """工厂函数：按模式创建成本追踪器

    Args:
        mode: "memory" / "pg" / "auto"
            - auto：优先 PG（未实现），降级内存
        **kwargs: 传给具体 tracker 的参数
    Returns:
        BaseCostTracker 实例
    """
    mode = mode.lower()

    if mode == "memory":
        return InMemoryCostTracker(**kwargs)

    if mode == "pg":
        # 文档 6.1 节"按天聚合写入 PG"——生产环境实现
        # 当前版本未实现，降级为内存版（与项目降级风格一致）
        logger.warning("PgCostTracker 暂未实现，降级为 InMemoryCostTracker")
        return InMemoryCostTracker(**kwargs)

    if mode == "auto":
        # 自动选 PG 还是内存：环境变量切换
        if os.getenv("DATABASE_URL"):
            logger.info("检测到 DATABASE_URL，但 PgCostTracker 暂未实现，使用内存版")
        return InMemoryCostTracker(**kwargs)

    raise ValueError(f"未知 cost tracker mode: {mode}")

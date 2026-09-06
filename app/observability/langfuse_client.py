"""Langfuse LLM 调用链埋点模块（项目文档 6.1 节）

文档 6.1 节"Langfuse 追 LLM 调用链"要求：
- 每次 LLM/VLM 调用自动上报 trace
- trace 包含：prompt / response / token 数 / 延迟 / 成本
- 一个用户问题从"检索 → 记忆池注入 → LLM 生成 → 引用验证"全链路可回放

容错：langfuse 未安装时 mock 模式控制台打印（文档 6.3 节降级思想），
保证无第三方依赖也能跑通全链路。

设计：
- BaseLangfuseClient ABC：支持 Langfuse / Mock / Noop 热插拔
- LangfuseClient：真实 Langfuse SDK 实现
- MockLangfuseClient：langfuse 不可用时降级，控制台打印 trace
- LangfuseTrace：trace 上下文管理器（支持嵌套 span）
- create_langfuse_client：工厂函数
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SpanRecord:
    """单个 span 记录（一次子操作）

    Attributes:
        span_id: span 唯一 ID
        name: span 名称（如 rag.retrieve / llm.generate）
        start_ts: 开始时间戳
        end_ts: 结束时间戳
        latency_ms: 耗时毫秒
        input: 输入数据（prompt / query）
        output: 输出数据（response）
        metadata: 元数据（model / tokens / cost / status 等）
        status: ok / error
        error: 错误信息
    """

    span_id: str
    name: str
    start_ts: datetime = field(default_factory=datetime.now)
    end_ts: Optional[datetime] = None
    latency_ms: int = 0
    input: Any = None
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "latency_ms": self.latency_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class TraceRecord:
    """完整 trace 记录（一个用户问题 / 一次任务的全链路）

    Attributes:
        trace_id: trace 唯一 ID
        name: trace 名称
        user_id: 用户 ID
        session_id: 会话 ID
        paper_id: 论文 ID
        start_ts: 开始时间
        spans: span 列表
        metadata: 全局元数据
    """

    trace_id: str
    name: str = ""
    user_id: str = ""
    session_id: str = ""
    paper_id: str = ""
    start_ts: datetime = field(default_factory=datetime.now)
    spans: List[SpanRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> int:
        """trace 总耗时（最后一个 span 的结束 - trace 开始）"""
        if not self.spans:
            return 0
        last = self.spans[-1]
        end = last.end_ts or datetime.now()
        return int((end - self.start_ts).total_seconds() * 1000)

    @property
    def total_cost_yuan(self) -> float:
        """trace 累计成本"""
        return sum(s.metadata.get("cost_yuan", 0.0) for s in self.spans)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "start_ts": self.start_ts.isoformat(),
            "total_latency_ms": self.total_latency_ms,
            "total_cost_yuan": self.total_cost_yuan,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseLangfuseClient(ABC):
    """LLM 调用链埋点抽象基类

    设计目的：Langfuse / Mock / 其他追踪系统热插拔。
    """

    @abstractmethod
    def start_trace(
        self,
        name: str,
        user_id: str = "",
        session_id: str = "",
        paper_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceRecord:
        """开始一个 trace"""
        ...

    @abstractmethod
    def end_trace(self, trace: TraceRecord) -> None:
        """结束 trace（上报）"""
        ...

    @abstractmethod
    @contextmanager
    def span(
        self,
        trace: TraceRecord,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanRecord]:
        """记录一个 span（上下文管理器）

        Args:
            trace: 所属 trace
            name: span 名称
            input: 输入数据
            metadata: 元数据
        Yields:
            SpanRecord（可在 with 块内更新 output / metadata）
        """
        ...


# ---------------------------------------------------------------------------
# Langfuse 真实实现（生产环境用）
# ---------------------------------------------------------------------------


class LangfuseClient(BaseLangfuseClient):
    """Langfuse SDK 真实实现

    工作流程：
    1. start_trace 创建 Langfuse trace 对象
    2. span 上下文管理器创建 span，结束时上报
    3. end_trace 上报整个 trace

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse 服务地址
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._client = None
        self._mock_mode = not bool(self.public_key and self.secret_key)

        if self._mock_mode:
            logger.warning(
                "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 未配置，"
                "LangfuseClient 进入 mock 模式（控制台打印 trace）"
            )

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if self._mock_mode:
            return False
        try:
            from langfuse import Langfuse  # type: ignore
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            return True
        except ImportError:
            logger.warning("langfuse 包未安装，LangfuseClient 不可用")
            return False

    def start_trace(
        self,
        name: str,
        user_id: str = "",
        session_id: str = "",
        paper_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceRecord:
        trace_id = uuid.uuid4().hex[:16]
        trace = TraceRecord(
            trace_id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
            metadata=metadata or {},
        )
        # Langfuse SDK 的 trace 对象暂存在 metadata 中（私有）
        if self._ensure_client():
            try:
                lf_trace = self._client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id or None,
                    session_id=session_id or None,
                    metadata=metadata or {},
                )
                trace.metadata["_lf_trace"] = lf_trace
            except Exception as e:  # noqa: BLE001
                logger.warning("Langfuse trace 创建失败: %s", e)
        return trace

    def end_trace(self, trace: TraceRecord) -> None:
        # Langfuse SDK 自动异步上报，这里只记录日志
        logger.info(
            "Langfuse trace 结束: id=%s, spans=%d, latency=%dms, cost=%.4f元",
            trace.trace_id, len(trace.spans),
            trace.total_latency_ms, trace.total_cost_yuan,
        )

    @contextmanager
    def span(
        self,
        trace: TraceRecord,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanRecord]:
        span = SpanRecord(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            input=input,
            metadata=metadata or {},
        )
        t0 = time.time()
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            raise
        finally:
            span.end_ts = datetime.now()
            span.latency_ms = int((time.time() - t0) * 1000)
            trace.spans.append(span)

            # 上报 Langfuse
            lf_trace = trace.metadata.get("_lf_trace")
            if lf_trace is not None:
                try:
                    lf_trace.span(
                        name=span.name,
                        input=span.input,
                        output=span.output,
                        metadata=span.metadata,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Langfuse span 上报失败: %s", e)


# ---------------------------------------------------------------------------
# Mock Langfuse 客户端（langfuse 未安装时降级）
# ---------------------------------------------------------------------------


class MockLangfuseClient(BaseLangfuseClient):
    """Mock Langfuse 客户端（文档 6.3 节降级）

    langfuse 包未安装或 API key 未配置时使用：
    - 同样维护 trace / span 数据结构（保证接口一致）
    - end_trace 时在控制台打印完整 trace（JSON 格式）
    - 所有 trace 历史保存在内存中，便于测试断言
    """

    def __init__(self):
        self._traces: List[TraceRecord] = []
        self._lock = threading.Lock()

    def start_trace(
        self,
        name: str,
        user_id: str = "",
        session_id: str = "",
        paper_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceRecord:
        trace_id = uuid.uuid4().hex[:16]
        return TraceRecord(
            trace_id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
            metadata=metadata or {},
        )

    def end_trace(self, trace: TraceRecord) -> None:
        with self._lock:
            self._traces.append(trace)
        # 控制台打印（文档 6.1 节"mock 模式控制台打印"）
        print(
            "[MockLangfuse] trace 上报:\n"
            + json.dumps(trace.to_dict(), ensure_ascii=False, indent=2, default=str)
        )

    @contextmanager
    def span(
        self,
        trace: TraceRecord,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanRecord]:
        span = SpanRecord(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            input=input,
            metadata=metadata or {},
        )
        t0 = time.time()
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            raise
        finally:
            span.end_ts = datetime.now()
            span.latency_ms = int((time.time() - t0) * 1000)
            trace.spans.append(span)

    @property
    def traces(self) -> List[TraceRecord]:
        """历史 trace 列表（测试断言用）"""
        return list(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


# ---------------------------------------------------------------------------
# 全局单例 + 便捷接口
# ---------------------------------------------------------------------------


_global_client: Optional[BaseLangfuseClient] = None
_global_lock = threading.Lock()


def get_langfuse_client() -> BaseLangfuseClient:
    """获取全局 langfuse 客户端（单例）"""
    global _global_client
    with _global_lock:
        if _global_client is None:
            _global_client = create_langfuse_client("auto")
        return _global_client


def set_langfuse_client(client: BaseLangfuseClient) -> None:
    """替换全局 langfuse 客户端"""
    global _global_client
    with _global_lock:
        _global_client = client


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_langfuse_client(mode: str = "auto", **kwargs) -> BaseLangfuseClient:
    """工厂函数：按模式创建 langfuse 客户端

    Args:
        mode: "langfuse" / "mock" / "auto"
            - auto：优先 Langfuse，无 API key 降级 Mock
        **kwargs: 传给具体 client 的参数
    Returns:
        BaseLangfuseClient 实例
    """
    mode = mode.lower()

    if mode == "mock":
        return MockLangfuseClient()

    if mode == "langfuse":
        return LangfuseClient(**kwargs)

    if mode == "auto":
        client = LangfuseClient(**kwargs)
        if client._ensure_client():
            return client
        logger.info("auto 模式：Langfuse 不可用，降级为 MockLangfuseClient")
        return MockLangfuseClient()

    raise ValueError(f"未知 langfuse client mode: {mode}")

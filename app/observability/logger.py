"""结构化日志模块（项目文档 6.1 节）

文档 6.1 节"结构化日志"要求：
- Python structlog 输出 JSON 日志
- 关键字段：trace_id / user_id / paper_id / step / latency_ms / cost_yuan / status
- 写文件 + 控制台，方便 grep 和后续接 ELK

容错：structlog 未安装时降级用标准 logging + JSON 格式化，
保证无第三方依赖也能跑通全链路（文档 6.3 节降级思想）。

设计：
- StructuredLogger：统一结构化日志接口（bind / info / warning / error）
- StructlogLogger：structlog 实现（生产环境用）
- StdJsonLogger：标准 logging + JSON 序列化（structlog 不可用时降级）
- create_logger：工厂函数
- get_logger：模块级便捷获取（自动绑定 trace_id）

trace_id 策略：
- 一个请求/任务一个 trace_id
- 通过 bind(trace_id=...) 注入
- 后续日志自动带上该字段
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文档 6.1 节关键字段
# ---------------------------------------------------------------------------


# 结构化日志的标准字段（文档 6.1 节）
LOG_FIELDS = (
    "trace_id",   # 请求/任务追踪 ID
    "user_id",    # 用户 ID
    "paper_id",   # 论文 ID
    "step",       # 流程步骤（如 "rag.retrieve" / "review.generate"）
    "latency_ms", # 步骤耗时毫秒
    "cost_yuan",  # 本次调用成本（元）
    "status",     # ok / warn / error
    "msg",        # 日志消息
)


# ---------------------------------------------------------------------------
# trace_id 上下文（线程级隔离）
# ---------------------------------------------------------------------------


_trace_local = threading.local()


def new_trace_id() -> str:
    """生成新的 trace_id（UUID4 截短）"""
    return uuid.uuid4().hex[:16]


def get_current_trace_id() -> str:
    """获取当前线程的 trace_id（未设置时返回空串）"""
    return getattr(_trace_local, "trace_id", "")


def set_current_trace_id(trace_id: str) -> None:
    """设置当前线程的 trace_id"""
    _trace_local.trace_id = trace_id


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class StructuredLogger(ABC):
    """结构化日志抽象基类

    设计目的：structlog / 标准日志 / 其他实现热插拔。
    """

    @abstractmethod
    def bind(self, **fields: Any) -> "StructuredLogger":
        """绑定上下文字段（返回新实例，原实例不变）

        常用字段：trace_id / user_id / paper_id / step
        """
        ...

    @abstractmethod
    def info(self, msg: str, **fields: Any) -> None:
        ...

    @abstractmethod
    def warning(self, msg: str, **fields: Any) -> None:
        ...

    @abstractmethod
    def error(self, msg: str, **fields: Any) -> None:
        ...

    @abstractmethod
    def debug(self, msg: str, **fields: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# structlog 实现（生产环境首选）
# ---------------------------------------------------------------------------


class StructlogLogger(StructuredLogger):
    """structlog 实现的结构化日志（文档 6.1 节推荐）

    工作流程：
    1. 配置 structlog（JSONRenderer + 写文件 + 控制台）
    2. bind(...) 链式绑定上下文字段
    3. info/warning/error/debug 输出 JSON 日志

    容错：structlog 未安装时由 StdJsonLogger 降级。
    """

    _configured = False

    def __init__(self, name: str = "app", bound: Optional[Dict[str, Any]] = None):
        try:
            import structlog  # type: ignore
            self._structlog = structlog
            self._ensure_configured()
        except ImportError:
            # 不应到这里，由工厂函数保证降级
            raise ImportError("structlog 未安装，请用 StdJsonLogger 降级")

        self._name = name
        self._log = structlog.get_logger(name)
        self._bound: Dict[str, Any] = dict(bound or {})

        # 默认带上当前 trace_id
        if "trace_id" not in self._bound:
            tid = get_current_trace_id()
            if tid:
                self._bound["trace_id"] = tid

        if self._bound:
            self._log = self._log.bind(**self._bound)

    @classmethod
    def _ensure_configured(cls) -> None:
        """全局配置 structlog（只配置一次）"""
        if cls._configured:
            return
        structlog = cls._structlog if hasattr(cls, "_structlog") else None
        if structlog is None:
            import structlog as _sl  # type: ignore
            structlog = _sl
            cls._structlog = structlog

        # 配置：JSON 输出 + 标准库接管（便于 handler 配置）
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        cls._configured = True

    def bind(self, **fields: Any) -> "StructlogLogger":
        new_bound = {**self._bound, **fields}
        return StructlogLogger(name=self._name, bound=new_bound)

    def info(self, msg: str, **fields: Any) -> None:
        self._log.info(msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log.warning(msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._log.error(msg, **fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log.debug(msg, **fields)


# ---------------------------------------------------------------------------
# 标准日志 + JSON 降级实现（structlog 不可用时）
# ---------------------------------------------------------------------------


class StdJsonLogger(StructuredLogger):
    """标准 logging + JSON 序列化降级实现

    structlog 未安装时使用，输出与 structlog JSONRenderer 兼容的格式，
    方便后续 grep / ELK 接入不受影响。

    工作流程：
    1. 创建一个标准 logging.Logger
    2. 添加 FileHandler + StreamHandler
    3. bind(...) 保存上下文字段，输出时合并到 JSON
    """

    _file_handlers: Dict[str, logging.FileHandler] = {}

    def __init__(
        self,
        name: str = "app",
        bound: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
    ):
        self._name = name
        self._bound: Dict[str, Any] = dict(bound or {})
        self._log = logging.getLogger(name)
        self._log.setLevel(logging.INFO)

        # 默认 trace_id
        if "trace_id" not in self._bound:
            tid = get_current_trace_id()
            if tid:
                self._bound["trace_id"] = tid

        # 配置 handler（每个 name 只配一次）
        if not self._log.handlers:
            formatter = logging.Formatter("%(message)s")

            # 控制台
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            self._log.addHandler(sh)

            # 文件
            file_path = log_file or os.getenv(
                "APP_LOG_FILE", "logs/app.log"
            )
            try:
                # 确保目录存在
                log_dir = os.path.dirname(file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(file_path, encoding="utf-8")
                fh.setFormatter(formatter)
                self._log.addHandler(fh)
                self._file_handlers[name] = fh
            except (OSError, PermissionError) as e:
                # 文件不可写时只输出到控制台
                logger.warning("日志文件不可写，仅输出到控制台: %s", e)

    def bind(self, **fields: Any) -> "StdJsonLogger":
        new_bound = {**self._bound, **fields}
        return StdJsonLogger(name=self._name, bound=new_bound)

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        """组装 JSON 并输出"""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "logger": self._name,
            "msg": msg,
        }
        # 合并 bound + 临时字段（临时覆盖 bound）
        record.update(self._bound)
        record.update(fields)

        # 补全 trace_id（可能 bind 后才设置）
        if not record.get("trace_id"):
            tid = get_current_trace_id()
            if tid:
                record["trace_id"] = tid

        line = json.dumps(record, ensure_ascii=False, default=str)

        if level == "debug":
            self._log.debug(line)
        elif level == "info":
            self._log.info(line)
        elif level == "warning":
            self._log.warning(line)
        elif level == "error":
            self._log.error(line)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("info", msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("warning", msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("error", msg, fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("debug", msg, fields)


# ---------------------------------------------------------------------------
# 工厂函数 + 便捷接口
# ---------------------------------------------------------------------------


# 全局默认 logger 实例（延迟初始化）
_default_logger: Optional[StructuredLogger] = None
_default_lock = threading.Lock()


def create_logger(
    name: str = "app",
    prefer_structlog: bool = True,
    **fields: Any,
) -> StructuredLogger:
    """工厂函数：创建结构化 logger

    Args:
        name: logger 名称
        prefer_structlog: 是否优先用 structlog（默认 True）
        **fields: 初始绑定的上下文字段
    Returns:
        StructuredLogger 实例（structlog 不可用时降级为 StdJsonLogger）
    """
    if prefer_structlog:
        try:
            return StructlogLogger(name=name, bound=fields or None)
        except ImportError:
            logger.info("structlog 未安装，降级为 StdJsonLogger")

    return StdJsonLogger(name=name, bound=fields or None)


def get_logger(name: str = "app") -> StructuredLogger:
    """获取全局默认 logger（单例）

    首次调用时创建，后续返回同一实例。
    """
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            _default_logger = create_logger(name=name)
        return _default_logger


# ---------------------------------------------------------------------------
# 便捷上下文管理器
# ---------------------------------------------------------------------------


@dataclass
class LogContext:
    """日志上下文（文档 6.1 节 trace_id 传递）

    用法：
        with LogContext(trace_id="abc123", user_id="u1") as ctx:
            ctx.logger.info("开始处理")
            # 后续日志都带上 trace_id 和 user_id
    """

    trace_id: str = field(default_factory=new_trace_id)
    user_id: str = ""
    paper_id: str = ""
    step: str = ""
    _prev_trace_id: str = ""
    _logger: Optional[StructuredLogger] = None

    def __enter__(self) -> "LogContext":
        self._prev_trace_id = get_current_trace_id()
        set_current_trace_id(self.trace_id)

        fields: Dict[str, Any] = {"trace_id": self.trace_id}
        if self.user_id:
            fields["user_id"] = self.user_id
        if self.paper_id:
            fields["paper_id"] = self.paper_id
        if self.step:
            fields["step"] = self.step

        self._logger = get_logger().bind(**fields)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        set_current_trace_id(self._prev_trace_id)

    @property
    def logger(self) -> StructuredLogger:
        if self._logger is None:
            self._logger = get_logger()
        return self._logger

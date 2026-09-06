"""可观测性模块（项目文档 6.1 节）

包含三层观测（文档 6.1 节）：
1. Langfuse 追 LLM 调用链
2. structlog 结构化日志（trace_id / user_id / paper_id / step / latency_ms / cost_yuan / status）
3. API 成本看板（model / input_tokens / output_tokens / cost，按天聚合）

模块总览（项目文档附录）：
- langfuse_client.py  LLM 调用链埋点
- logger.py           structlog 结构化日志
- cost_tracker.py     API 成本追踪 + 看板
"""

from .logger import (
    LogContext,
    StdJsonLogger,
    StructlogLogger,
    StructuredLogger,
    create_logger,
    get_current_trace_id,
    get_logger,
    new_trace_id,
    set_current_trace_id,
)
from .cost_tracker import (
    BaseCostTracker,
    CostDashboard,
    CostRecord,
    InMemoryCostTracker,
    create_cost_tracker,
    get_cost_tracker,
    record_cost,
    set_cost_tracker,
)
from .langfuse_client import (
    BaseLangfuseClient,
    LangfuseClient,
    MockLangfuseClient,
    SpanRecord,
    TraceRecord,
    create_langfuse_client,
    get_langfuse_client,
    set_langfuse_client,
)

__all__ = [
    # logger
    "LogContext",
    "StdJsonLogger",
    "StructlogLogger",
    "StructuredLogger",
    "create_logger",
    "get_current_trace_id",
    "get_logger",
    "new_trace_id",
    "set_current_trace_id",
    # cost_tracker
    "BaseCostTracker",
    "CostDashboard",
    "CostRecord",
    "InMemoryCostTracker",
    "create_cost_tracker",
    "get_cost_tracker",
    "record_cost",
    "set_cost_tracker",
    # langfuse_client
    "BaseLangfuseClient",
    "LangfuseClient",
    "MockLangfuseClient",
    "SpanRecord",
    "TraceRecord",
    "create_langfuse_client",
    "get_langfuse_client",
    "set_langfuse_client",
]

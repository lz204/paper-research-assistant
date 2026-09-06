"""LLM 调用封装模块（项目文档 6.2 / 6.3 节）

包含：DashScope Qwen-Plus 调用 + 重试 + 缓存 + 部分结果降级。

模块总览（项目文档附录）：
- client.py    LLM 调用封装（重试 + 缓存 + 降级）
"""

from .client import (
    BaseLLMClient,
    DashScopeLLMClient,
    LLMResponse,
    MockLLMClient,
    QACache,
    create_llm_client,
    estimate_cost_yuan,
)

__all__ = [
    "BaseLLMClient",
    "DashScopeLLMClient",
    "LLMResponse",
    "MockLLMClient",
    "QACache",
    "create_llm_client",
    "estimate_cost_yuan",
]

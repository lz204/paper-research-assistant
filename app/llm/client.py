"""LLM 调用封装：DashScope（Qwen-Plus）+ 重试 + 缓存 + 部分结果降级。

模块组成：
- LLMResponse：统一返回结构（text / usage / cost / from_cache / partial）
- BaseLLMClient ABC：支持 DashScope / Mock 热插拔
- DashScopeLLMClient：真实 Qwen-Plus 调用（含重试、缓存、成本估算）
- MockLLMClient：无 Key 时模板化回答，保证本地链路可联调
- QACache：问答级缓存（命中不计费）
- create_llm_client：工厂函数

失败不抛异常；partial=True 由调用方处理。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """LLM 调用统一返回结构。"""

    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_yuan: float = 0.0
    from_cache: bool = False
    partial: bool = False
    latency_ms: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        """调用是否完全成功（非缓存非部分）"""
        return bool(self.text) and not self.partial and not self.error


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """LLM 调用抽象基类

    设计目的：支持 DashScope / OpenAI / Mock 等多种后端热插拔，
    本地无 API key 时也能跑通全链路。
    """

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """LLM 聊天接口

        Args:
            prompt: 用户输入
            system: 系统提示（可选）
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        Returns:
            LLMResponse（最终失败时返回 partial=True 的部分结果，不抛异常）
        """
        ...

    def chat_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> List[LLMResponse]:
        """批量聊天（默认串行，子类可重写为并发）"""
        return [
            self.chat(p, system=system, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]


# ---------------------------------------------------------------------------
# 问答级缓存
# ---------------------------------------------------------------------------


class QACache:
    """问答级缓存：(system, prompt, temperature, max_tokens) → LLMResponse。

    命中时返回副本并置 cost_yuan=0/from_cache=True，避免重复计费；
    超容量时删一半最旧的（简单 FIFO 清理）。
    """

    def __init__(self, capacity: int = 1000):
        self._store: Dict[str, LLMResponse] = {}
        self._capacity = capacity

    @staticmethod
    def _make_key(
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raw = f"{system or ''}||{prompt}||{temperature}||{max_tokens}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[LLMResponse]:
        key = self._make_key(system, prompt, temperature, max_tokens)
        cached = self._store.get(key)
        if cached is not None:
            # 标记 from_cache（返回副本，避免外部修改）
            return LLMResponse(
                text=cached.text,
                model=cached.model,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                cost_yuan=0.0,  # 命中缓存不重复计费
                from_cache=True,
                partial=False,
                latency_ms=0,
                error="",
            )
        return None

    def put(
        self,
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        response: LLMResponse,
    ) -> None:
        if len(self._store) >= self._capacity:
            # 简单清理：删一半最旧的
            keys = list(self._store.keys())
            for k in keys[: len(keys) // 2]:
                del self._store[k]
        key = self._make_key(system, prompt, temperature, max_tokens)
        self._store[key] = response

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# 成本估算（元/千 token，DashScope 公开价目）
# ---------------------------------------------------------------------------

# 实际计费以账单为准；这里使用 DashScope 官网公开价格做近似估算
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "qwen-plus": {"input": 0.004, "output": 0.012},
    "qwen-turbo": {"input": 0.002, "output": 0.006},
    "qwen-max": {"input": 0.02, "output": 0.06},
}


def estimate_cost_yuan(model: str, input_tokens: int, output_tokens: int) -> float:
    """按模型公开价目表估算人民币成本（未知模型返回 0）。"""
    price = PRICE_TABLE.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens / 1000.0 * price["input"]
        + output_tokens / 1000.0 * price["output"]
    )


# ---------------------------------------------------------------------------
# DashScope LLM 客户端（缓存 → 调用 → 重试 → partial 降级）
# ---------------------------------------------------------------------------


class DashScopeLLMClient(BaseLLMClient):
    """DashScope Qwen-Plus 客户端：缓存命中优先，失败重试，最终 partial 降级。"""

    MAX_RETRY = 2
    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        cache: Optional[QACache] = None,
        max_retry: int = MAX_RETRY,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.cache = cache or QACache()
        self.max_retry = max_retry
        self.timeout = timeout
        self._client = None
        self._mock_mode = not bool(self.api_key)
        if self._mock_mode:
            logger.warning(
                "DASHSCOPE_API_KEY 未配置，DashScopeLLMClient 进入 mock 模式"
            )

    def _ensure_client(self) -> bool:
        """懒加载 DashScope client"""
        if self._client is not None:
            return True
        if not self.api_key:
            return False
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning("dashscope 包未安装，DashScopeLLMClient 不可用")
            return False

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # 1. 缓存命中检查
        cached = self.cache.get(system, prompt, temperature, max_tokens)
        if cached is not None:
            logger.debug("QA 缓存命中")
            return cached

        # 2. 无 Key 或 dashscope 未安装 → 降级 Mock
        if self._mock_mode or not self._ensure_client():
            return MockLLMClient().chat(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens
            )

        # 3. 调 API + 指数退避重试
        t0 = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                resp = self._call_api(prompt, system, temperature, max_tokens)
                latency_ms = int((time.time() - t0) * 1000)
                input_tokens, output_tokens = self._extract_usage(resp)
                cost = estimate_cost_yuan(self.model, input_tokens, output_tokens)
                response = LLMResponse(
                    text=self._extract_text(resp),
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_yuan=cost,
                    from_cache=False,
                    partial=False,
                    latency_ms=latency_ms,
                )
                # 4. 写缓存（相同 system+prompt+temperature+max_tokens 下次直接命中）
                self.cache.put(system, prompt, temperature, max_tokens, response)
                return response

            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retry:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM 调用失败（attempt %d/%d），%ds 后重试: %s",
                        attempt + 1, self.max_retry + 1, wait, e,
                    )
                    time.sleep(wait)

        # 5. 重试全部失败：返回 partial=True（含错误信息），由调用方做降级
        latency_ms = int((time.time() - t0) * 1000)
        logger.error("LLM 调用最终失败: %s", last_error)
        return LLMResponse(
            text="",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            cost_yuan=0.0,
            from_cache=False,
            partial=True,
            latency_ms=latency_ms,
            error=str(last_error) if last_error else "unknown error",
        )

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """从 DashScope 响应中抽取文本。

        DashScope 支持两种 result_format：
        - "message"：文本位于 output.choices[0].message.content（官方推荐）
        - "text"   ：文本位于 output.text
        兜底按 output.text → choices 路径返回，保证 None 时返回空串。
        """
        output = getattr(resp, "output", None)
        if output is None:
            return ""
        text = getattr(output, "text", None)
        if isinstance(text, str) and text:
            return text
        choices = getattr(output, "choices", None)
        if isinstance(choices, list) and choices:
            first = choices[0]
            msg = getattr(first, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    return content
        return text or ""

    def _call_api(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """实际调 DashScope API（result_format=message）"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.Generation.call(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format="message",
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope LLM API 错误: code={resp.code}, msg={resp.message}"
            )
        return resp

    @staticmethod
    def _extract_usage(resp: Any) -> tuple:
        """从 DashScope 响应中提取 token 用量

        DashScope usage 字段: {input_tokens, output_tokens, total_tokens}
        """
        usage = getattr(resp, "usage", None) or {}
        if isinstance(usage, dict):
            return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        # 兼容对象形式
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Mock LLM 客户端（无 API key 时降级）
# ---------------------------------------------------------------------------


class MockLLMClient(BaseLLMClient):
    """无 Key 或 dashscope 缺包时的模板化客户端。

    不调 API，返回基于 prompt 的模板化文本，保证下游链路可联调。
    """

    def __init__(self, model: str = "mock-qwen-plus"):
        self.model = model

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # 模板化回答
        prompt_preview = prompt.strip().replace("\n", " ")[:60]
        text = (
            f"[Mock LLM 回复] 关于「{prompt_preview}...」的模板化回答。"
            f"实际部署需配置 DASHSCOPE_API_KEY 后由 Qwen-Plus 生成。"
        )

        # 估算 token：粗略按字符数/4
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(text) // 4)

        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_yuan=0.0,  # Mock 不计费
            from_cache=False,
            partial=False,
            latency_ms=1,
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_llm_client(mode: str = "auto", **kwargs) -> BaseLLMClient:
    """按模式创建 LLM 客户端：auto 优先 DashScope，无 Key/缺包时降级 Mock。"""
    mode = mode.lower()
    if mode == "mock":
        return MockLLMClient(**kwargs)
    if mode == "dashscope":
        return DashScopeLLMClient(**kwargs)
    if mode == "auto":
        client = DashScopeLLMClient(**kwargs)
        if client._ensure_client():
            return client
        logger.info("auto 模式：DashScope LLM 不可用，降级为 MockLLMClient")
        return MockLLMClient()
    raise ValueError(f"未知 llm client mode: {mode}")

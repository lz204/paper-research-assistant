"""Qwen-VL-Max API 客户端（项目文档 4.1 节 + 6.2/6.3 章）

文档 4.1 节关键技术细节：
1. VLM 太慢：Qwen-VL-Max 单页 3-5 秒，50 页串行 4 分钟
   → asyncio + Semaphore(5) 并发，50 页压到 50 秒
   → 超并发 429 时指数退避重试
2. 结果缓存：解析结果按 PDF MD5 哈希缓存到 Redis
   → key: parse:{md5}:{page_no}，TTL 30 天
   → 二次打开免费，缓存命中率约 60%（文档 6.2 成本治理）

输出结构化结果：{text, formulas: [{latex, location_desc, confidence}], tables: [...]}

工程现实（文档 4.1 节"公式怎么绑定到段落"）：
- Qwen-VL-Max 不返回精确坐标框，只返回自然语言位置描述
- 如"第 3 段后""公式(1)位于第二栏中部"
- 因此 formula_binder 不能做 IoU 对齐，只能做规则匹配
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class VlmFormula:
    """VLM 识别出的公式

    Attributes:
        latex: LaTeX 文本
        location_desc: VLM 返回的自然语言位置描述（如"第 3 段后""第二栏中部"）
        confidence: VLM 自评置信度 [0, 1]
        formula_id: 公式编号（如"(1)" "(2)"，VLM 输出或后端生成）
    """

    latex: str
    location_desc: str
    confidence: float
    formula_id: str = ""


@dataclass
class VlmTable:
    """VLM 识别出的表格

    Attributes:
        markdown: Markdown 格式表格
        location_desc: 位置描述
        confidence: 置信度
    """

    markdown: str
    location_desc: str
    confidence: float


@dataclass
class VlmPageResult:
    """VLM 单页识别结果

    Attributes:
        page_no: 页码（0-based）
        text: VLM 识别的正文
        formulas: 公式列表
        tables: 表格列表
        confidence: 整页置信度
        from_cache: 是否来自缓存
        formula_resolved: 公式是否成功解析（降级时为 False）
    """

    page_no: int
    text: str
    formulas: List[VlmFormula] = field(default_factory=list)
    tables: List[VlmTable] = field(default_factory=list)
    confidence: float = 0.0
    from_cache: bool = False
    formula_resolved: bool = True


# ---------------------------------------------------------------------------
# VLM 客户端
# ---------------------------------------------------------------------------


# 文档 4.1 节：VLM prompt 要求返回结构化 JSON
DEFAULT_VLM_PROMPT = """\
请识别这页论文图片，返回 JSON 格式：
{
  "text": "正文文本（不含公式和表格）",
  "formulas": [
    {
      "latex": "公式 LaTeX",
      "location_desc": "公式在页面中的位置描述（如'第3段后'、'第二栏中部'、'页面底部'）",
      "formula_id": "公式编号（如(1)、(2)，无则留空）",
      "confidence": 0.0-1.0
    }
  ],
  "tables": [
    {
      "markdown": "Markdown 格式表格",
      "location_desc": "位置描述",
      "confidence": 0.0-1.0
    }
  ],
  "confidence": 整页识别置信度 0.0-1.0
}
只返回 JSON，不要其他文字。"""


class QwenVlmClient:
    """Qwen-VL-Max API 客户端

    特性：
    1. asyncio + Semaphore 并发控制（文档 4.1 节，默认并发 5）
    2. 429 限流时指数退避重试
    3. MD5 + page_no 缓存（文档 6.2 节，TTL 30 天）
    4. 无 API key 时进入 mock 模式（本地联调）

    用法：
        client = QwenVlmClient(api_key="sk-xxx")
        results = client.recognize_pages_sync(images)
    """

    # 文档 4.1 节：DashScope 并发限制
    DEFAULT_CONCURRENCY = 5
    MAX_RETRY = 2
    CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 天

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        concurrency: int = DEFAULT_CONCURRENCY,
        cache_ttl: int = CACHE_TTL_SECONDS,
    ):
        """
        Args:
            api_key: DashScope API key，默认从 DASHSCOPE_API_KEY 环境变量读取
            model: VLM 模型名
            concurrency: 最大并发数（文档默认 5）
            cache_ttl: 缓存 TTL（秒），文档默认 30 天
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.concurrency = concurrency
        self.cache_ttl = cache_ttl
        self._client = None
        # 内存缓存（生产环境替换为 Redis）
        # key: parse:{pdf_md5}:{page_no}
        self._cache: Dict[str, VlmPageResult] = {}
        self._mock_mode = not bool(self.api_key)
        if self._mock_mode:
            logger.warning(
                "DASHSCOPE_API_KEY 未配置，VLM 客户端进入 mock 模式。"
                "将返回模拟结果用于本地联调，不调用真实 API"
            )

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _cache_key(self, pdf_md5: str, page_no: int) -> str:
        """文档 6.2 节：key 格式 parse:{md5}:{page_no}"""
        return f"parse:{pdf_md5}:{page_no}"

    def _get_cache(self, pdf_md5: str, page_no: int) -> Optional[VlmPageResult]:
        key = self._cache_key(pdf_md5, page_no)
        cached = self._cache.get(key)
        if cached is not None:
            cached.from_cache = True
        return cached

    def _put_cache(self, pdf_md5: str, page_no: int, result: VlmPageResult) -> None:
        key = self._cache_key(pdf_md5, page_no)
        result.from_cache = True
        self._cache[key] = result

    @staticmethod
    def compute_pdf_md5(pdf_path: str) -> str:
        """计算 PDF 文件 MD5 哈希（用于缓存 key）"""
        h = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # 异步识别
    # ------------------------------------------------------------------

    async def recognize_page(
        self,
        image_b64: str,
        page_no: int,
        pdf_md5: str = "",
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VlmPageResult:
        """单页识别（异步）

        Args:
            image_b64: base64 编码的页面图片
            page_no: 页码
            pdf_md5: PDF 的 MD5（用于缓存，空则不缓存）
            prompt: VLM prompt
        Returns:
            VlmPageResult
        """
        # 1. 缓存命中
        if pdf_md5:
            cached = self._get_cache(pdf_md5, page_no)
            if cached is not None:
                logger.debug("VLM 缓存命中: page=%d", page_no)
                return cached

        # 2. mock 模式（无 API key）
        if self._mock_mode:
            result = self._mock_recognize(page_no)
            if pdf_md5:
                self._put_cache(pdf_md5, page_no, result)
            return result

        # 3. 真实 API 调用
        result = await self._call_api_with_retry(image_b64, page_no, prompt)
        if result is not None and pdf_md5:
            self._put_cache(pdf_md5, page_no, result)
        return result or VlmPageResult(
            page_no=page_no,
            text="",
            confidence=0.0,
            formula_resolved=False,
        )

    async def recognize_pages(
        self,
        images: List[Tuple[int, str]],
        pdf_md5: str = "",
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> List[VlmPageResult]:
        """并发识别多页（文档 4.1 节：Semaphore 控制并发）

        Args:
            images: [(page_no, image_b64), ...]
            pdf_md5: PDF MD5
            prompt: VLM prompt
        Returns:
            VlmPageResult 列表，按 page_no 排序
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded_recognize(page_no: int, img_b64: str) -> VlmPageResult:
            async with semaphore:
                return await self.recognize_page(img_b64, page_no, pdf_md5, prompt)

        tasks = [_bounded_recognize(pno, img) for pno, img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        final: List[VlmPageResult] = []
        for i, r in enumerate(results):
            page_no = images[i][0]
            if isinstance(r, Exception):
                logger.error("VLM 识别失败 page=%d: %s", page_no, r)
                final.append(
                    VlmPageResult(
                        page_no=page_no,
                        text="",
                        confidence=0.0,
                        formula_resolved=False,
                    )
                )
            else:
                final.append(r)

        final.sort(key=lambda x: x.page_no)
        return final

    def recognize_pages_sync(
        self,
        images: List[Tuple[int, str]],
        pdf_md5: str = "",
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> List[VlmPageResult]:
        """同步包装（便于非 async 代码调用）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在 event loop 中，创建新 loop
                import threading
                result_holder: List = []
                def _run():
                    new_loop = asyncio.new_event_loop()
                    try:
                        result_holder.append(
                            new_loop.run_until_complete(
                                self.recognize_pages(images, pdf_md5, prompt)
                            )
                        )
                    finally:
                        new_loop.close()
                t = threading.Thread(target=_run)
                t.start()
                t.join()
                return result_holder[0] if result_holder else []
        except RuntimeError:
            pass

        return asyncio.run(self.recognize_pages(images, pdf_md5, prompt))

    def recognize_page_sync(
        self,
        image_b64: str,
        page_no: int,
        pdf_md5: str = "",
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VlmPageResult:
        """单页识别的同步包装（便于二次验证等非 async 代码调用）

        与 recognize_pages_sync 同理：
        - 已在事件循环中运行时（如 FastAPI async 请求内触发二次验证），
          改用新线程中的新事件循环执行，避免 asyncio.run 抛 RuntimeError；
        - 否则直接 asyncio.run。
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在 event loop 中，创建新线程 + 新 loop 执行
                import threading
                result_holder: List = []

                def _run():
                    new_loop = asyncio.new_event_loop()
                    try:
                        result_holder.append(
                            new_loop.run_until_complete(
                                self.recognize_page(
                                    image_b64, page_no, pdf_md5, prompt
                                )
                            )
                        )
                    finally:
                        new_loop.close()

                t = threading.Thread(target=_run)
                t.start()
                t.join()
                if result_holder:
                    return result_holder[0]
        except RuntimeError:
            pass

        return asyncio.run(
            self.recognize_page(image_b64, page_no, pdf_md5, prompt)
        )

    # ------------------------------------------------------------------
    # API 调用
    # ------------------------------------------------------------------

    async def _call_api_with_retry(
        self,
        image_b64: str,
        page_no: int,
        prompt: str,
    ) -> Optional[VlmPageResult]:
        """调用 Qwen-VL-Max API，指数退避重试

        文档 6.3 节：超时/限流重试 2 次，指数退避
        """
        if not self._ensure_client():
            return None

        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRY + 1):
            try:
                resp = await self._call_api(image_b64, prompt)
                result = self._parse_response(resp, page_no)
                logger.info(
                    "VLM 识别成功 page=%d, formulas=%d, tables=%d",
                    page_no, len(result.formulas), len(result.tables),
                )
                return result
            except _RateLimitError as e:
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt
                    logger.warning(
                        "VLM 429 限流 page=%d, %ds 后重试 (attempt %d/%d)",
                        page_no, wait, attempt + 1, self.MAX_RETRY + 1,
                    )
                    await asyncio.sleep(wait)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.MAX_RETRY:
                    wait = 2 ** attempt
                    logger.warning(
                        "VLM 调用失败 page=%d: %s, %ds 后重试",
                        page_no, e, wait,
                    )
                    await asyncio.sleep(wait)

        logger.error("VLM 最终失败 page=%d: %s", page_no, last_error)
        return None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            self._client = dashscope
            return True
        except ImportError:
            logger.warning("dashscope 包未安装，VLM API 不可用")
            return False

    async def _call_api(self, image_b64: str, prompt: str) -> dict:
        """实际 API 调用（dashscope 同步接口包装为 async）"""
        # dashscope 的 VLM 接口是同步的，用 run_in_executor 包装
        loop = asyncio.get_event_loop()

        def _sync_call():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/png;base64,{image_b64}"},
                        {"text": prompt},
                    ],
                }
            ]
            resp = self._client.MultiModalConversation.call(
                model=self.model,
                messages=messages,
            )
            if resp.status_code == 429:
                raise _RateLimitError(f"VLM 429 限流: {resp.message}")
            if resp.status_code != 200:
                raise RuntimeError(
                    f"VLM API 错误: code={resp.code}, msg={resp.message}"
                )
            return resp

        return await loop.run_in_executor(None, _sync_call)

    @staticmethod
    def _parse_response(resp, page_no: int) -> VlmPageResult:
        """解析 VLM API 响应为 VlmPageResult"""
        # dashscope MultiModalConversation 返回 resp.output.choices[0].message.content
        content = resp.output.choices[0].message.content
        if isinstance(content, list):
            # content 是 [{"text": "..."}] 格式
            text_content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        else:
            text_content = str(content)

        # 解析 JSON
        try:
            # 去掉 markdown 代码块包裹
            text_content = text_content.strip()
            if text_content.startswith("```"):
                lines = text_content.split("\n")
                text_content = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )
            data = json.loads(text_content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("VLM 响应 JSON 解析失败 page=%d: %s", page_no, e)
            return VlmPageResult(
                page_no=page_no,
                text=text_content,
                confidence=0.5,
                formula_resolved=False,
            )

        # 构造公式列表
        formulas = [
            VlmFormula(
                latex=f.get("latex", ""),
                location_desc=f.get("location_desc", ""),
                confidence=float(f.get("confidence", 0.5)),
                formula_id=f.get("formula_id", ""),
            )
            for f in data.get("formulas", [])
        ]

        # 构造表格列表
        tables = [
            VlmTable(
                markdown=t.get("markdown", ""),
                location_desc=t.get("location_desc", ""),
                confidence=float(t.get("confidence", 0.5)),
            )
            for t in data.get("tables", [])
        ]

        return VlmPageResult(
            page_no=page_no,
            text=data.get("text", ""),
            formulas=formulas,
            tables=tables,
            confidence=float(data.get("confidence", 0.5)),
            formula_resolved=len(formulas) > 0 or not data.get("formulas"),
        )

    # ------------------------------------------------------------------
    # Mock 模式（本地无 API key 时）
    # ------------------------------------------------------------------

    def _mock_recognize(self, page_no: int) -> VlmPageResult:
        """构造 mock VLM 结果（本地联调用）

        返回包含示例公式和表格的模拟结果，验证后续融合 + 绑定逻辑
        """
        mock_text = (
            f"This is page {page_no + 1}. "
            "We propose a novel architecture. "
            "Experiments show that our method achieves strong performance."
        )
        mock_formulas = [
            VlmFormula(
                latex=r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
                location_desc=f"第{page_no + 1}段后",
                confidence=0.95,
                formula_id=f"({page_no + 1})",
            ),
            VlmFormula(
                latex=r"\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)",
                location_desc="第二栏中部",
                confidence=0.65,
                formula_id=f"({page_no + 2})",
            ),
            VlmFormula(
                latex=r"E = mc^2",
                location_desc="softmax 计算公式如下",
                confidence=0.55,
                formula_id="",
            ),
        ]
        return VlmPageResult(
            page_no=page_no,
            text=mock_text,
            formulas=mock_formulas,
            tables=[],
            confidence=0.8,
            formula_resolved=True,
        )


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class _RateLimitError(Exception):
    """429 限流异常（触发指数退避重试）"""
    pass

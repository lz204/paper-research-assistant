"""PDF 转图模块（项目文档 4.1 节 + 第 5 章选型）

选型（文档第 5 章）：
- PyMuPDF（fitz）按页转图
- 理由：不依赖系统 poppler，纯 Python 包，Windows 上省装机环境
- 代价：包体积略大

文档 4.1 节关键技术细节：
- VLM 是页面级输入，必须整页图片喂入
  1. 公式与正文的版面关系要整页才能识别
  2. 切分会破坏坐标对齐的基准

输出：PageImage 列表，每页含 image_bytes + base64 编码（供 VLM API 直接使用）
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PageImage:
    """单页图片

    Attributes:
        page_no: 页码（0-based）
        image_bytes: PNG/JPEG 图片字节
        image_b64: base64 编码字符串（供 VLM API 直接使用）
        width: 图片宽度（像素）
        height: 图片高度（像素）
        dpi: 渲染 DPI
    """

    page_no: int
    image_bytes: bytes
    image_b64: str
    width: int
    height: int
    dpi: int


# ---------------------------------------------------------------------------
# PDF 转图器
# ---------------------------------------------------------------------------


class PdfToImageConverter:
    """PyMuPDF 按页转图

    Args:
        dpi: 渲染 DPI，默认 200（平衡清晰度和文件大小）
            - 150: 文件小，但小字号公式可能模糊
            - 200: 推荐，VLM 识别准确率高
            - 300: 高清，但文件大、API 上传慢
        image_format: 图片格式，"png" 或 "jpeg"
        jpeg_quality: JPEG 质量（1-100），仅 image_format="jpeg" 时生效
    """

    def __init__(
        self,
        dpi: int = 200,
        image_format: str = "png",
        jpeg_quality: int = 85,
    ):
        if dpi < 72 or dpi > 600:
            raise ValueError(f"dpi 应在 [72, 600] 区间，收到 {dpi}")
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.jpeg_quality = jpeg_quality
        self._fitz = None

    def _ensure_fitz(self) -> bool:
        """懒加载 PyMuPDF"""
        if self._fitz is not None:
            return True
        try:
            import fitz  # type: ignore
            self._fitz = fitz
            return True
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) 未安装，PDF 转图不可用。"
                "pip install pymupdf 后可启用"
            )
            return False

    def convert(
        self,
        pdf_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[PageImage]:
        """按页转图

        Args:
            pdf_path: PDF 文件路径
            page_range: (start, end) 页码区间（0-based，闭区间），
                       None 表示全部页
        Returns:
            PageImage 列表，按页码顺序
        """
        if not self._ensure_fitz():
            return []

        doc = self._fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            if page_range is None:
                start, end = 0, total_pages - 1
            else:
                start, end = page_range
                start = max(0, start)
                end = min(total_pages - 1, end)

            # 文档 6.5 节安全：限制单 PDF 页数 < 200
            if total_pages > 200:
                logger.warning(
                    "PDF 页数 %d 超过 200 上限，仅处理前 200 页（文档 6.5 安全限制）",
                    total_pages,
                )
                end = min(end, 199)

            images: List[PageImage] = []
            for page_no in range(start, end + 1):
                page = doc.load_page(page_no)
                # zoom = dpi / 72（PDF 默认 72 DPI）
                zoom = self.dpi / 72.0
                matrix = self._fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)

                # 转字节
                if self.image_format == "jpeg":
                    img_bytes = pix.tobytes(output="jpeg", jpg_quality=self.jpeg_quality)
                else:
                    img_bytes = pix.tobytes(output="png")

                img_b64 = base64.b64encode(img_bytes).decode("ascii")

                images.append(
                    PageImage(
                        page_no=page_no,
                        image_bytes=img_bytes,
                        image_b64=img_b64,
                        width=pix.width,
                        height=pix.height,
                        dpi=self.dpi,
                    )
                )

            logger.info(
                "PDF 转图完成: %s, %d 页, dpi=%d",
                pdf_path, len(images), self.dpi,
            )
            return images
        finally:
            doc.close()

    def convert_single_page(
        self,
        pdf_path: str,
        page_no: int,
    ) -> Optional[PageImage]:
        """转单页图片（用于二次验证时裁剪区域前的整页获取）"""
        images = self.convert(pdf_path, page_range=(page_no, page_no))
        return images[0] if images else None


# ---------------------------------------------------------------------------
# 区域裁剪（用于文档 4.1 节"裁剪该区域图片单独送 VLM 二次验证"）
# ---------------------------------------------------------------------------


def crop_region(
    page_image: PageImage,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
) -> Optional[PageImage]:
    """从整页图片裁剪指定区域（坐标为 PDF 坐标系，需归一化）

    Args:
        page_image: 整页图片
        x0, y0, x1, y1: PDF 坐标系下的区域边界
        page_width, page_height: PDF 页面宽高（用于归一化）
    Returns:
        裁剪后的 PageImage（page_no 保持原值），失败返回 None
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logger.warning("Pillow 未安装，区域裁剪不可用。pip install pillow 后可启用")
        return None

    # PDF 坐标 → 图片像素坐标（归一化）
    x0_ratio = max(0.0, min(1.0, x0 / page_width)) if page_width > 0 else 0.0
    y0_ratio = max(0.0, min(1.0, y0 / page_height)) if page_height > 0 else 0.0
    x1_ratio = max(0.0, min(1.0, x1 / page_width)) if page_width > 0 else 1.0
    y1_ratio = max(0.0, min(1.0, y1 / page_height)) if page_height > 0 else 1.0

    left = int(x0_ratio * page_image.width)
    upper = int(y0_ratio * page_image.height)
    right = int(x1_ratio * page_image.width)
    lower = int(y1_ratio * page_image.height)

    if right <= left or lower <= upper:
        logger.warning("裁剪区域无效: (%d,%d,%d,%d)", left, upper, right, lower)
        return None

    img = Image.open(io.BytesIO(page_image.image_bytes))
    cropped = img.crop((left, upper, right, lower))

    buf = io.BytesIO()
    fmt = "PNG" if page_image.image_bytes[:4] == b"\x89PNG" else "JPEG"
    cropped.save(buf, format=fmt)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    return PageImage(
        page_no=page_image.page_no,
        image_bytes=img_bytes,
        image_b64=img_b64,
        width=cropped.width,
        height=cropped.height,
        dpi=page_image.dpi,
    )

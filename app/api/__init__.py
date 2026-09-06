"""FastAPI 路由模块（项目文档第 3 / 10 章 + 6.5 节安全）

包含：PDF 上传、论文问答、综述生成三大核心功能的 HTTP 接口。

模块总览（项目文档附录）：
- routes.py    FastAPI 路由（/upload /ask /review /health）

容错：fastapi 未安装时模块仍可 import（schema dataclass 可独立使用），
create_app / get_app 返回 None，保证无 fastapi 依赖也能跑通业务联调。
"""

from .routes import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    MAX_QUESTION_LEN,
    MAX_REVIEW_TOPIC_LEN,
    ReviewRequest,
    ReviewResponse,
    UploadRequest,
    UploadResponse,
    create_app,
    get_app,
    run_dev,
)

__all__ = [
    # 请求/响应 schema
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "ReviewRequest",
    "ReviewResponse",
    "UploadRequest",
    "UploadResponse",
    # 文档 6.5 节安全限制
    "MAX_PDF_PAGES",
    "MAX_PDF_SIZE_MB",
    "MAX_QUESTION_LEN",
    "MAX_REVIEW_TOPIC_LEN",
    # 应用工厂
    "create_app",
    "get_app",
    "run_dev",
]

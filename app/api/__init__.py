"""FastAPI 路由模块（项目文档第 3 / 10 章 + 6.5 节安全）

包含：PDF 上传（可选 VLM 解析链路）、论文问答、综述生成、多语种翻译、
Agent 工具调用、文档索引与成本看板接口。

模块总览（项目文档附录）：
- routes.py    FastAPI 路由：
    POST /upload（use_vlm 可选 VLM 解析链路，返回 vlm_used/formulas/parse_mode）
    POST /ask、POST /review、POST /translate
    POST /agent、GET /agent/tools、POST /agent/tools/call（供 MCP 适配器）
    GET /health、GET /documents、GET /cost/summary

容错：fastapi 未安装时模块仍可 import（schema dataclass 可独立使用），
create_app / get_app 返回 None，保证无 fastapi 依赖也能跑通业务联调。
"""

from .routes import (
    AgentBody,
    AgentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    MAX_AGENT_QUESTION_LEN,
    MAX_AGENT_STEPS,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    MAX_QUESTION_LEN,
    MAX_REVIEW_TOPIC_LEN,
    MAX_TRANSLATE_LEN,
    MAX_VLM_PAGES,
    ReviewRequest,
    ReviewResponse,
    TranslateRequest,
    TranslateResponse,
    UploadRequest,
    UploadResponse,
    create_app,
    get_app,
    run_dev,
)

__all__ = [
    # 请求/响应 schema
    "AgentBody",
    "AgentResponse",
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "ReviewRequest",
    "ReviewResponse",
    "TranslateRequest",
    "TranslateResponse",
    "UploadRequest",
    "UploadResponse",
    # 文档 6.5 节安全限制 + 翻译输入长度限制 + VLM/Agent 限制
    "MAX_AGENT_QUESTION_LEN",
    "MAX_AGENT_STEPS",
    "MAX_PDF_PAGES",
    "MAX_PDF_SIZE_MB",
    "MAX_QUESTION_LEN",
    "MAX_REVIEW_TOPIC_LEN",
    "MAX_TRANSLATE_LEN",
    "MAX_VLM_PAGES",
    # 应用工厂
    "create_app",
    "get_app",
    "run_dev",
]

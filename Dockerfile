# =====================================================================
# 多语种文献翻译与论文研读助手 — Dockerfile
# 文档第 10 章：FastAPI 单体 + Streamlit 前端，纯 CPU（无 GPU）
# 镜像：Python 3.13 slim，镜像大小约 1.2GB，启动后内存 < 1GB
# =====================================================================

FROM python:3.13-slim AS base

LABEL maintainer="paper-research-assistant@example.com"
LABEL description="多语种文献翻译与论文研读助手 — 求职项目（阿里云 DashScope API-first）"

# ---- 系统级依赖 ----
# poppler-utils: pdf2image 需要
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- 工作目录 ----
WORKDIR /app

# ---- 先装依赖层（利用 Docker 缓存）----
# 优先用 requirements.lock 精确安装（避免容器内解析出与 FastAPI 锁定的
# starlette 版本冲突，例如 streamlit 1.51+ / fastapi 新版）；无 lock 时回退 txt
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip \
    && if [ -f requirements.lock ]; then \
         pip install --no-cache-dir -r requirements.lock; \
       else \
         pip install --no-cache-dir -r requirements.txt; \
       fi

# ---- 复制项目代码 ----
COPY app/      app/
COPY tests/    tests/
COPY logs/     logs/
COPY Makefile  Makefile

# ---- 环境变量默认值（生产环境在 docker-compose 或 .env 中覆盖）----
ENV APP_ENV=production
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- 健康检查 ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# ---- 暴露端口：FastAPI 8000 + Streamlit 8501 ----
EXPOSE 8000
EXPOSE 8501

# ---- 默认入口：用 supervisord 同时起 FastAPI + Streamlit？
#      遵循文档 10 章"不上微服务，单体"的精神，
#      简化为 FastAPI 主进程（Streamlit 单独起容器更清晰）。
#      若需在单容器启动两者，可改 CMD 启动 supervisord。
CMD ["python", "-m", "uvicorn", \
     "app.api.routes:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

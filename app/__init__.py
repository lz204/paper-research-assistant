"""paper-research-assistant 应用主包"""

# 子包导入（保证 from app import xxx 可用）
from . import (
    api,
    export,
    llm,
    observability,
    parser,
    rag,
    review,
)

__all__ = [
    "api",
    "export",
    "llm",
    "observability",
    "parser",
    "rag",
    "review",
]


# ---------------------------------------------------------------------------
# 启动时自动加载 .env：让所有 os.getenv(...) 自动读到项目根的 .env
# 优先用 python-dotenv；没装则用纯 Python 解析（不依赖第三方包，避免缺包不可用）
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    from pathlib import Path
    import os

    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        v = v.strip().strip("'").strip('"')
        # override=False：已存在的系统环境变量优先级更高
        os.environ.setdefault(k, v)


_load_dotenv()


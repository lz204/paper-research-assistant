"""pytest 共享配置。

所有 pytest 用例统一在 mock 模式运行：清空 DashScope / Langfuse 环境变量，
避免开发机 .env 中的真实 key 让测试误走真实 API（与 tests/unit/verify_*.py 的隔离策略一致）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path（配合 pyproject 的 pythonpath 双保险）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在本模块被任何 app.* import 之前完成，否则 app/__init__._load_dotenv 已把 key 读入
for _k in (
    "DASHSCOPE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
):
    os.environ.pop(_k, None)
    os.environ[_k] = ""

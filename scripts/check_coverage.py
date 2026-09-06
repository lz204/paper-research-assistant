"""核心模块覆盖率门槛校验（文档 6.4 节：context_manager / fusion_engine / citation_verifier >= 85%）。

用法（先生成覆盖率数据，如 make coverage）：
    python scripts/check_coverage.py [--json .coverage-core.json]

退出码：所有核心模块均 >= 85% 返回 0，否则返回 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 核心模块：覆盖率必须 >= 85%
CORE_MODULES = [
    "app/rag/context_manager.py",
    "app/parser/fusion_engine.py",
    "app/review/citation_verifier.py",
]
THRESHOLD = 85.0


def _norm(p: str) -> str:
    # coverage json 在 Windows 使用反斜杠，统一为正斜杠便于匹配
    return Path(p).as_posix()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=".coverage-core.json", help="coverage json 数据文件")
    args = ap.parse_args(argv)

    data_path = Path(args.json)
    if not data_path.exists():
        print(f"[coverage-check] 未找到 {args.json}，请先运行 make coverage", file=sys.stderr)
        return 1

    data = json.loads(data_path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    # 路径归一化后建索引（Windows 反斜杠 → 正斜杠）
    index = {_norm(p): v for p, v in files.items()}

    failures: list[str] = []
    for module in CORE_MODULES:
        info = index.get(module)
        if info is None:
            failures.append(f"{module}: 无覆盖率数据（未被执行）")
            continue
        rate = info["summary"]["percent_covered"]
        print(f"[coverage-check] {module}: {rate:.1f}%")
        if rate < THRESHOLD:
            failures.append(f"{module}: {rate:.1f}% < {THRESHOLD:.0f}%")

    if failures:
        print("[coverage-check] 未达标：", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    print(f"[coverage-check] ✓ 核心模块覆盖率均 >= {THRESHOLD:.0f}%（文档 6.4 节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

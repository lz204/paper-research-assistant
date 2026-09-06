# =====================================================================
# 多语种文献翻译与论文研读助手 — Makefile
# 文档第 10 章：make up / make test / make eval
#
# 说明：
# - Linux / macOS / WSL 下直接 make <target>
# - Windows 原生若无 make，可直接执行注释中的等价命令（已标注 PowerShell 等价）
# - 所有命令在项目根目录执行
# =====================================================================

.PHONY: help env init up down restart api frontend test test-unit \
        verify-infra verify-rag verify-review verify-parser verify-context \
        test-pytest coverage coverage-check lint lint-ruff typecheck \
        e2e e2e-mock e2e-dashscope eval dataset clean-deps logs \
        install-deps install-dev lint

# ---- 默认目标 ----
help:
	@echo "可用目标："
	@echo "  env              复制 .env.example → .env（首次运行）"
	@echo "  init             初始化（安装依赖 + 生成 .env + 建 PG 容器 db）"
	@echo "  up               docker-compose 启动（PG + Redis + App）"
	@echo "  down             docker-compose 停止"
	@echo "  restart          重启应用容器"
	@echo "  api              本地启动 FastAPI 后端（Windows: python -m uvicorn...）"
	@echo "  frontend         本地启动 Streamlit 前端（Windows: streamlit run ...）"
	@echo ""
	@echo "  test             运行全部单测（文档 6.4 节：push 触发）"
	@echo "  verify-context   运行 context_manager 单测"
	@echo "  verify-rag       运行 RAG 单测"
	@echo "  verify-review    运行综述单测"
	@echo "  verify-parser    运行 parser 单测"
	@echo "  verify-infra     运行基础设施单测"
	@echo ""
	@echo "  e2e              运行全部 E2E（mock + DashScope 真实）"
	@echo "  e2e-mock         只跑 mock 模式 E2E（必通过，无 API key 也能跑）"
	@echo "  e2e-dashscope    只跑 DashScope 真实调用 E2E（无 key 会提示并跳过）"
	@echo ""
	@echo "  eval             自动化评测全集（文档 6.4 节，约 30 分钟）"
	@echo "  dataset          构建/扩展 QA 数据集（内置样例集，可抓取更多 arXiv）"
	@echo ""
	@echo "  install-deps     pip install -r requirements.txt"
	@echo "  install-dev      pip install -r requirements-dev.txt（pytest/coverage/ruff/mypy）"
	@echo "  lint             语法校验（无第三方 linter 时纯 py_compile）"
	@echo "  logs             查看 App 容器日志"

# ---- 初始化 ----
env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "已生成 .env，请填入 DASHSCOPE_API_KEY"; \
	else echo ".env 已存在，跳过"; fi

install-deps:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

init: env install-deps
	@echo "初始化完成。下一步："
	@echo "  1. 编辑 .env，填入真实 DASHSCOPE_API_KEY（必填项）"
	@echo "  2. 运行 make up 启动 docker-compose"
	@echo "  3. 运行 make e2e-mock 验证 mock 模式全链路"

# ---- docker-compose 三容器（文档 10 章）----
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart app

logs:
	docker compose logs -f app

# ---- 本地启动（不走 docker）----
api:
	python -m uvicorn app.api.routes:create_app --factory --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.headless true

# ---- 单测（文档 6.4 节 push 触发）----
# 按粒度拆分，便于 CI 按失败模块 rerun
test: verify-context verify-rag verify-review verify-parser verify-infra
	@echo "全部单测通过 ✓"

# pytest 回归测试（需 pip install -r requirements-dev.txt）
# 不显式传路径：pytest 使用 pyproject testpaths=tests/unit，只收集 test_*.py
test-pytest:
	python -m pytest -v

# 覆盖率报告（需 requirements-dev.txt）
# 关键：verify_*.py 脚本是核心模块的主要执行路径（含 chunker/fusion/verifier 全场景），
# pytest 只覆盖一部分，因此 coverage 必须合并两者，否则核心模块覆盖率会被严重低估。
coverage:
	@rm -f .coverage; \
	python -m coverage run --source=app tests/unit/verify_context_manager.py > /dev/null 2>&1; \
	python -m coverage run --source=app --append tests/unit/verify_rag_pipeline.py > /dev/null 2>&1; \
	python -m coverage run --source=app --append tests/unit/verify_review_pipeline.py > /dev/null 2>&1; \
	python -m coverage run --source=app --append tests/unit/verify_parser_pipeline.py > /dev/null 2>&1; \
	python -m coverage run --source=app --append tests/unit/verify_infrastructure.py > /dev/null 2>&1; \
	python -m pytest --cov=app --cov-append --cov-report=term-missing --cov-report=xml

# 覆盖率门槛校验（文档 6.4 节：核心模块 >= 85%）
# 先由 coverage json 导出单文件数据，再交给 scripts/check_coverage.py 判定
coverage-check:
	@python -m coverage json -o .coverage-core.json > /dev/null 2>&1; \
	python scripts/check_coverage.py --json .coverage-core.json

verify-context:
	python tests/unit/verify_context_manager.py

verify-rag:
	python tests/unit/verify_rag_pipeline.py

verify-review:
	python tests/unit/verify_review_pipeline.py

verify-parser:
	python tests/unit/verify_parser_pipeline.py

verify-infra:
	python tests/unit/verify_infrastructure.py

# ---- E2E 端到端（文档第 3 章架构图验证）----
e2e: e2e-mock e2e-dashscope
	@echo "全部 E2E 执行完成 ✓"

e2e-mock:
	python tests/e2e/test_e2e_mock.py

e2e-dashscope:
	@if [ -z "$$DASHSCOPE_API_KEY" ]; then \
		echo "⚠ 未设置 DASHSCOPE_API_KEY 环境变量，跳过真实调用 E2E"; \
		echo "  填入方式：export DASHSCOPE_API_KEY=sk-xxxxxxx  或  修改 .env 文件"; \
	else \
		python tests/e2e/test_e2e_dashscope.py; \
	fi

# ---- 自动化评测（文档 6.4 节，nightly）----
dataset:
	python tests/qa_dataset/build_dataset.py

eval:
	python tests/eval/run_eval.py

# ---- Lint（企业级：ruff + mypy；未装 dev 依赖时退化为 py_compile）----
lint:
	@if python -c "import ruff" 2>/dev/null; then \
		python -m ruff check app tests; \
		python -m ruff format --check app tests; \
	else \
		echo "ruff 未安装，退回 py_compile（pip install -r requirements-dev.txt 后启用）"; \
		python -m compileall -q app tests || echo "py_compile 发现语法错误"; \
	fi
	@echo "lint 完成"

lint-ruff:
	python -m ruff check --fix app tests

typecheck:
	@if python -c "import mypy" 2>/dev/null; then \
		python -m mypy app; \
	else \
		echo "mypy 未安装，跳过类型检查（pip install -r requirements-dev.txt 后启用）"; \
	fi

# ---- 清理（清理容器 / 缓存，不删 .env）----
clean-deps:
	rm -rf __pycache__ app/__pycache__ app/*/__pycache__ tests/*/__pycache__ .pytest_cache .coverage .coverage-core.json htmlcov

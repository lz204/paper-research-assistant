# =====================================================================
# 多语种文献翻译与论文研读助手 — Windows 原生开发脚本（替代 make）
#
# 背景：项目 Makefile 依赖 POSIX shell（/dev/null、rm、if [ ] 等），
#       Windows 原生环境通常没有 make，因此提供本脚本作为等价入口。
# 编码：本文件为 UTF-8 with BOM（Windows PowerShell 5.1 正确解析中文所需）。
#
# 用法（在项目根目录或任意目录执行均可，脚本自动定位项目根）：
#   .\scripts\dev.ps1                  # 显示帮助
#   .\scripts\dev.ps1 install          # 安装运行依赖（可选装开发依赖）
#   .\scripts\dev.ps1 test             # 全部 verify 脚本 + E2E mock + 补齐模块验证
#   .\scripts\dev.ps1 test-pytest      # pytest 回归测试
#   .\scripts\dev.ps1 coverage         # 覆盖率报告 + 核心模块 >=85% 门控
#   .\scripts\dev.ps1 e2e-mock         # E2E mock（无需 API Key）
#   .\scripts\dev.ps1 eval             # 自动化评测 quick 模式（mock，无需 Key）
#   .\scripts\dev.ps1 eval -EvalMode nightly   # 真实 DashScope 评测（需 Key）
#   .\scripts\dev.ps1 verify-completion        # 翻译/GB-T/结构拆解验证（mock，不产生费用）
#   .\scripts\dev.ps1 api              # 启动 FastAPI 后端（http://127.0.0.1:8000）
#   .\scripts\dev.ps1 frontend         # 启动 Streamlit 前端（http://127.0.0.1:8501）
#   .\scripts\dev.ps1 all              # test + test-pytest + e2e-mock + eval（CI 等价）
#
# 说明：
# - 优先使用项目 .venv\Scripts\python.exe，不存在时回退到 PATH 中的 python
# - 统一设置 PYTHONIOENCODING=utf-8 / PYTHONUTF8=1：避免 Windows 管道/重定向
#   下 ✓ 等字符触发 GBK UnicodeEncodeError
# - Windows PowerShell 5.1 会把 native 程序的 stderr 包装成 NativeCommandError，
#   因此本脚本不用 $ErrorActionPreference='Stop'，统一以 exit code 判定失败
# =====================================================================
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'help', 'install', 'test', 'test-pytest', 'coverage', 'e2e-mock',
        'eval', 'verify-completion', 'verify-agent', 'agent-eval', 'mcp-smoke',
        'api', 'frontend', 'all'
    )]
    [string]$Action = 'help',

    # eval 模式：quick（mock，默认）/ nightly（真实 DashScope，需 Key）
    [ValidateSet('quick', 'nightly')]
    [string]$EvalMode = 'quick',

    # 评测报告输出路径（相对项目根）
    [string]$ReportPath = 'report_quick.json'
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = 'python'
    Write-Host "[warn] 未找到 .venv\Scripts\python.exe，回退使用 PATH 中的 python" -ForegroundColor Yellow
}

# 统一 UTF-8 输出（修复 Windows GBK 控制台/管道下中文与 ✓ 的编码问题）
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

function Show-Help {
    Write-Host ""
    Write-Host "多语种文献翻译与论文研读助手 — Windows 开发脚本" -ForegroundColor Cyan
    Write-Host ("项目根: {0}" -f $ProjectRoot)
    Write-Host ("Python: {0}" -f $Python)
    Write-Host ""
    Write-Host "可用动作:"
    Write-Host "  install              安装 requirements.lock（含前端依赖，回退 requirements.txt）+ 开发依赖"
    Write-Host "  test                 全部 verify 脚本 + E2E mock + 补齐模块验证（mock）"
    Write-Host "  test-pytest          pytest 回归测试"
    Write-Host "  coverage             覆盖率报告 + 核心模块 >=85% 门控"
    Write-Host "  e2e-mock             E2E mock 全链路"
    Write-Host "  eval                 自动化评测（默认 quick/mock）"
    Write-Host "  verify-completion    翻译/GB-T 7714/论文结构拆解验证（mock）"
    Write-Host "  verify-agent         Agent 链路验证（工具/ReAct/LangGraph/边界，mock）"
    Write-Host "  agent-eval           Agent 评测（任务成功率/工具选择/步数/成本，mock）"
    Write-Host "  mcp-smoke            MCP server 协议冒烟（tools/call 需先启动 api）"
    Write-Host "  api                  启动 FastAPI 后端 http://127.0.0.1:8000"
    Write-Host "  frontend             启动 Streamlit 前端 http://127.0.0.1:8501"
    Write-Host "  all                  test + test-pytest + e2e-mock + eval"
    Write-Host ""
    Write-Host "示例: .\scripts\dev.ps1 test"
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string[]]$PyArgs,
        [string]$Title = ''
    )
    if ($Title) {
        Write-Host ""
        Write-Host ("==> {0}" -f $Title) -ForegroundColor Cyan
    }
    # 合并 stderr 逐行展示：PS 5.1 下 Python 日志走 stderr，不合并会被当错误文本
    & $Python @PyArgs 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw ("命令失败（exit {0}）: python {1}" -f $LASTEXITCODE, ($PyArgs -join ' '))
    }
}

function Test-PythonModule {
    param([string]$ModuleName)
    & $Python -c ("import importlib; importlib.import_module('{0}')" -f $ModuleName) 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        'help' { Show-Help }

        'install' {
            # 优先 requirements.lock：精确可复现（含 streamlit==1.50.0 等前端依赖，
            # 且不会把 starlette 解析到与 fastapi==0.110 冲突的版本）
            $reqFile = 'requirements.txt'
            if (Test-Path (Join-Path $ProjectRoot 'requirements.lock')) {
                $reqFile = 'requirements.lock'
            }
            Invoke-Python -PyArgs @('-m', 'pip', 'install', '-r', $reqFile) -Title ("安装运行依赖（{0}）" -f $reqFile)
            if (Test-Path (Join-Path $ProjectRoot 'requirements-dev.txt')) {
                Invoke-Python -PyArgs @('-m', 'pip', 'install', '-r', 'requirements-dev.txt') -Title '安装开发依赖'
            }
        }

        'test' {
            $scripts = @(
                'tests/unit/verify_context_manager.py',
                'tests/unit/verify_rag_pipeline.py',
                'tests/unit/verify_review_pipeline.py',
                'tests/unit/verify_parser_pipeline.py',
                'tests/unit/verify_infrastructure.py',
                'tests/e2e/test_e2e_mock.py',
                'tests/unit/verify_completion_modules.py --mock',
                'tests/unit/verify_agent_pipeline.py'
            )
            foreach ($s in $scripts) {
                $parts = $s -split ' '
                $file = $parts[0]
                $extra = @()
                if ($parts.Length -gt 1) { $extra = $parts[1..($parts.Length - 1)] }
                Invoke-Python -PyArgs (@($file) + $extra) -Title $file
            }
            Write-Host ""
            Write-Host "全部验证脚本通过 ✓" -ForegroundColor Green
        }

        'test-pytest' {
            Invoke-Python -PyArgs @('-m', 'pytest', '-v') -Title 'pytest 回归测试'
        }

        'coverage' {
            if (-not (Test-PythonModule 'coverage')) {
                throw 'coverage 未安装，请先执行 .\scripts\dev.ps1 install 或 pip install -r requirements-dev.txt'
            }
            $coverageFile = Join-Path $ProjectRoot '.coverage'
            if (Test-Path $coverageFile) { Remove-Item $coverageFile -Force }
            $verifyScripts = @(
                'tests/unit/verify_context_manager.py',
                'tests/unit/verify_rag_pipeline.py',
                'tests/unit/verify_review_pipeline.py',
                'tests/unit/verify_parser_pipeline.py',
                'tests/unit/verify_infrastructure.py'
            )
            $first = $true
            foreach ($s in $verifyScripts) {
                $covArgs = @('-m', 'coverage', 'run', '--source=app')
                if (-not $first) { $covArgs += '--append' }
                $covArgs += $s
                $first = $false
                & $Python @covArgs 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    throw ("覆盖率采集失败: {0}" -f $s)
                }
            }
            if (Test-PythonModule 'pytest_cov') {
                Invoke-Python -PyArgs @('-m', 'pytest', '--cov=app', '--cov-append', '-q') -Title 'pytest 覆盖率采集'
            } else {
                Write-Host "[warn] pytest-cov 未安装，覆盖率仅包含 verify 脚本口径" -ForegroundColor Yellow
            }
            Invoke-Python -PyArgs @('-m', 'coverage', 'json', '-o', '.coverage-core.json') -Title '导出覆盖率 JSON'
            Invoke-Python -PyArgs @('scripts/check_coverage.py', '--json', '.coverage-core.json') -Title '核心模块覆盖率门控'
        }

        'e2e-mock' {
            Invoke-Python -PyArgs @('tests/e2e/test_e2e_mock.py') -Title 'E2E mock 全链路'
        }

        'eval' {
            Invoke-Python -PyArgs @('tests/eval/run_eval.py', '--mode', $EvalMode, '--output', $ReportPath) `
                -Title ("自动化评测（mode={0}）" -f $EvalMode)
        }

        'verify-completion' {
            Invoke-Python -PyArgs @('tests/unit/verify_completion_modules.py', '--mock') `
                -Title '补齐模块验证（翻译 / GB/T 7714 / 结构拆解，mock）'
        }

        'verify-agent' {
            Invoke-Python -PyArgs @('tests/unit/verify_agent_pipeline.py') `
                -Title 'Agent 链路验证（工具注册 / ReAct / LangGraph / 边界，mock）'
        }

        'agent-eval' {
            Invoke-Python -PyArgs @('tests/eval/run_agent_eval.py', '--mode', 'mock', '--output', 'report_agent.json') `
                -Title 'Agent 评测（mock：任务成功率 / 工具选择 / 步数 / 成本）'
        }

        'mcp-smoke' {
            Invoke-Python -PyArgs @('scripts/agent_mcp_smoke.py') `
                -Title 'MCP server 协议冒烟'
        }

        'api' {
            if (-not (Test-PythonModule 'fastapi')) {
                throw 'fastapi 未安装，请先执行 .\scripts\dev.ps1 install'
            }
            if (-not (Test-PythonModule 'uvicorn')) {
                throw 'uvicorn 未安装，请先执行 .\scripts\dev.ps1 install'
            }
            Write-Host "启动 FastAPI: http://127.0.0.1:8000 （Ctrl+C 停止）" -ForegroundColor Cyan
            Write-Host "接口文档: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
            & $Python -m uvicorn app.api.routes:create_app --factory --host 127.0.0.1 --port 8000 --reload 2>&1 |
                ForEach-Object { Write-Host $_ }
        }

        'frontend' {
            if (-not (Test-PythonModule 'streamlit')) {
                throw 'streamlit 未安装，请先执行 .\scripts\dev.ps1 install（或 pip install streamlit plotly）'
            }
            Write-Host "启动 Streamlit: http://127.0.0.1:8501 （Ctrl+C 停止）" -ForegroundColor Cyan
            Write-Host "提示：先另开终端执行 .\scripts\dev.ps1 api 可启用前端 API 模式" -ForegroundColor Cyan
            & $Python -m streamlit run app/frontend/streamlit_app.py `
                --server.port 8501 --server.address 127.0.0.1 --server.headless true 2>&1 |
                ForEach-Object { Write-Host $_ }
        }

        'all' {
            & $PSCommandPath test
            if ($LASTEXITCODE -ne 0) { throw 'test 阶段失败' }
            & $PSCommandPath test-pytest
            if ($LASTEXITCODE -ne 0) { throw 'test-pytest 阶段失败' }
            & $PSCommandPath e2e-mock
            if ($LASTEXITCODE -ne 0) { throw 'e2e-mock 阶段失败' }
            & $PSCommandPath eval
            if ($LASTEXITCODE -ne 0) { throw 'eval 阶段失败' }
            & $PSCommandPath verify-completion
            Write-Host ""
            Write-Host "全部检查通过 ✓" -ForegroundColor Green
        }
    }
} catch {
    Write-Host ""
    Write-Host ("[ERROR] {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

@echo off
REM LLM服务器启动脚本 (Windows版本)
echo ========================================
echo LLM大模型服务器启动器 (Windows)
echo ========================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo 警告: 虚拟环境不存在
    echo 请先运行: python -m venv venv
    pause
    exit /b 1
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 检查依赖
python -c "import torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo 错误: 缺少依赖，请运行: pip install -r requirements.txt
    pause
    exit /b 1
)

REM 检查模型
if not exist "data\models" (
    echo 警告: 模型目录不存在
    echo 请先下载模型: python model_download.py download --model qwen2.5-32b
    pause
    exit /b 1
)

echo 环境检查通过
echo.
echo 请选择启动方式:
echo 1. VLLM引擎 (高性能，推荐)
echo 2. Transformers引擎 (标准)
echo 3. 退出

set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo 启动VLLM服务器...
    python vllm_server.py
) else if "%choice%"=="2" (
    echo 启动Transformers服务器...
    python model_server.py
) else if "%choice%"=="3" (
    echo 再见!
    exit /b 0
) else (
    echo 无效选择
    pause
    exit /b 1
)

pause

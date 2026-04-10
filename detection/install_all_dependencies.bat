@echo off
REM Conda 环境完整依赖安装脚本
REM 请在 conda activate vmamba 后运行此脚本

echo ========================================
echo VMamba 检测任务 - 完整依赖安装
echo ========================================
echo.

REM 检查是否在 conda 环境中
if "%CONDA_DEFAULT_ENV%"=="" (
    echo 警告: 未检测到 conda 环境！
    echo 请先运行: conda activate vmamba
    echo.
    set /p continue="是否继续？(y/n): "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
) else (
    echo 当前 conda 环境: %CONDA_DEFAULT_ENV%
)

echo.
echo 步骤 1: 安装项目基础依赖
echo ----------------------------------------
cd /d "%~dp0\.."
if exist requirements.txt (
    echo 正在安装 requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 基础依赖安装失败！
        pause
        exit /b 1
    )
    echo ✓ 基础依赖安装成功
) else (
    echo 警告: 未找到 requirements.txt
)

echo.
echo 步骤 2: 安装 Selective Scan
echo ----------------------------------------
if exist kernels\selective_scan (
    cd kernels\selective_scan
    echo 正在安装 selective_scan...
    pip install .
    if %errorlevel% neq 0 (
        echo Selective Scan 安装失败！
        pause
        exit /b 1
    )
    echo ✓ Selective Scan 安装成功
    cd ..\..
) else (
    echo 警告: 未找到 kernels\selective_scan 目录
)

echo.
echo 步骤 3: 安装检测依赖
echo ----------------------------------------
echo 正在安装 mmengine, mmcv 等...
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
if %errorlevel% neq 0 (
    echo 检测基础依赖安装失败！
    pause
    exit /b 1
)
echo ✓ 检测基础依赖安装成功

echo.
echo 正在安装 MMDetection...
pip install mmdet==3.3.0
if %errorlevel% neq 0 (
    echo MMDetection 安装失败！
    pause
    exit /b 1
)
echo ✓ MMDetection 安装成功

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 正在验证安装...
python -c "from mmengine.config import Config; print('✓ mmengine')" 2>nul
python -c "from mmdet.apis import init_detector; print('✓ mmdet')" 2>nul
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>nul

echo.
echo 验证完成！
pause

@echo off
REM conda 环境依赖安装脚本（Windows）
echo ========================================
echo 检测依赖安装脚本 (Conda)
echo ========================================
echo.

REM 检查是否在 conda 环境中
if "%CONDA_DEFAULT_ENV%"=="" (
    echo 警告: 未检测到 conda 环境
    echo 请先激活 conda 环境:
    echo   conda activate vmamba
    echo.
    pause
    exit /b 1
) else (
    echo 当前 conda 环境: %CONDA_DEFAULT_ENV%
)

echo.
echo 正在安装基础依赖...
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
if %errorlevel% neq 0 (
    echo 基础依赖安装失败！
    pause
    exit /b 1
)

echo.
echo 正在安装 MMDetection...
pip install mmdet==3.3.0
if %errorlevel% neq 0 (
    echo MMDetection 安装失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 正在验证安装...
python -c "from mmengine.config import Config; print('✓ mmengine 安装成功')"
python -c "from mmdet.apis import init_detector; print('✓ mmdet 安装成功')"
python -c "from mmdet.utils import register_all_modules; print('✓ mmdet.utils 安装成功')"

echo.
echo 验证完成！
pause


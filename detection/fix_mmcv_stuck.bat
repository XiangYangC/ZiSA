@echo off
REM 快速解决 MMCV 编译卡住问题
echo ========================================
echo 解决 MMCV 编译卡住问题
echo ========================================
echo.
echo 请先按 Ctrl+C 停止当前卡住的编译进程
echo 然后按任意键继续...
pause >nul

echo.
echo 步骤 1: 卸载卡住的 mmcv 版本
echo ----------------------------------------
pip uninstall mmcv mmcv-full -y

echo.
echo 步骤 2: 安装预编译版本（CUDA 11.8）
echo ----------------------------------------
echo 如果你使用的是 CUDA 12.1，请手动修改脚本
echo.
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ MMCV 安装成功！
    echo ========================================
    echo.
    echo 下一步: pip install mmdet==3.3.0
) else (
    echo.
    echo ========================================
    echo ✗ 安装失败，请尝试其他版本
    echo ========================================
    echo.
    echo CUDA 12.1:
    echo   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
    echo.
    echo CPU 版本:
    echo   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
)

pause


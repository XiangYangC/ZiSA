#!/bin/bash
# conda 环境依赖安装脚本（Linux/Mac）
# Windows 用户请使用 install_dependencies_conda.bat

echo "========================================"
echo "检测依赖安装脚本 (Conda)"
echo "========================================"
echo ""

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到 conda 环境"
    echo "请先激活 conda 环境:"
    echo "  conda activate vmamba"
    echo ""
    read -p "是否继续？(y/n): " answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
else
    echo "当前 conda 环境: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "正在安装基础依赖..."
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex

if [ $? -ne 0 ]; then
    echo "基础依赖安装失败！"
    exit 1
fi

echo ""
echo "正在安装 MMDetection..."
pip install mmdet==3.3.0

if [ $? -ne 0 ]; then
    echo "MMDetection 安装失败！"
    exit 1
fi

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "正在验证安装..."
python -c "from mmengine.config import Config; print('✓ mmengine 安装成功')"
python -c "from mmdet.apis import init_detector; print('✓ mmdet 安装成功')"
python -c "from mmdet.utils import register_all_modules; print('✓ mmdet.utils 安装成功')"

echo ""
echo "验证完成！"


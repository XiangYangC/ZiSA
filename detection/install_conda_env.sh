#!/bin/bash
# Conda 环境依赖安装脚本（Linux/Mac）

echo "========================================"
echo "VMamba 检测任务 - Conda 环境安装"
echo "========================================"
echo

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令！"
    echo "请先安装 Anaconda 或 Miniconda"
    exit 1
fi

echo "步骤 1: 创建 conda 环境（如果不存在）"
echo "----------------------------------------"
if conda env list | grep -q "vmamba"; then
    echo "✓ vmamba 环境已存在"
else
    echo "正在创建 vmamba 环境..."
    conda create -n vmamba python=3.9 -y
    if [ $? -ne 0 ]; then
        echo "环境创建失败！"
        exit 1
    fi
    echo "✓ 环境创建成功"
fi

echo
echo "步骤 2: 激活环境"
echo "----------------------------------------"
echo "请运行: conda activate vmamba"
echo

echo "步骤 3: 安装 PyTorch（如果未安装）"
echo "----------------------------------------"
echo "请根据你的 CUDA 版本选择安装命令："
echo
echo "CUDA 11.8:"
echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
echo
echo "CUDA 12.1:"
echo "  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
echo
echo "CPU 版本:"
echo "  conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
echo

echo "步骤 4: 安装项目基础依赖"
echo "----------------------------------------"
echo "请运行: pip install -r requirements.txt"
echo

echo "步骤 5: 安装 Selective Scan"
echo "----------------------------------------"
echo "请运行: cd kernels/selective_scan && pip install ."
echo

echo "步骤 6: 安装检测依赖"
echo "----------------------------------------"
echo "请运行以下命令:"
echo "  pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex"
echo "  pip install mmdet==3.3.0"
echo

echo "========================================"
echo "脚本完成！"
echo "========================================"
echo
echo "完整的安装命令序列："
echo
echo "  conda activate vmamba"
echo "  pip install -r requirements.txt"
echo "  cd kernels/selective_scan && pip install ."
echo "  cd ../.."
echo "  pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex"
echo "  pip install mmdet==3.3.0"
echo


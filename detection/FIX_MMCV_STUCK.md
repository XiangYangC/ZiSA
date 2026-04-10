# 解决 MMCV 编译卡住问题

## 🚨 当前问题

MMCV 编译卡住了，需要停止并使用预编译版本。

## ✅ 快速解决方案

### 方法 1：使用快速安装脚本（最简单）

```bash
# 1. 停止当前卡住的进程（按 Ctrl+C）
# 2. 运行快速安装脚本
cd D:\Pythoncode\VMamba-main\VMamba-main\detection
python quick_install_mmcv.py
```

### 方法 2：手动安装（推荐）

```bash
# 1. 停止当前进程（Ctrl+C）

# 2. 卸载卡住的版本
pip uninstall mmcv mmcv-full -y

# 3. 安装预编译版本（根据你的 CUDA 版本选择）

# CUDA 11.8（最常见）
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# CUDA 12.1
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html

# CPU 版本
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

### 方法 3：一键命令（自动检测）

```bash
# 停止当前进程（Ctrl+C）
# 然后运行：
python install_mmcv_helper.py
```

## 🔍 如何查看 CUDA 版本

如果不确定 CUDA 版本，运行：

```bash
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## 📝 完整安装流程（避免卡住）

```bash
# 1. 激活 conda 环境
conda activate VMamba

# 2. 安装其他依赖（不包含 mmcv）
pip install mmengine==0.10.1 opencv-python-headless ftfy regex

# 3. 使用预编译版本安装 mmcv（快速，不会卡住）
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 4. 安装 MMDetection
pip install mmdet==3.3.0
```

## ✅ 推荐操作（立即执行）

**立即执行**：

```bash
# 1. 按 Ctrl+C 停止当前编译
# 2. 运行以下命令（一键解决）
pip uninstall mmcv mmcv-full -y && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

然后继续安装其他依赖：

```bash
pip install mmdet==3.3.0
```


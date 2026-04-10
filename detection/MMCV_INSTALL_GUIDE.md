# MMCV 安装优化指南

## ⚠️ 当前状态

你正在从源码编译 `mmcv`，这可能需要 **5-15 分钟**，取决于你的 CPU 性能。

## 🚀 推荐方案：使用预编译版本（更快）

### 方法 1：根据 CUDA 版本安装预编译版本

**停止当前安装**（按 `Ctrl+C`），然后使用以下命令：

#### CUDA 11.8:
```bash
pip uninstall mmcv -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

#### CUDA 12.1:
```bash
pip uninstall mmcv -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
```

#### CPU 版本:
```bash
pip uninstall mmcv -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

### 方法 2：检查你的 PyTorch 和 CUDA 版本

```bash
# 查看 PyTorch 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 查看 CUDA 版本
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## 📋 安装命令模板

根据你的环境选择对应的命令：

```bash
# 1. 卸载当前正在编译的版本
pip uninstall mmcv -y

# 2. 安装预编译版本（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# CUDA 12.1
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html

# CPU only
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

## ⏳ 如果继续编译

如果你选择继续等待编译完成：

1. **编译时间**: 通常需要 5-15 分钟
2. **CPU 使用**: 编译时会占用较高 CPU
3. **内存需求**: 至少需要 4GB 可用内存
4. **不要中断**: 中断后需要重新开始

## 🔍 检查编译进度

编译过程中会显示类似信息：
```
Building wheel for mmcv (setup.py) ... \
```

编译完成后会显示：
```
Successfully built mmcv
Installing collected packages: mmcv
Successfully installed mmcv-2.1.0
```

## ⚠️ 如果编译失败

### 常见错误和解决方案

#### 1. 缺少编译工具（Linux/Mac）
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum install gcc gcc-c++ make

# Mac
xcode-select --install
```

#### 2. 缺少 CUDA 开发工具
```bash
# 确保安装了 CUDA Toolkit
# 检查 CUDA 版本
nvcc --version
```

#### 3. 内存不足
使用预编译版本可以避免编译，减少内存消耗。

## ✅ 验证安装

安装完成后验证：

```bash
python -c "import mmcv; print(f'✓ mmcv {mmcv.__version__} 安装成功')"
```

## 📝 完整安装脚本（使用预编译版本）

```bash
# 激活 conda 环境
conda activate VMamba

# 安装其他依赖（跳过 mmcv）
pip install mmengine==0.10.1 opencv-python-headless ftfy regex

# 安装 MMDetection（会自动安装 mmcv 的依赖）
pip install mmdet==3.3.0 --no-deps

# 手动安装预编译的 mmcv（根据你的 CUDA 版本）
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 安装 MMDetection 的其他依赖
pip install mmdet==3.3.0
```

## 🎯 建议

**强烈建议使用预编译版本**，因为：
- ✅ 安装速度快（几秒钟 vs 几分钟）
- ✅ 不需要编译工具
- ✅ 更稳定可靠
- ✅ 减少内存占用


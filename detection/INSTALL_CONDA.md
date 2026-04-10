# Conda 环境安装指南

## 📦 使用 Conda 安装检测依赖

### 方法 1：使用脚本（推荐）

#### Windows:
```bash
# 1. 运行安装脚本
install_conda_env.bat

# 2. 按照脚本提示执行后续步骤
```

#### Linux/Mac:
```bash
# 1. 添加执行权限
chmod +x install_conda_env.sh

# 2. 运行安装脚本
./install_conda_env.sh

# 3. 按照脚本提示执行后续步骤
```

### 方法 2：手动安装（详细步骤）

#### 步骤 1: 创建并激活 conda 环境

```bash
# 创建环境（如果不存在）
conda create -n vmamba python=3.9 -y

# 激活环境
conda activate vmamba
```

#### 步骤 2: 安装 PyTorch

根据你的 CUDA 版本选择命令：

**CUDA 11.8:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CUDA 12.1:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU 版本:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

#### 步骤 3: 安装项目基础依赖

```bash
# 切换到项目根目录
cd D:\Pythoncode\VMamba-main\VMamba-main

# 安装基础依赖
pip install -r requirements.txt
```

#### 步骤 4: 安装 Selective Scan（必需）

```bash
# 进入 selective_scan 目录
cd kernels/selective_scan

# 安装 selective_scan
pip install .

# 返回项目根目录
cd ../..
```

#### 步骤 5: 安装检测依赖

```bash
# 安装基础检测依赖
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex

# 安装 MMDetection
pip install mmdet==3.3.0
```

## 🚀 一键安装命令（完整流程）

在 **conda 环境激活后**，依次运行：

```bash
# 1. 基础依赖
pip install -r requirements.txt

# 2. Selective Scan
cd kernels/selective_scan
pip install .
cd ../..

# 3. 检测依赖
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0
```

## ✅ 验证安装

安装完成后，验证依赖：

```bash
# 确保在 vmamba 环境中
conda activate vmamba

# 验证 PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA 可用: {torch.cuda.is_available()}')"

# 验证 MMEngine
python -c "from mmengine.config import Config; print('✓ mmengine 安装成功')"

# 验证 MMDetection
python -c "from mmdet.apis import init_detector; print('✓ mmdet 安装成功')"
```

## 📋 环境信息

- **环境名称**: `vmamba`
- **Python 版本**: 3.9（推荐）
- **PyTorch**: 根据 CUDA 版本选择
- **MMEngine**: 0.10.1
- **MMCV**: 2.1.0
- **MMDetection**: 3.3.0

## 🔧 环境管理

### 查看所有环境
```bash
conda env list
```

### 激活环境
```bash
conda activate vmamba
```

### 退出环境
```bash
conda deactivate
```

### 删除环境（如果不需要）
```bash
conda env remove -n vmamba
```

## ⚠️ 常见问题

### 1. conda 命令未找到

**Windows**: 确保安装了 Anaconda 或 Miniconda，并且添加到 PATH。

**Linux/Mac**: 
```bash
# 初始化 conda（如果未初始化）
conda init bash  # 或 conda init zsh
# 重启终端后生效
```

### 2. mmcv 安装失败

如果 `mmcv==2.1.0` 安装失败，尝试：

```bash
# 根据 CUDA 版本选择
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### 3. selective_scan 编译失败

确保已安装：
- Visual Studio Build Tools (Windows)
- GCC/G++ (Linux/Mac)
- CUDA Toolkit（如果使用 GPU）

### 4. 环境已存在但没有依赖

如果 `vmamba` 环境已存在但缺少依赖：

```bash
# 激活环境
conda activate vmamba

# 直接安装缺失的依赖
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0
```

## 📝 完整安装流程总结

```bash
# 1. 创建环境
conda create -n vmamba python=3.9 -y

# 2. 激活环境
conda activate vmamba

# 3. 安装 PyTorch（根据 CUDA 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装项目依赖
cd D:\Pythoncode\VMamba-main\VMamba-main
pip install -r requirements.txt

# 5. 安装 Selective Scan
cd kernels/selective_scan
pip install .
cd ../..

# 6. 安装检测依赖
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0

# 7. 验证
python -c "from mmengine.config import Config; from mmdet.apis import init_detector; print('✓ 所有依赖安装成功')"
```

## 🎯 下一步

安装完成后：

1. 确保在 `vmamba` 环境中运行检测脚本
2. 更新 IDE 的 Python 解释器指向 conda 环境
3. 运行 `detect_windfarm_yolo.py` 开始检测

## 💡 IDE 配置

### VS Code
1. 按 `Ctrl+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择 `vmamba` 环境的 Python 解释器
   - 通常在: `C:\Users\你的用户名\anaconda3\envs\vmamba\python.exe`

### PyCharm
1. File → Settings → Project → Python Interpreter
2. 添加新解释器 → Conda Environment
3. 选择 `vmamba` 环境


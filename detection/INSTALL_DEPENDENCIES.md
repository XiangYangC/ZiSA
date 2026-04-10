# 检测依赖安装指南

## 🔴 问题：导入语句显示红色

如果 IDE 中 `mmengine`、`mmdet` 等导入语句显示红色，说明这些依赖包未安装。

## ✅ 解决方案

### 方法 1：使用检查脚本（推荐）

运行检查脚本，它会自动检测并安装缺失的依赖：

```bash
cd D:\Pythoncode\VMamba-main\VMamba-main\detection
python check_dependencies.py
```

### 方法 2：手动安装

根据 VMamba 官方文档，运行以下命令安装检测任务所需的依赖：

#### 步骤 1：安装基础依赖

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
```

#### 步骤 2：安装 MMDetection

```bash
pip install mmdet==3.3.0
```

（可选）如果需要分割功能：
```bash
pip install mmsegmentation==1.2.2 mmpretrain==1.2.0
```

### 方法 3：完整安装（从项目根目录）

```bash
# 1. 安装基础依赖
cd D:\Pythoncode\VMamba-main\VMamba-main
pip install -r requirements.txt

# 2. 安装 selective_scan（必需）
cd kernels/selective_scan
pip install .

# 3. 安装检测依赖
cd ../..
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0
```

## 📋 依赖包说明

### 必需依赖

- **mmengine**: MMDetection 的底层引擎
- **mmcv**: OpenMMLab 计算机视觉基础库
- **mmdet**: MMDetection 目标检测框架
- **opencv-python-headless**: OpenCV（无 GUI 版本）
- **ftfy**: 文本修复工具
- **regex**: 正则表达式库

### 版本要求

根据 VMamba 官方文档，推荐版本：
- `mmengine==0.10.1`
- `mmcv==2.1.0`
- `mmdet==3.3.0`

## 🔍 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "from mmengine.config import Config; print('✓ mmengine 安装成功')"
python -c "from mmdet.apis import init_detector; print('✓ mmdet 安装成功')"
```

或者运行检查脚本：

```bash
python check_dependencies.py
```

## ⚠️ 常见问题

### 1. mmcv 安装失败

如果 `mmcv==2.1.0` 安装失败，尝试：

```bash
# 先安装预编译版本
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 或者根据你的 CUDA 版本选择
# CUDA 11.8: https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
# CUDA 12.1: https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
```

### 2. 虚拟环境问题

确保在正确的虚拟环境中安装：

```bash
# 激活 conda 环境
conda activate vmamba

# 或激活 venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. IDE 无法识别

如果安装后 IDE 仍然显示红色：

1. **VS Code**: 
   - 按 `Ctrl+Shift+P`
   - 输入 "Python: Select Interpreter"
   - 选择正确的 Python 解释器（包含已安装依赖的环境）

2. **PyCharm**:
   - File → Settings → Project → Python Interpreter
   - 确保选择了正确的解释器

3. **重启 IDE**: 安装依赖后重启 IDE

## 📝 完整安装命令（一键复制）

```bash
# Windows PowerShell
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0

# 验证
python -c "from mmengine.config import Config; from mmdet.apis import init_detector; print('✓ 所有依赖安装成功')"
```

## 🎯 下一步

安装完成后：

1. 重启 IDE 使导入生效
2. 运行 `detect_windfarm_yolo.py` 开始检测
3. 如果仍有问题，检查 Python 解释器路径是否正确


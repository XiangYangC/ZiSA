# 解决常见问题指南

## ✅ 已修复的问题

### 1. ❌ 警告: 无法导入 model 模块: No module named 'mmseg'

**问题**: `mmseg` (mmsegmentation) 是可选依赖，仅用于分割任务。检测任务不需要它。

**解决方案**: 
- ✅ **已修复**: `model.py` 和 `detect_windfarm_yolo.py` 已更新，会自动忽略 `mmseg` 相关的导入错误
- ✅ **不影响使用**: 检测任务不需要 `mmseg`，可以安全忽略这个警告

**如果想消除警告**（可选）:
```bash
pip install mmsegmentation==1.2.2
```

---

### 2. ❌ 错误: 模型权重不存在

**问题**: 脚本需要模型权重文件才能进行检测。

**解决方案**:

#### 方法 1: 自动查找（推荐）
脚本现在会自动查找以下位置的权重文件：
- `work_dirs/windfarm_yolo_head/latest.pth`
- `work_dirs/windfarm_yolo_head/best.pth`
- `work_dirs/windfarm_yolo_head/epoch_1.pth`
- `work_dirs/windfarm_yolo_head/epoch_2.pth`

#### 方法 2: 手动指定
编辑 `detect_windfarm_yolo.py`，修改以下行：

```python
# Linux 路径示例
CHECKPOINT = "/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/windfarm_yolo_head/latest.pth"

# Windows 路径示例
CHECKPOINT = r"D:\Pythoncode\VMamba-main\VMamba-main\detection\work_dirs\windfarm_yolo_head\latest.pth"
```

#### 方法 3: 命令行参数
```bash
python detect_windfarm_yolo.py --checkpoint /path/to/checkpoint.pth
```

---

## 📋 完整使用流程

### 步骤 1: 训练模型（如果还没有）

```bash
cd /home/cruiy/code/python/VMamba-main/VMamba-main/detection

# 训练模型
python tools/train.py \
    configs/windfarm/vmamba_yolo_head_windfarm.py \
    --work-dir work_dirs/windfarm_yolo_head
```

训练完成后，权重文件会保存在 `work_dirs/windfarm_yolo_head/latest.pth`

### 步骤 2: 配置检测脚本

编辑 `detect_windfarm_yolo.py`：

```python
# 1. 设置模型权重路径（如果脚本没有自动找到）
CHECKPOINT = "/path/to/work_dirs/windfarm_yolo_head/latest.pth"

# 2. 设置检测模式
MODE = 'single'  # 单张图片检测
# 或
MODE = 'dataset'  # 数据集批量检测

# 3. 如果单张图片模式，设置图片路径
SINGLE_IMAGE_PATH = "/path/to/image.jpg"
```

### 步骤 3: 运行检测

```bash
python detect_windfarm_yolo.py
```

或使用命令行参数：

```bash
# 单张图片检测
python detect_windfarm_yolo.py \
    --checkpoint work_dirs/windfarm_yolo_head/latest.pth \
    --mode single \
    --image /path/to/image.jpg

# 数据集检测
python detect_windfarm_yolo.py \
    --checkpoint work_dirs/windfarm_yolo_head/latest.pth \
    --mode dataset \
    --split val
```

---

## 🔍 检查权重文件是否存在

```bash
# 检查权重文件
ls -lh work_dirs/windfarm_yolo_head/*.pth

# 或查看最新文件
ls -lht work_dirs/windfarm_yolo_head/ | head -5
```

---

## ⚠️ 常见问题

### Q: 如果还没有训练模型怎么办？

A: 你需要先训练模型。运行：

```bash
python tools/train.py \
    configs/windfarm/vmamba_yolo_head_windfarm.py \
    --work-dir work_dirs/windfarm_yolo_head
```

### Q: 可以使用其他模型的权重吗？

A: 可以，但需要确保：
1. 模型架构匹配（VMamba + YOLO 检测头）
2. 类别数匹配（7个类别）
3. 权重格式正确（.pth 格式）

### Q: mmseg 警告可以忽略吗？

A: 可以！检测任务不需要 `mmseg`。这个警告已经被修复，不会再显示。

---

## ✅ 验证环境

运行以下命令验证所有依赖都正确安装：

```bash
python -c "from mmengine.config import Config; from mmdet.apis import init_detector; print('✓ 所有依赖正常')"
```


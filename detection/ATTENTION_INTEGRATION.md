# 注意力机制集成说明

## 📋 概述

已将自定义的 `ZoomInSelfAttention` 注意力机制模块化集成到检测框架中。该注意力机制可以增强 FPN 输出的特征，提升检测性能。

## 🏗️ 模块结构

```
detection/
├── attention_modules.py          # 注意力机制模块（ZoomInSelfAttention）
├── enhanced_fpn.py              # 增强的 FPN（支持注意力）
├── model.py                     # 模块注册（自动导入）
└── configs/windfarm/
    ├── mask_rcnn_vssm_windfarm.py                    # 原始配置（不使用注意力）
    └── faster_rcnn_vssm_windfarm_with_attention.py  # 带注意力的配置
```

## 🔧 使用方法

### 方法 1：使用带注意力的配置文件（推荐）

直接使用已配置好的配置文件：

```bash
python detection/train_windfarm_detection.py
```

然后在 `train_windfarm_detection.py` 中修改配置文件路径：

```python
CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "faster_rcnn_vssm_windfarm_with_attention.py")
```

### 方法 2：在现有配置中启用注意力

在配置文件中修改 `neck` 配置：

```python
# 原始配置（不使用注意力）
neck=dict(
    type='FPN',
    in_channels=[96, 192, 384, 768],
    out_channels=256,
    num_outs=5,
)

# 启用注意力（替换为 EnhancedFPN）
neck=dict(
    type='EnhancedFPN',  # 使用增强的 FPN
    in_channels=[96, 192, 384, 768],
    out_channels=256,
    num_outs=5,
    # 注意力配置
    attention_cfg=dict(
        type='ZoomInSelfAttention',
        num_heads=4,        # 多头注意力头数
        reduction=16,      # SE 通道注意力压缩比
        kv_downsample=1,   # K/V 降采样倍数（1表示不降采样）
    ),
    # attention_indices=[0, 1, 2, 3, 4],  # 在所有输出层应用注意力（默认）
)
```

### 方法 3：仅在特定层应用注意力

如果只想在部分 FPN 输出层应用注意力（例如只在 P3, P4, P5 上）：

```python
neck=dict(
    type='EnhancedFPN',
    in_channels=[96, 192, 384, 768],
    out_channels=256,
    num_outs=5,
    attention_cfg=dict(
        type='ZoomInSelfAttention',
        num_heads=4,
        reduction=16,
        kv_downsample=1,
    ),
    attention_indices=[1, 2, 3],  # 只在 P3, P4, P5 上应用（索引从0开始）
)
```

## ⚙️ 注意力参数说明

### `ZoomInSelfAttention` 参数

- **`in_channels`** (int): 输入通道数（自动从 FPN 的 `out_channels` 获取）
- **`num_heads`** (int, 默认=4): 多头注意力的头数
  - 必须能被 `in_channels` 整除
  - 建议值：4, 8, 16
- **`reduction`** (int, 默认=16): SE 通道注意力的压缩比
  - 值越大，参数量越少，但可能影响性能
  - 建议值：8, 16, 32
- **`kv_downsample`** (int, 默认=1): Key/Value 的降采样倍数
  - `1`: 不降采样（计算量最大，效果最好）
  - `2`: 降采样到 1/2（计算量减少约 75%）
  - `4`: 降采样到 1/4（计算量减少约 94%）
  - 建议：显存充足时用 1，显存紧张时用 2 或 4

### 性能与显存权衡

| kv_downsample | 计算复杂度 | 显存占用 | 推荐场景 |
|--------------|----------|---------|---------|
| 1            | O(HW²)   | 高      | 显存充足（24GB+） |
| 2            | O(HW²/4) | 中      | 显存中等（16GB） |
| 4            | O(HW²/16)| 低      | 显存紧张（8GB） |

## 📊 模块化设计

### 1. **完全模块化**
- ✅ 不修改原有 VMamba 代码
- ✅ 不修改 MMDetection 框架代码
- ✅ 通过配置文件灵活启用/禁用

### 2. **向后兼容**
- ✅ `EnhancedFPN` 完全兼容标准 `FPN`
- ✅ 不配置 `attention_cfg` 时，行为与标准 FPN 完全相同
- ✅ 可以随时切换回标准 FPN

### 3. **灵活配置**
- ✅ 可以选择在哪些层应用注意力
- ✅ 可以调整注意力参数
- ✅ 可以调整 K/V 降采样以平衡性能和显存

## 🔍 工作原理

```
输入图像
    ↓
VMamba Backbone (特征提取)
    ↓
FPN Neck (多尺度特征融合)
    ↓
EnhancedFPN (应用注意力) ← 在这里增强特征
    ├─ P2 (应用 ZoomInSelfAttention)
    ├─ P3 (应用 ZoomInSelfAttention)
    ├─ P4 (应用 ZoomInSelfAttention)
    ├─ P5 (应用 ZoomInSelfAttention)
    └─ P6 (应用 ZoomInSelfAttention)
    ↓
RPN Head (区域提议)
    ↓
ROI Head (检测和分类)
    ↓
检测结果
```

## 📝 注意事项

1. **显存占用**：使用注意力会增加显存占用，建议：
   - 使用注意力时，将 `batch_size` 适当减小（例如从 4 减到 2）
   - 如果显存不足，可以设置 `kv_downsample=2` 或 `4`

2. **训练时间**：注意力机制会增加训练时间，但通常能提升检测精度

3. **参数数量**：注意力模块会增加模型参数量，但通常增加量不大（约 5-10%）

4. **兼容性**：
   - 可以随时在配置文件中切换是否使用注意力
   - 不使用注意力时，`EnhancedFPN` 的行为与标准 `FPN` 完全相同

## 🚀 快速开始

1. **使用带注意力的配置训练**：
   ```bash
   # 修改 train_windfarm_detection.py 中的配置文件路径
   CFG_PATH = "configs/windfarm/faster_rcnn_vssm_windfarm_with_attention.py"
   
   # 运行训练
   python detection/train_windfarm_detection.py
   ```

2. **对比实验**：
   - 使用 `mask_rcnn_vssm_windfarm.py`（无注意力）训练一个模型
   - 使用 `faster_rcnn_vssm_windfarm_with_attention.py`（有注意力）训练另一个模型
   - 对比两个模型的检测精度

## 📚 相关文件

- `attention_modules.py`: 注意力机制实现
- `enhanced_fpn.py`: 增强的 FPN 实现
- `model.py`: 模块注册
- `configs/windfarm/faster_rcnn_vssm_windfarm_with_attention.py`: 带注意力的配置示例


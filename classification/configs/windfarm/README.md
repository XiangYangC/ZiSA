# Windfarm 数据集训练说明

## 数据集结构

Windfarm 数据集应该按照以下结构组织：

```
Windfarm_VMamba/
├── train/
│   ├── burning/      (230 张图片)
│   ├── crack/        (362 张图片)
│   ├── deformity/    (465 张图片)
│   ├── dirt/         (1353 张图片)
│   ├── oil/          (92 张图片)
│   ├── peeling/      (824 张图片)
│   └── rusty/        (247 张图片)
└── val/
    ├── burning/      (31 张图片)
    ├── crack/        (34 张图片)
    ├── deformity/    (65 张图片)
    ├── dirt/         (175 张图片)
    ├── oil/          (10 张图片)
    ├── peeling/      (92 张图片)
    └── rusty/        (39 张图片)
```

共 7 个类别：burning（燃烧）、crack（裂缝）、deformity（变形）、dirt（污垢）、oil（油污）、peeling（剥落）、rusty（生锈）

## 训练命令

### 单 GPU 训练

```bash
cd /home/cruiy/code/python/VMamba-main/VMamba-main/classification

python main.py \
  --cfg configs/windfarm/vmambav2_tiny_224_windfarm.yaml \
  --data-path /path/to/Windfarm_VMamba \
  --batch-size 32 \
  --output ./output/windfarm
```

### 多 GPU 训练（推荐）

```bash
cd /home/cruiy/code/python/VMamba-main/VMamba-main/classification

python -m torch.distributed.launch \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr="127.0.0.1" \
  --master_port=29501 \
  main.py \
  --cfg configs/windfarm/vmambav2_tiny_224_windfarm.yaml \
  --data-path /path/to/Windfarm_VMamba \
  --batch-size 32 \
  --output ./output/windfarm
```

### 使用预训练权重进行微调

```bash
python -m torch.distributed.launch \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr="127.0.0.1" \
  --master_port=29501 \
  main.py \
  --cfg configs/windfarm/vmambav2_tiny_224_windfarm.yaml \
  --data-path /path/to/Windfarm_VMamba \
  --batch-size 32 \
  --pretrained /path/to/pretrained/checkpoint.pth \
  --output ./output/windfarm
```

### 评估模型

```bash
python -m torch.distributed.launch \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr="127.0.0.1" \
  --master_port=29501 \
  main.py \
  --cfg configs/windfarm/vmambav2_tiny_224_windfarm.yaml \
  --data-path /path/to/Windfarm_VMamba \
  --batch-size 32 \
  --eval \
  --pretrained /path/to/checkpoint.pth
```

## 配置文件说明

配置文件位于：`configs/windfarm/vmambav2_tiny_224_windfarm.yaml`

主要配置项：
- `DATA.DATASET: windfarm` - 数据集类型
- `DATA.IMG_SIZE: 224` - 输入图像尺寸
- `DATA.BATCH_SIZE: 32` - 批次大小（可根据 GPU 内存调整）
- `MODEL.NUM_CLASSES: 7` - 类别数量（自动设置）
- `TRAIN.EPOCHS: 100` - 训练轮数
- `TRAIN.BASE_LR: 1e-3` - 基础学习率

## 参数调整建议

1. **批次大小**：根据 GPU 显存调整 `BATCH_SIZE`
   - 8GB GPU: batch_size=16-32
   - 16GB GPU: batch_size=32-64
   - 24GB+ GPU: batch_size=64-128

2. **学习率**：小数据集建议使用较小学习率
   - 从头训练：1e-3 到 5e-4
   - 微调：1e-4 到 5e-5

3. **训练轮数**：根据验证集性能调整
   - 建议：50-100 epochs

4. **数据增强**：可在配置文件中调整
   - `AUG.MIXUP`: 0.8
   - `AUG.CUTMIX: 1.0`

## 注意事项

1. 确保数据集路径正确（`--data-path`）
2. 多 GPU 训练时，实际批次大小 = batch_size × GPU数量
3. 检查点会自动保存在 `--output` 指定的目录
4. 训练日志会保存在 `output` 目录下的 `log.txt`


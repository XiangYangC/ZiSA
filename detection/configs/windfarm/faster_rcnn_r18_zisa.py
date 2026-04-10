# detection/configs/windfarm/faster_rcnn_r18_zisa.py

import sys
import os

# ========================================================
# 🔧 最终修复：正确导入 + 防止日志报错
# ========================================================
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
det_path = os.path.join(cwd, 'detection')
if os.path.exists(det_path) and det_path not in sys.path:
    sys.path.insert(0, det_path)

# 1. 寻找注册表
try:
    from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
except ImportError:
    try:
        from mmcv.cnn import PLUGIN_LAYERS
    except ImportError:
        pass

# 2. 从 attention_modules.py 导入并注册
try:
    # 尝试直接从 detection 包导入
    from detection.attention_modules import ZoomInSelfAttention

    # 强制注册
    if 'PLUGIN_LAYERS' in locals():
        PLUGIN_LAYERS.register_module(module=ZoomInSelfAttention, force=True)
        print(">>> [Success] ZiSA (from attention_modules) 已成功注册！")

    # 💀【关键】删除变量，防止 yapf 格式化报错 💀
    del ZoomInSelfAttention

except ImportError:
    # 备用：如果直接在 detection 目录下运行
    try:
        # from attention_modules import ZoomInSelfAttention
        from ...attention_modules import ZoomInSelfAttention
        if 'PLUGIN_LAYERS' in locals():
            PLUGIN_LAYERS.register_module(module=ZoomInSelfAttention, force=True)
            print(">>> [Success] ZiSA (local import) 已成功注册！")

        # 💀【关键】删除变量 💀
        del ZoomInSelfAttention

    except Exception as e:
        print(f">>> [Warning] 自动注册失败: {e}")

# ========================================================
# 🚀 你的高配实验设置
# ========================================================

_base_ = './faster_rcnn_r50_windfarm.py'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),

        # 插入插件
        plugins=[
            dict(
                cfg=dict(type='ZoomInSelfAttention'),
                stages=(False, True, True, True),
                position='after_conv2'
            )
        ]
    )
)

# 保持高 Batch Size (双卡共16)
train_dataloader = dict(
    batch_size=8,
    num_workers=4
)

# 这一行其实可以注释掉了，因为我们上面已经手动导入了
# custom_imports = dict(imports=['detection.attention_modules'], allow_failed_imports=True)
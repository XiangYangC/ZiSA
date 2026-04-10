"""
Faster R-CNN (ResNet-50) 从零训练配置 - 终极修复版
修复点：添加 _delete_=True 彻底阻断配置继承，防止 checkpoint 参数残留
"""
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. 数据集设置
data_root = '/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/'

metainfo = dict(
    classes=('burning', 'crack', 'deformity', 'dirt', 'oil', 'peeling', 'rusty'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100)]
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        metainfo=metainfo,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json', metric='bbox')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json', metric='bbox')

# 2. 模型配置
model = dict(
    type='FasterRCNN',
    backbone=dict(
        frozen_stages=-1,
        norm_eval=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),

        # 【关键修改】添加 _delete_=True，强制覆盖掉父配置中的 checkpoint
        init_cfg=dict(
            _delete_=True,
            type='Kaiming',
            layer='Conv2d',
            a=0,
            distribution='normal',
            mode='fan_out',
            nonlinearity='relu'
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
        )
    )
)

# 3. 训练策略
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)

# 4. 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

# 5. 学习率调度
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=300, by_epoch=True, milestones=[200, 260], gamma=0.1),
]

load_from = None
work_dir = './work_dirs/faster_rcnn_r50_scratch_bs8'
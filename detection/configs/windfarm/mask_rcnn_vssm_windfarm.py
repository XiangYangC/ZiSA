_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 数据集配置
data_root = '/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/'

# 类别信息（7个缺陷类别）
metainfo = dict(
    classes=('burning', 'crack', 'deformity', 'dirt', 'oil', 'peeling', 'rusty'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), 
             (106, 0, 228), (0, 60, 100), (0, 80, 100)]
)

train_dataloader = dict(
    batch_size=8,  # 根据GPU显存调整，8GB用2，16GB用4，24GB+用8
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
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
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        metainfo=metainfo,
        test_mode=True,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/images/'),
        metainfo=metainfo,
        test_mode=True,
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
)

# 模型配置（使用 VMamba-Tiny 作为 backbone）
model = dict(
    type='FasterRCNN',  # 使用 Faster R-CNN（不需要 mask 标注）
    backbone=dict(
        _delete_=True,  # 删除基础配置中的 backbone
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="",  # 可以使用分类任务的预训练权重
        # VMamba-Tiny 配置
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        norm_layer="ln2d",
    ),
    # FPN Neck 配置（VMamba-Tiny 的输出通道: [96, 192, 384, 768]）
    neck=dict(
        in_channels=[96, 192, 384, 768],  # 对应 VMamba-Tiny 的4个stage输出
    ),
    # RPN Head（区域提议网络）- 继承自基础配置，用于生成候选框
    # rpn_head 已在基础配置中定义，这里不需要修改
    
    # ROI Head（只包含 BBox Head，不包含 Mask Head）
    roi_head=dict(
        # BBox Head（边界框回归和分类头）
        bbox_head=dict(
            num_classes=7,  # 7个缺陷类别：burning, crack, deformity, dirt, oil, peeling, rusty
        ),
    ),
)

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置（覆盖基础配置中的 SGD，使用 AdamW）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,  # 删除基础配置中的 optimizer
        type='AdamW',
        lr=0.0001,  # 初始学习率
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'backbone': dict(lr_mult=0.1)},  # backbone使用较小的学习率
    ),
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,  # warmup
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 工作目录
work_dir = './work_dirs/windfarm_detection'


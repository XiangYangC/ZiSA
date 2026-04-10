custom_imports = dict(
    imports=[
        'ZISA_ablation.zisa_wrapper',
        'ZISA_ablation.ablation_fpn',
    ],
    allow_failed_imports=False,
)

_base_ = '../../configs/windfarm/faster_rcnn_r50_windfarm.py'

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

model = dict(
    neck=dict(
        _delete_=True,
        type='ZiSAAblationFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        attention_cfg=None,
        attention_indices=[0, 1, 2],
    )
)

work_dir = './detection/ZISA_ablation/work_dirs/base'

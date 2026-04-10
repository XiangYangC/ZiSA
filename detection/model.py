import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET

# mmseg 是可选的（仅用于分割任务）
try:
    from mmseg.registry import MODELS as MODELS_MMSEG
    HAS_MMSEG = True
except ImportError:
    MODELS_MMSEG = None
    HAS_MMSEG = False

# ========== 导入自定义注意力模块和增强 FPN ==========
# 自动导入，使其注册到 MMDetection
try:
    # 尝试相对导入（当作为包的一部分导入时）
    try:
        from .attention_modules import ZoomInSelfAttention
        from .enhanced_fpn import EnhancedFPN
    except ImportError:
        # 如果相对导入失败，尝试绝对导入（当直接运行时）
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from attention_modules import ZoomInSelfAttention
        from enhanced_fpn import EnhancedFPN
    # 模块已自动注册到 MMDetection
except ImportError as e:
    import warnings
    warnings.warn(f"无法导入自定义注意力模块: {e}。如果需要使用注意力机制，请确保相关文件存在。")
# ==========================================

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

# 注册到 MMDetection（必需）
# 如果安装了 mmsegmentation，也注册到 MMSegmentation（可选）
if HAS_MMSEG:
    @MODELS_MMSEG.register_module()
    @MODELS_MMDET.register_module()
    class MM_VSSM(BaseModule, Backbone_VSSM):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            Backbone_VSSM.__init__(self, *args, **kwargs)
else:
    # mmsegmentation 未安装，只注册到 MMDetection（不影响检测任务）
    @MODELS_MMDET.register_module()
    class MM_VSSM(BaseModule, Backbone_VSSM):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            Backbone_VSSM.__init__(self, *args, **kwargs)




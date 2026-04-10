"""
增强的 FPN Neck，支持在输出特征上应用注意力机制
保持与标准 FPN 的完全兼容性，可以无缝替换
"""
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.necks import FPN


@MODELS.register_module()
class EnhancedFPN(FPN):
    """
    增强的 FPN，支持在输出特征上应用注意力机制
    
    使用方式：
    1. 不启用注意力（默认行为，与标准 FPN 完全相同）：
       neck=dict(type='EnhancedFPN', in_channels=[...], ...)
    
    2. 启用注意力（在所有输出特征上应用）：
       neck=dict(
           type='EnhancedFPN',
           in_channels=[...],
           attention_cfg=dict(
               type='ZoomInSelfAttention',
               num_heads=4,
               reduction=16,
               kv_downsample=1,
           ),
           ...
       )
    
    3. 仅在特定层启用注意力：
       neck=dict(
           type='EnhancedFPN',
           in_channels=[...],
           attention_cfg=dict(
               type='ZoomInSelfAttention',
               num_heads=4,
               reduction=16,
               kv_downsample=1,
           ),
           attention_indices=[0, 1, 2],  # 只在 P3, P4, P5 上应用
           ...
       )
    """

    def __init__(
        self,
        attention_cfg: Optional[Dict] = None,
        attention_indices: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Args:
            attention_cfg: 注意力模块的配置字典，如果为 None 则不使用注意力
            attention_indices: 在哪些输出层应用注意力，None 表示所有层都应用
            **kwargs: FPN 的其他参数（in_channels, out_channels, num_outs 等）
        """
        super().__init__(**kwargs)
        
        self.attention_cfg = attention_cfg
        self.attention_indices = attention_indices
        
        # 如果配置了注意力，创建注意力模块
        if attention_cfg is not None:
            # 复制配置字典，避免修改原始配置
            attention_cfg = attention_cfg.copy()
            attention_type = attention_cfg.pop('type', 'ZoomInSelfAttention')
            attention_kwargs = attention_cfg.copy()
            
            # 根据输出通道数创建注意力模块
            # FPN 的输出通道数是 out_channels（默认 256）
            out_channels = kwargs.get('out_channels', 256)
            
            # 创建注意力模块列表（每个输出层一个）
            num_outs = kwargs.get('num_outs', 5)
            self.attention_modules = nn.ModuleList()
            
            for i in range(num_outs):
                if attention_indices is None or i in attention_indices:
                    # 导入注意力模块（使用绝对导入）
                    try:
                        from .attention_modules import ZoomInSelfAttention
                    except ImportError:
                        # 如果相对导入失败，尝试绝对导入
                        import sys
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        from attention_modules import ZoomInSelfAttention
                    
                    attention_module = ZoomInSelfAttention(
                        in_channels=out_channels,
                        **attention_kwargs
                    )
                    self.attention_modules.append(attention_module)
                else:
                    # 不使用注意力，使用 Identity
                    self.attention_modules.append(nn.Identity())
        else:
            # 不使用注意力
            self.attention_modules = None

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """
        前向传播
        
        Args:
            inputs: 来自 backbone 的多尺度特征列表
        
        Returns:
            增强后的多尺度特征列表
        """
        # 调用父类的 forward 方法获取标准 FPN 输出
        outputs = super().forward(inputs)
        
        # 如果配置了注意力，在输出特征上应用
        if self.attention_modules is not None:
            enhanced_outputs = []
            for i, feat in enumerate(outputs):
                if i < len(self.attention_modules):
                    enhanced_feat = self.attention_modules[i](feat)
                    enhanced_outputs.append(enhanced_feat)
                else:
                    enhanced_outputs.append(feat)
            return tuple(enhanced_outputs)
        
        return outputs


from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DETECTION_DIR = os.path.dirname(_CURRENT_DIR)
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from attention_modules import ZoomInSelfAttention


@MODELS.register_module()
class ZiSAWrapper(nn.Module):
    """External wrapper for ZiSA ablation without changing the original module."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        reduction: int = 16,
        kv_downsample: int = 1,
        use_zoom: bool = True,
        use_mhsa: bool = True,
        use_se: bool = True,
        use_gate: bool = True,
        local_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_zoom = use_zoom
        self.use_mhsa = use_mhsa
        self.use_se = use_se
        self.use_gate = use_gate

        self.zisa = ZoomInSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            reduction=reduction,
            kv_downsample=kv_downsample,
        )

        padding = local_kernel_size // 2
        self.local_fallback = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=local_kernel_size,
                padding=padding,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, '
            f'use_zoom={self.use_zoom}, '
            f'use_mhsa={self.use_mhsa}, '
            f'use_se={self.use_se}, '
            f'use_gate={self.use_gate}'
        )

    def _reshape_to_heads(self, tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, channels, _, _ = tensor.shape
        return tensor.view(
            batch_size,
            self.zisa.num_heads,
            self.zisa.dim_head,
            height * width,
        ).permute(0, 1, 3, 2)

    def _forward_mhsa(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        query = self.zisa.q_proj(x)
        key = self.zisa.k_proj(x)
        value = self.zisa.v_proj(x)

        if self.zisa.kv_pool is not None:
            key_small = self.zisa.kv_pool(key)
            value_small = self.zisa.kv_pool(value)
            key_height, key_width = key_small.shape[2:]
        else:
            key_small = key
            value_small = value
            key_height, key_width = height, width

        query_heads = self._reshape_to_heads(query, height, width)
        key_heads = self._reshape_to_heads(key_small, key_height, key_width)
        value_heads = self._reshape_to_heads(value_small, key_height, key_width)

        scale = 1.0 / math.sqrt(self.zisa.dim_head)
        attention = torch.softmax(
            torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale,
            dim=-1,
        )
        out = torch.matmul(attention, value_heads)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, height, width)
        return self.zisa.out_proj(out)

    def _forward_se(self, x: torch.Tensor) -> torch.Tensor:
        channel_stats = F.adaptive_avg_pool2d(x, 1)
        channel_stats = F.relu(self.zisa.se_fc1(channel_stats))
        channel_attn = torch.sigmoid(self.zisa.se_fc2(channel_stats))
        return x * channel_attn

    def _forward_gate(self, residual: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        residual_gap = F.adaptive_avg_pool2d(residual, 1)
        enhanced_gap = F.adaptive_avg_pool2d(enhanced, 1)
        gate_input = torch.cat([residual_gap, enhanced_gap], dim=1)
        gate = self.zisa.gate_fc(gate_input)
        return residual * (1 - gate) + enhanced * gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_zoom and self.use_mhsa and self.use_se and self.use_gate:
            return self.zisa(x)

        zoomed_x = x * self.zisa.heatmap_gen(x) if self.use_zoom else x

        if self.use_mhsa:
            attn_out = self._forward_mhsa(zoomed_x)
        else:
            attn_out = self.local_fallback(zoomed_x)

        enhanced = self._forward_se(attn_out) if self.use_se else attn_out
        fused_feature = zoomed_x + enhanced if self.use_zoom else x + enhanced

        if self.use_gate:
            return self._forward_gate(x, fused_feature)
        return x + fused_feature

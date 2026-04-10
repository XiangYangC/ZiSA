"""
自定义注意力机制模块
用于检测任务的注意力增强
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoomInSelfAttention(nn.Module):
    """
    Zoom-in + Multi-Head Self-Attention + Channel SE + Adaptive Gate 融合版

    特点:
    - 修正了 Q @ K^T / sqrt(d_k) 的注意力公式与维度处理（多头按常规模型实现）
    - 更强的 Zoom-in 热图生成器（小卷积瓶颈）
    - 可选 K/V 降采样（通过 stride 或 avg_pool）以降低 HW^2 复杂度
    - SE-like 通道注意力 + 基于 gap 的自适应门控用于融合
    - 输入/输出 shape 恒为 (B, C, H, W)，方便直接插入检测模块
    """

    def __init__(self, in_channels, num_heads=4, reduction=16, kv_downsample=1):
        """
        Args:
            in_channels (int): 输入通道 C（要求 C % num_heads == 0 最好）
            num_heads (int): 多头数
            reduction (int): 通道注意力中的压缩比（SE）
            kv_downsample (int): Key/Value 的降采样倍数（1 表示不降采样；>1 会在 K/V 上下采样）
        """
        super().__init__()
        assert in_channels % num_heads == 0, "建议 in_channels 能被 num_heads 整除以避免维度问题"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = in_channels // num_heads
        self.kv_downsample = kv_downsample

        # Q/K/V 投影（1x1 conv）
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 更强的 Zoom-in 热图生成器（小瓶颈 conv）
        self.heatmap_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 通道注意力（SE-like）
        self.se_fc1 = nn.Conv2d(in_channels, max(in_channels // reduction, 4), kernel_size=1)
        self.se_fc2 = nn.Conv2d(max(in_channels // reduction, 4), in_channels, kernel_size=1)

        # gate：用于融合原始输入与增强特征（输出 [B, C, 1, 1] 的 sigmoid gate）
        self.gate_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 可选的降采样算子（对 K 和 V）
        if kv_downsample > 1:
            # 使用平均池化后再映射，保持通道数不变
            self.kv_pool = nn.AvgPool2d(kernel_size=kv_downsample, stride=kv_downsample)
        else:
            self.kv_pool = None

        # 小型输出投影（可选）—— 这里不改变通道，保持兼容
        self.out_proj = nn.Identity()

        # 参数初始化（可选）
        self._init_weights()
        self.enable_vis = False
        self.vis_cache = {}

    def _init_weights(self):
        # 轻量初始化，Conv2d 的默认初始化通常可以接受
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _prepare_vis_map(self, tensor, spatial_size=None):
        """Convert a tensor to a single-channel CPU map for visualization."""
        if tensor is None:
            return None

        vis_tensor = tensor.detach()

        if vis_tensor.dim() == 4:
            if vis_tensor.shape[1] != 1:
                vis_tensor = vis_tensor.mean(dim=1, keepdim=True)
        elif vis_tensor.dim() == 3:
            vis_tensor = vis_tensor.mean(dim=1, keepdim=True).unsqueeze(-1)
        elif vis_tensor.dim() == 2:
            vis_tensor = vis_tensor.unsqueeze(1).unsqueeze(-1)
        else:
            return None

        if spatial_size is not None and vis_tensor.shape[-2:] != spatial_size:
            vis_tensor = F.interpolate(
                vis_tensor,
                size=spatial_size,
                mode='bilinear',
                align_corners=False,
            )

        return vis_tensor.cpu()

    def _prepare_attn_map(self, attn, spatial_size):
        """Aggregate attention weights into a 2D query-side heatmap."""
        if attn is None:
            return None

        attn_vis = attn.detach()

        if attn_vis.dim() >= 4:
            query_map = attn_vis.mean(dim=1).mean(dim=-1)
        elif attn_vis.dim() == 3:
            query_map = attn_vis.mean(dim=-1)
        elif attn_vis.dim() == 2:
            query_map = attn_vis
        else:
            return self._prepare_vis_map(attn_vis, spatial_size=spatial_size)

        if query_map.dim() == 1:
            query_map = query_map.unsqueeze(0)

        target_h, target_w = spatial_size
        target_len = target_h * target_w

        if query_map.shape[-1] != target_len:
            return self._prepare_vis_map(
                query_map.reshape(query_map.shape[0], 1, 1, -1),
                spatial_size=spatial_size,
            )

        query_map = query_map.reshape(query_map.shape[0], 1, target_h, target_w)
        return query_map.cpu()

    def _cache_visualizations(self, zoom_map=None, attn_map=None, out_feat=None, spatial_size=None):
        if not self.enable_vis:
            return

        self.vis_cache = {}
        self.vis_cache['zoom_map'] = self._prepare_vis_map(zoom_map, spatial_size=spatial_size)
        self.vis_cache['attn_map'] = self._prepare_attn_map(attn_map, spatial_size=spatial_size)
        self.vis_cache['out_feat'] = self._prepare_vis_map(out_feat, spatial_size=spatial_size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        Lq = H * W  # query length

        # -------------------------
        # Step1: Zoom-in 粗定位（生成热图并放大该区域）
        # -------------------------
        heatmap = self.heatmap_gen(x)                        # (B,1,H,W), [0,1]
        zoomed_x = x * heatmap                               # (B,C,H,W)

        # -------------------------
        # Step2: Q/K/V 投影与可选 K/V 降采样
        # -------------------------
        Q = self.q_proj(zoomed_x)                            # (B,C,H,W)
        K = self.k_proj(zoomed_x)                            # (B,C,H,W)
        V = self.v_proj(zoomed_x)                            # (B,C,H,W)

        # 如果需要降低 K/V 空间分辨率以减少计算
        if self.kv_pool is not None:
            K_small = self.kv_pool(K)                        # (B,C,Hk,Wk)
            V_small = self.kv_pool(V)                        # (B,C,Hk,Wk)
            Hk, Wk = K_small.shape[2], K_small.shape[3]
            Lk = Hk * Wk
        else:
            K_small = K
            V_small = V
            Lk = Lq
            Hk, Wk = H, W

        # reshape -> (B, heads, L, dim_head)
        def reshape_to_heads(tensor, Ht, Wt):
            B, Cc, _, _ = tensor.shape
            # (B, num_heads, dim_head, L) -> permute to (B, num_heads, L, dim_head)
            tensor = tensor.view(B, self.num_heads, self.dim_head, Ht * Wt).permute(0, 1, 3, 2)
            return tensor

        Qh = reshape_to_heads(Q, H, W)                       # (B, heads, Lq, dim_head)
        Kh = reshape_to_heads(K_small, Hk, Wk)               # (B, heads, Lk, dim_head)
        Vh = reshape_to_heads(V_small, Hk, Wk)               # (B, heads, Lk, dim_head)

        # -------------------------
        # Step3: Attention 计算（标准 Q @ K^T / sqrt(d_k)）
        #        attn: (B, heads, Lq, Lk)
        #        out:  (B, heads, Lq, dim_head)
        # -------------------------
        # 注意 scale 用 dim_head（每个 head 的维度）
        scale = 1.0 / math.sqrt(self.dim_head)
        # (B, heads, Lq, dim_head) @ (B, heads, dim_head, Lk) -> (B, heads, Lq, Lk)
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale
        attn = torch.softmax(attn_scores, dim=-1)

        out_h = torch.matmul(attn, Vh)                       # (B, heads, Lq, dim_head)

        # 合并 heads -> (B, C, H, W)
        out = out_h.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        out = self.out_proj(out)

        # -------------------------
        # Step4: 通道注意力（SE）在 out 上
        # -------------------------
        gap = F.adaptive_avg_pool2d(out, 1)                   # (B,C,1,1)
        ch = F.relu(self.se_fc1(gap))
        ch_attn = torch.sigmoid(self.se_fc2(ch))             # (B,C,1,1)
        out_ch = out * ch_attn                                # (B,C,H,W)

        # -------------------------
        # Step5: 自适应融合（基于原始 x 与增强特征的 gap 拼接做 gate）
        # -------------------------
        gap_x = F.adaptive_avg_pool2d(x, 1)                   # (B,C,1,1)
        gap_enh = F.adaptive_avg_pool2d(out_ch, 1)           # (B,C,1,1)
        gate_input = torch.cat([gap_x, gap_enh], dim=1)       # (B, 2C, 1, 1)
        gate = self.gate_fc(gate_input)                       # (B, C, 1, 1), in (0,1)

        fused = x * (1 - gate) + (zoomed_x + out_ch) * gate   # (B,C,H,W)
        if self.enable_vis:
            # In the current implementation:
            # - heatmap is the zoom-related spatial map
            # - attn is the attention weight tensor
            # - fused is the final output feature
            self._cache_visualizations(
                zoom_map=heatmap,
                attn_map=attn,
                out_feat=fused,
                spatial_size=(H, W),
            )

        return fused


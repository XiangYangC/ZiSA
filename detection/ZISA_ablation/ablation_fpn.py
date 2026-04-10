from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class ZiSAAblationFPN(FPN):
    """FPN variant that applies ZiSA ablation wrappers on selected outputs."""

    def __init__(
        self,
        attention_cfg: Optional[Dict] = None,
        attention_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.attention_cfg = attention_cfg.copy() if attention_cfg is not None else None
        self.attention_indices = attention_indices

        if self.attention_cfg is None:
            self.attention_modules = None
            return

        module_cfg = self.attention_cfg.copy()
        module_type = module_cfg.pop('type', 'ZiSAWrapper')
        out_channels = kwargs.get('out_channels', 256)

        self.attention_modules = nn.ModuleList()
        for idx in range(self.num_outs):
            if self.attention_indices is None or idx in self.attention_indices:
                layer_cfg = dict(type=module_type, in_channels=out_channels, **module_cfg)
                self.attention_modules.append(MODELS.build(layer_cfg))
            else:
                self.attention_modules.append(nn.Identity())

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        outputs = list(super().forward(inputs))
        if self.attention_modules is None:
            return tuple(outputs)

        for idx, feature in enumerate(outputs):
            outputs[idx] = self.attention_modules[idx](feature)
        return tuple(outputs)

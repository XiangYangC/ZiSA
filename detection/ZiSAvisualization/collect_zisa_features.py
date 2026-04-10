import os
import re
import sys
from contextlib import contextmanager
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

try:
    from mmdet.apis import inference_detector, init_detector
except ImportError as exc:
    raise ImportError("MMDetection is required for ZiSA visualization.") from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DETECTION_DIR)

for path in (PROJECT_ROOT, DETECTION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import model  # noqa: F401
from attention_modules import ZoomInSelfAttention


@contextmanager
def force_torch_load_weights_only_false():
    """
    PyTorch 2.6 changed torch.load(weights_only) default to True.
    Older MMEngine checkpoints may contain non-tensor metadata such as
    HistoryBuffer, which breaks loading unless weights_only=False is used.
    This patch is scoped to visualization-only checkpoint loading.
    """
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_model(config_path: str, checkpoint_path: str, device: Optional[str] = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with force_torch_load_weights_only_false():
        return init_detector(config_path, checkpoint_path, device=device)


def is_zisa_module(module: nn.Module, module_name: str) -> bool:
    cls_name = module.__class__.__name__.lower()
    lower_name = module_name.lower()
    return (
        isinstance(module, ZoomInSelfAttention)
        or "zisa" in cls_name
        or "zoominselfattention" in cls_name
        or "zisa" in lower_name
    )


def build_display_name(module_name: str) -> str:
    """Create a visualization-friendly module name."""
    match = re.search(r"(^|\.)(attention_modules)\.(\d+)$", module_name)
    if match:
        level_idx = int(match.group(3))
        return f"neck.P{level_idx + 3}.zisa"
    return module_name


def _matches_keywords(module_name: str, target_keywords: Optional[Iterable[str]]) -> bool:
    if not target_keywords:
        return True
    lower_name = module_name.lower()
    return any(keyword.lower() in lower_name for keyword in target_keywords)


def find_zisa_modules(
    model: nn.Module,
    target_keywords: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    matches: List[Tuple[str, nn.Module]] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        display_name = build_display_name(module_name)
        if is_zisa_module(module, module_name) and (
            _matches_keywords(module_name, target_keywords)
            or _matches_keywords(display_name, target_keywords)
        ):
            matches.append((display_name, module))
    return matches


def summarize_candidate_modules(model: nn.Module, max_items: int = 20) -> List[str]:
    """Collect module names that may help diagnose why ZiSA was not found."""
    candidates: List[str] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        cls_name = module.__class__.__name__.lower()
        lower_name = module_name.lower()
        if (
            "neck" in lower_name
            or "attention" in lower_name
            or "fpn" in lower_name
            or "attention" in cls_name
            or "fpn" in cls_name
        ):
            candidates.append(f"{module_name} ({module.__class__.__name__})")
        if len(candidates) >= max_items:
            break
    return candidates


def enable_visualization(modules: Iterable[Tuple[str, nn.Module]]) -> None:
    for _, module in modules:
        module.enable_vis = True
        module.vis_cache = {}


def disable_visualization(modules: Iterable[Tuple[str, nn.Module]]) -> None:
    for _, module in modules:
        module.enable_vis = False


def infer_and_collect(
    model: nn.Module,
    image_path: str,
    target_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules = find_zisa_modules(model, target_keywords=target_keywords)
    if not modules:
        keyword_text = ", ".join(target_keywords) if target_keywords else "ALL"
        candidates = summarize_candidate_modules(model)
        candidate_text = "\n".join(candidates) if candidates else "No neck/attention-like modules were found."
        raise RuntimeError(
            "No ZiSA modules were found in the loaded model.\n"
            f"Requested keywords: {keyword_text}\n"
            "This usually means the current config/checkpoint is not a ZiSA model "
            "(for example, a Base FPN config without ZoomInSelfAttention).\n"
            f"Candidate modules:\n{candidate_text}"
        )

    enable_visualization(modules)
    try:
        with torch.no_grad():
            inference_detector(model, image_path)

        collected: Dict[str, Dict[str, torch.Tensor]] = OrderedDict()
        for module_name, module in modules:
            cache = getattr(module, "vis_cache", {}) or {}
            valid_items: Dict[str, torch.Tensor] = {}
            for key in ("zoom_map", "attn_map", "out_feat"):
                value = cache.get(key)
                if value is not None:
                    valid_items[key] = value[:1]
            if valid_items:
                collected[module_name] = valid_items
        return collected
    finally:
        disable_visualization(modules)


def load_model_and_collect(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    target_keywords: Optional[Iterable[str]] = None,
    device: Optional[str] = None,
):
    model = load_model(config_path, checkpoint_path, device=device)
    features = infer_and_collect(model, image_path, target_keywords=target_keywords)
    return model, features

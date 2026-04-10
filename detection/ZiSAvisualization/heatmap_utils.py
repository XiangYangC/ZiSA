import os
from typing import Tuple

import cv2
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_numpy_map(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a normalized 2D numpy map."""
    if tensor is None:
        raise ValueError("tensor is None")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")

    array = tensor.detach().cpu().float()
    if array.dim() == 4:
        array = array[0, 0]
    elif array.dim() == 3:
        array = array[0]
    elif array.dim() != 2:
        raise ValueError(f"Unsupported tensor shape: {tuple(array.shape)}")

    array = array.numpy()
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val > min_val:
        array = (array - min_val) / (max_val - min_val)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return array.astype(np.float32)


def load_image_rgb(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(os.path.normpath(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_map(heatmap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def to_colormap(heatmap: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mixed = image_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.normpath(out_path), image_bgr)


def save_heatmap_and_overlay(
    tensor: torch.Tensor,
    image_rgb: np.ndarray,
    heat_out_path: str,
    overlay_out_path: str,
) -> None:
    heatmap = tensor_to_numpy_map(tensor)
    resized = resize_map(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heat_rgb = to_colormap(resized)
    overlay_rgb = overlay_heatmap(image_rgb, heat_rgb)
    save_rgb_image(heat_rgb, heat_out_path)
    save_rgb_image(overlay_rgb, overlay_out_path)

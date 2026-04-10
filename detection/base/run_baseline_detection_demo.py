import argparse
import os
import sys
from contextlib import contextmanager

import mmcv
import torch
from mmengine.utils import mkdir_or_exist

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.registry import VISUALIZERS
except ImportError as exc:
    raise ImportError("MMDetection is required for baseline detection visualization.") from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DETECTION_DIR)

for path in (PROJECT_ROOT, DETECTION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import custom registry definitions before model initialization.
import model  # noqa: F401


# 如果你希望通过右键/双击直接运行本脚本，
# DEFAULT_CONFIG_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/config.py"
# DEFAULT_CHECKPOINT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/epoch_300.pth"
# DEFAULT_IMAGE_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/generate_heatmap.jpg"
# DEFAULT_OUT_DIR = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/attn_vis/ZoomInSelfAttention"
# "D:\Pythoncode\VMamba-main\Windfarm_VMambaDetection\val\images\6-40-_jpg.rf.c940f1ec77fd6988e3c72793a067cbcb.jpg"
# 请在下面填写默认路径；如果使用命令行参数运行，也可以留空。/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/val/images/6-40-_jpg.rf.c940f1ec77fd6988e3c72793a067cbcb.jpg
DEFAULT_IMAGE_PATH = r"/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/val/images/1-27-_jpg.rf.00d23549923fd25ec64bfc5e0de9a88c.jpg"  # 输入图片路径，建议与 ZiSA 可视化使用同一张图
DEFAULT_CONFIG_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/Base/config.py"  # baseline 配置文件路径，不要使用带 ZiSA 的配置
DEFAULT_CHECKPOINT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/Base/epoch_300.pth"  # baseline 权重文件路径
DEFAULT_OUTPUT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/Output/Base"  # 输出图片路径，例如 baseline_detection.jpg
DEFAULT_DEVICE = "cuda:0"  # 推理设备，例如 cuda:0 或 cpu
DEFAULT_SCORE_THR = 0.3  # 检测框显示阈值

@contextmanager
def force_torch_load_weights_only_false():
    """Compatibility patch for PyTorch 2.6 + older MMEngine checkpoints."""
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline detection visualization in official MMDetection demo style."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--config", required=True, help="Path to the baseline config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the baseline checkpoint.")
    parser.add_argument("--out", required=True, help="Path to the output image.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, for example cuda:0 or cpu.")
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR, help="BBox score threshold.")
    return parser.parse_args()


def build_default_args():
    return argparse.Namespace(
        image=DEFAULT_IMAGE_PATH,
        config=DEFAULT_CONFIG_PATH,
        checkpoint=DEFAULT_CHECKPOINT_PATH,
        out=DEFAULT_OUTPUT_PATH,
        device=DEFAULT_DEVICE,
        score_thr=DEFAULT_SCORE_THR,
    )


def normalize_output_path(output_path: str) -> str:
    output_path = os.path.normpath(output_path)
    root, ext = os.path.splitext(output_path)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not ext:
        return os.path.join(output_path, "baseline_detection2.jpg")
    if ext.lower() not in valid_exts:
        return f"{output_path}.jpg"
    return output_path


def ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        mkdir_or_exist(parent_dir)


def save_detection_visualization(model, image_path: str, result, output_path: str, score_thr: float) -> None:
    if hasattr(model, "show_result"):
        model.show_result(
            image_path,
            result,
            score_thr=score_thr,
            out_file=output_path,
        )
        return

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    if hasattr(visualizer, "save_dir"):
        visualizer.save_dir = os.path.dirname(os.path.abspath(output_path))

    image = mmcv.imread(image_path)
    visualizer.add_datasample(
        name="baseline_detection",
        image=image,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=score_thr,
        out_file=output_path,
    )


def validate_paths(args) -> None:
    missing_values = []
    for key in ("image", "config", "checkpoint", "out"):
        value = getattr(args, key, None)
        if not value:
            missing_values.append(key)

    if missing_values:
        raise ValueError(
            "Missing required paths: "
            + ", ".join(missing_values)
            + ". Fill the DEFAULT_* variables in this script or pass command line arguments."
        )

    missing_files = []
    for path_value, label in (
        (args.image, "image"),
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
    ):
        if not os.path.exists(path_value):
            missing_files.append(f"{label}: {path_value}")

    if missing_files:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_files))


def run_official_demo_style(args) -> str:
    output_path = normalize_output_path(args.out)
    ensure_parent_dir(output_path)
    validate_paths(args)

    with force_torch_load_weights_only_false():
        model = init_detector(args.config, args.checkpoint, device=args.device)

    result = inference_detector(model, args.image)
    save_detection_visualization(model, args.image, result, output_path, args.score_thr)
    return output_path


def main():
    args = parse_args() if len(sys.argv) > 1 else build_default_args()
    output_path = run_official_demo_style(args)
    print(f"Saved baseline detection visualization to: {output_path}")


if __name__ == "__main__":
    main()

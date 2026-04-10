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
    raise ImportError("MMDetection is required for detection visualization.") from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DETECTION_DIR)

for path in (PROJECT_ROOT, DETECTION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import custom registry definitions before model initialization.
import model  # noqa: F401


# Fill these defaults if you want to launch this script directly."D:\Pythoncode\VMamba-main\Windfarm_VMambaDetection\val\images\1-27-_jpg.rf.00d23549923fd25ec64bfc5e0de9a88c.jpg"
DEFAULT_CONFIG_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/config.py"
DEFAULT_CHECKPOINT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/epoch_300.pth"
DEFAULT_IMAGE_PATH = r"/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/val/images/1-27-_jpg.rf.00d23549923fd25ec64bfc5e0de9a88c.jpg"
DEFAULT_OUT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/Output/ZoomInSelfAttention"
DEFAULT_SCORE_THR = 0.3
DEFAULT_DEVICE = "cuda:0"


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
    parser = argparse.ArgumentParser(description="Run single-image detection visualization.")
    parser.add_argument("--config", required=True, help="Path to the model config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--out", required=True, help="Path to the output visualization image.")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for drawing detection boxes.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu. Default: cuda:0",
    )
    return parser.parse_args()


def build_default_args():
    return argparse.Namespace(
        config=DEFAULT_CONFIG_PATH,
        checkpoint=DEFAULT_CHECKPOINT_PATH,
        image=DEFAULT_IMAGE_PATH,
        out=DEFAULT_OUT_PATH,
        score_thr=DEFAULT_SCORE_THR,
        device=DEFAULT_DEVICE,
    )


def validate_args(args):
    missing_items = []
    for key in ("config", "checkpoint", "image", "out"):
        value = getattr(args, key, None)
        if not value:
            missing_items.append(key)

    if missing_items:
        raise ValueError(
            "Missing required paths: "
            + ", ".join(missing_items)
            + ". Fill the DEFAULT_* variables in this script or pass command line arguments."
        )


def normalize_output_path(output_path: str) -> str:
    output_path = os.path.normpath(output_path)
    root, ext = os.path.splitext(output_path)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not ext:
        return os.path.join(output_path, "zisa_detection2.jpg")
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

    save_dir = os.path.dirname(os.path.abspath(output_path))
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    if hasattr(visualizer, "save_dir"):
        visualizer.save_dir = save_dir

    image = mmcv.imread(image_path)
    visualizer.add_datasample(
        name="detection_result",
        image=image,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=score_thr,
        out_file=output_path,
    )


def main(args):
    validate_args(args)
    args.out = normalize_output_path(args.out)
    ensure_parent_dir(args.out)

    with force_torch_load_weights_only_false():
        model = init_detector(args.config, args.checkpoint, device=args.device)

    result = inference_detector(model, args.image)

    save_detection_visualization(
        model=model,
        image_path=args.image,
        result=result,
        output_path=args.out,
        score_thr=args.score_thr,
    )

    print(f"Saved detection visualization to: {os.path.normpath(args.out)}")


if __name__ == "__main__":
    runtime_args = parse_args() if len(sys.argv) > 1 else build_default_args()
    main(runtime_args)

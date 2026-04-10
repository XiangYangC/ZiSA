import argparse
import importlib.util
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_module(module_name: str, file_name: str):
    module_path = os.path.join(CURRENT_DIR, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_module = load_local_module("zisa_collect_zisa_features", "collect_zisa_features.py")
heatmap_module = load_local_module("zisa_heatmap_utils", "heatmap_utils.py")

load_model_and_collect = collect_module.load_model_and_collect
ensure_dir = heatmap_module.ensure_dir
load_image_rgb = heatmap_module.load_image_rgb
save_heatmap_and_overlay = heatmap_module.save_heatmap_and_overlay


# Direct-run defaults. Fill these if you want to launch this file directly.
DEFAULT_CONFIG_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/config.py"
DEFAULT_CHECKPOINT_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention12/epoch_300.pth"
# DEFAULT_IMAGE_PATH = r"/home/cruiy/code/python/VMamba-main/VMamba-main/generate_heatmap.jpg"
DEFAULT_IMAGE_PATH = r"/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection/val/images/1-27-_jpg.rf.00d23549923fd25ec64bfc5e0de9a88c.jpg"
DEFAULT_OUT_DIR = r"/home/cruiy/code/python/VMamba-main/VMamba-main/detection/attn_vis/ZoomInSelfAttention/1"
DEFAULT_TARGET_KEYWORDS = []
DEFAULT_DEVICE = None


def sanitize_module_name(module_name: str) -> str:
    return module_name.replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ZiSA intermediate feature visualization.")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", required=True, help="Directory for saving visualizations.")
    parser.add_argument(
        "--target-keywords",
        nargs="*",
        default=None,
        help="Only save ZiSA modules whose names contain these keywords, for example: P3 P4",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu or cuda:0. Defaults to cuda:0 when available.",
    )
    return parser.parse_args()


def build_default_args():
    return argparse.Namespace(
        config=DEFAULT_CONFIG_PATH,
        checkpoint=DEFAULT_CHECKPOINT_PATH,
        image=DEFAULT_IMAGE_PATH,
        out_dir=DEFAULT_OUT_DIR,
        target_keywords=DEFAULT_TARGET_KEYWORDS or None,
        device=DEFAULT_DEVICE,
    )


def validate_args(args):
    missing_items = []
    for key in ("config", "checkpoint", "image", "out_dir"):
        value = getattr(args, key, None)
        if not value:
            missing_items.append(key)

    if missing_items:
        raise ValueError(
            "Missing required paths: "
            + ", ".join(missing_items)
            + ". Fill the DEFAULT_* variables in this script or pass command line arguments."
        )


def main(args):
    validate_args(args)
    ensure_dir(args.out_dir)
    image_rgb = load_image_rgb(args.image)

    _, collected = load_model_and_collect(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        target_keywords=args.target_keywords,
        device=args.device,
    )

    for module_name, cache in collected.items():
        safe_name = sanitize_module_name(module_name)
        for cache_key in ("zoom_map", "attn_map", "out_feat"):
            tensor = cache.get(cache_key)
            if tensor is None:
                continue

            heat_out_path = os.path.join(args.out_dir, f"{safe_name}_{cache_key}_heat.jpg")
            overlay_out_path = os.path.join(args.out_dir, f"{safe_name}_{cache_key}_overlay.jpg")
            save_heatmap_and_overlay(
                tensor=tensor,
                image_rgb=image_rgb,
                heat_out_path=heat_out_path,
                overlay_out_path=overlay_out_path,
            )

    print(f"Saved ZiSA visualizations to: {os.path.normpath(args.out_dir)}")


if __name__ == "__main__":
    runtime_args = parse_args() if len(sys.argv) > 1 else build_default_args()
    main(runtime_args)

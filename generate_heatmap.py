import sys
import os
import types
import cv2
import numpy as np
import torch
import importlib.util
import functools

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ==================== 🔧 STEP -1: PyTorch 2.6+ 补丁 ====================
print("🔧 正在应用 PyTorch 2.6+ 兼容性补丁...")
_original_load = torch.load


@functools.wraps(_original_load)
def unsafe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = unsafe_load

# ==================== 🔧 STEP 0: Mock mmseg ====================
mmseg = types.ModuleType("mmseg")
mmseg.registry = types.ModuleType("registry")


class MockRegistry:
    def register_module(self, module=None, force=False):
        def _register(cls): return cls

        return _register


mmseg.registry.MODELS = MockRegistry()
sys.modules["mmseg"] = mmseg
sys.modules["mmseg.registry"] = mmseg.registry

# ==================== 🔧 配置区域 ====================
BACKBONE_DEF_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/model.py'
# D:\Pythoncode\VMamba-main\VMamba-main\detection\model.py
NECK_DEF_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/enhanced_fpn.py'
# IMG_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/generate_heatmap.jpg'
IMG_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/generate_heatmap.jpg'
CONFIG_FILE = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/configs/windfarm/faster_rcnn_vssm_windfarm_with_attention.py'
CHECKPOINT_FILE = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/ZoomInSelfAttention/epoch_300.pth'

# 【关键改动 1】换到 P4 层 (convs[1])，这层通常能让不同大小的目标热力更均衡！
TARGET_LAYER_NAME = 'neck.fpn_convs[2]'

# TARGET_LAYER_NAMES = [
#     'neck.fpn_convs[0]', # P3: 关注微小纹理
#     'neck.fpn_convs[1]', # P4: 关注中等轮廓
#     'neck.fpn_convs[2]'  # P5: 关注整体形态
# ]

OUTPUT_PATH = 'gradcam_final_clean.jpg'


# ==================== 🔧 加载逻辑 ====================
def load_custom_file(file_path, description):
    if not os.path.exists(file_path): return
    try:
        file_dir = os.path.dirname(file_path)
        project_root = os.path.dirname(os.path.dirname(file_path))
        if project_root not in sys.path: sys.path.insert(0, project_root)
        if file_dir not in sys.path: sys.path.insert(0, file_dir)
        module_name = "custom_loaded_" + os.path.basename(file_path).replace('.', '_')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"✅ 成功加载 {description}")
    except Exception as e:
        pass


load_custom_file(BACKBONE_DEF_PATH, "Backbone")
load_custom_file(NECK_DEF_PATH, "Neck")

# ==================== 🔧 核心逻辑 (Wrapper) ====================
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget


class MMDetGradCAMWrapper(torch.nn.Module):
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        results = self.model(x, self.data_samples, mode='predict')
        formatted_outputs = []
        for res in results:
            pred = res.pred_instances
            formatted_outputs.append({'boxes': pred.bboxes, 'scores': pred.scores, 'labels': pred.labels})
        return formatted_outputs


def get_gradcam_image(model, target_layer_name, img_path, output_path):
    if not os.path.exists(img_path): return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_size = (1024, 1024)
    img_resized = cv2.resize(img, input_size)
    rgb_img_float = np.float32(img_resized) / 255
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(next(model.parameters()).device)

    data_sample = DetDataSample()
    data_sample.set_metainfo({'img_shape': (1024, 1024), 'ori_shape': img.shape[:2], 'scale_factor': (1.0, 1.0)})
    batch_data_samples = [data_sample.to(next(model.parameters()).device)]

    print("🔍 正在进行前向推理...")
    with torch.no_grad():
        wrapper_temp = MMDetGradCAMWrapper(model, batch_data_samples)
        outputs = wrapper_temp(input_tensor)

    if len(outputs) == 0: return

    # 【关键改动 2】阈值设为 0.5！杀掉所有背景噪音，只留 0.99 和 0.65
    SCORE_THRESHOLD = 0.5
    scores = outputs[0]['scores']
    valid_mask = scores > SCORE_THRESHOLD

    if not valid_mask.any():
        print(f"⚠️ 没有置信度大于 {SCORE_THRESHOLD} 的目标。")
        return

    valid_boxes = outputs[0]['boxes'][valid_mask]
    valid_labels = outputs[0]['labels'][valid_mask]
    valid_scores = scores[valid_mask]

    print(f"📊 过滤后剩余 {len(valid_boxes)} 个主要目标 (Score > {SCORE_THRESHOLD})")

    try:
        target_layers = [eval(f"model.{target_layer_name}")]
    except Exception as e:
        print(f"❌ 找不到层 '{target_layer_name}': {e}")
        return

    wrapped_model = MMDetGradCAMWrapper(model, batch_data_samples)
    targets = []
    for i in range(len(valid_boxes)):
        targets.append(
            FasterRCNNBoxScoreTarget(labels=valid_labels[i].unsqueeze(0), bounding_boxes=valid_boxes[i].unsqueeze(0)))

    print(f"🎨 正在生成 P4 层热力图...")
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 画框
    for i in range(len(valid_boxes)):
        x1, y1, x2, y2 = valid_boxes[i].cpu().numpy().astype(int)
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(visualization, f"{valid_scores[i]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                    2)

    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, visualization)
    print(f"✅ 成功！干净的最终结果已保存至: {output_path}")


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 加载中...")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)
        get_gradcam_image(model, TARGET_LAYER_NAME, IMG_PATH, OUTPUT_PATH)
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


import sys
import os
import types
import cv2
import numpy as np
import torch
import importlib.util
import functools
from mmdet.structures.bbox import bbox2roi



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

# ==================== 🔧 配置区域 (Base 模型) ====================
BACKBONE_DEF_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/model.py'
NECK_DEF_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/enhanced_fpn.py'
IMG_PATH = '/home/cruiy/code/python/VMamba-main/VMamba-main/generate_heatmap.jpg'

# Base 配置文件
CONFIG_FILE = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/configs/windfarm/mask_rcnn_vssm_windfarm.py'
# Base 权重文件
CHECKPOINT_FILE = '/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/Base/epoch_300.pth'

# 目标层：还是先试 P4 (convs[1])，因为叶片是大物体，P4 概率最大
TARGET_LAYER_NAME = 'neck.fpn_convs[0]'
OUTPUT_PATH = 'gradcam_base_forced_grad.jpg'


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

# ==================== 🔧 核心逻辑 (普通 Wrapper - 仅用于获取框) ====================
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SimplePredictWrapper(torch.nn.Module):
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        return self.model(x, self.data_samples, mode='predict')


# ==================== 🔧 核心逻辑 (强力 Wrapper - 用于计算梯度) ====================
class DifferentiableWrapper(torch.nn.Module):
    def __init__(self, model, target_boxes):
        super().__init__()
        self.model = model
        self.target_boxes = target_boxes  # 这是一个 Tensor [N, 4]

    def forward(self, x):
        # 1. 提取特征 (Backbone + Neck)
        feat = self.model.extract_feat(x)

        # 2. 准备 ROI
        # 我们直接使用传入的 target_boxes 作为 ROI
        # bbox2roi 会自动添加 batch 索引 (0)
        rois = bbox2roi([self.target_boxes])

        # 3. ROI Extract (手动调用 bbox_roi_extractor)
        if self.model.roi_head.with_shared_head:
            # 如果有 shared head，逻辑会复杂点，但 FasterRCNN 通常没有
            pass

        bbox_feats = self.model.roi_head.bbox_roi_extractor(
            feat[:self.model.roi_head.bbox_roi_extractor.num_inputs], rois)

        # 4. Head Forward (FC layers)
        if self.model.roi_head.with_shared_head:
            bbox_feats = self.model.roi_head.shared_head(bbox_feats)

        flatten_feats = bbox_feats.flatten(1)
        cls_score, bbox_pred = self.model.roi_head.bbox_head(flatten_feats)

        # 返回分类分数 [N_boxes, N_classes]
        # 这是一个纯数学计算过程，绝对包含梯度！
        return cls_score


def get_gradcam_image(model, target_layer_name, img_path, output_path):
    if not os.path.exists(img_path): return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_size = (1024, 1024)
    img_resized = cv2.resize(img, input_size)
    rgb_img_float = np.float32(img_resized) / 255
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad = True

    data_sample = DetDataSample()
    data_sample.set_metainfo({'img_shape': (1024, 1024), 'ori_shape': img.shape[:2], 'scale_factor': (1.0, 1.0)})
    batch_data_samples = [data_sample.to(next(model.parameters()).device)]

    # === 第一步：先跑一遍预测，拿到框 ===
    print("🔍 [Step 1] Base 模型前向推理获取目标框...")
    with torch.no_grad():
        simple_wrapper = SimplePredictWrapper(model, batch_data_samples)
        results = simple_wrapper(input_tensor)

    pred = results[0].pred_instances
    scores = pred.scores
    SCORE_THRESHOLD = 0.5
    valid_mask = scores > SCORE_THRESHOLD

    if not valid_mask.any():
        print(f"⚠️ 没有目标 > {SCORE_THRESHOLD}")
        return

    valid_boxes = pred.bboxes[valid_mask]
    valid_labels = pred.labels[valid_mask]
    valid_scores = scores[valid_mask]

    print(f"📊 找到 {len(valid_boxes)} 个目标，准备计算梯度...")

    # === 第二步：换上“强力 Wrapper”计算梯度 ===

    # 开启所有参数梯度
    for param in model.parameters():
        param.requires_grad = True

    try:
        target_layers = [eval(f"model.{target_layer_name}")]
    except:
        print(f"❌ 找不到层 {target_layer_name}")
        return

    # 初始化强力 Wrapper，把刚才找到的框传进去
    diff_wrapper = DifferentiableWrapper(model, valid_boxes)

    # 构建 Targets
    # 因为我们的 Wrapper 返回的是 [N_boxes, N_classes]
    # 我们使用 ClassifierOutputTarget，针对每个框的预测类别进行最大化
    targets = []
    for i in range(len(valid_boxes)):
        targets.append(ClassifierOutputTarget(valid_labels[i].item()))

    print(f"🎨 [Step 2] 正在强制回传梯度 ({target_layer_name})...")
    cam = GradCAM(model=diff_wrapper, target_layers=target_layers)

    # 生成 CAM
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
    print(f"✅ 成功！Base 结果已保存至: {output_path}")


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Base 加载中...")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)
        get_gradcam_image(model, TARGET_LAYER_NAME, IMG_PATH, OUTPUT_PATH)
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
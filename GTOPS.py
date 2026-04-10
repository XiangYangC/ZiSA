import sys
import os
import torch
import torch.nn as nn
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmengine.analysis import get_model_complexity_info
from mmdet.structures import DetDataSample

# 引入官方工具的依赖
project_root = os.getcwd()
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'detection'))

# ================= 1. 环境准备与注册 =================
print(f"📂 项目根目录: {project_root}")
print("🔨 正在强制注册 detection.model ...")
try:
    import detection.model

    print("✅ 成功导入 'detection.model'")
except ImportError as e:
    print(f"⚠️ 导入 detection.model 失败: {e}")
    try:
        import model

        print("✅ 通过 'import model' 导入成功")
    except:
        pass

register_all_modules(init_default_scope=True)


# ================= 2. 吞噬输出的 Wrapper (关键修改) =================
class ModelWrapper(nn.Module):
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        # 1. 拆包防御 (防止字典传入)
        if isinstance(x, dict):
            x = x['inputs'] if 'inputs' in x else list(x.values())[0]

        # 2. 运行模型 (这里会产生真实的 FLOPs)
        # mode='predict' 会跑完 Backbone -> Neck -> RPN -> ROI Head
        # 虽然它返回的是非法类型 DetDataSample，但我们不把它传出去
        _ = self.model(x, data_samples=self.data_samples, mode='predict')

        # 3. 【关键修复】返回一个假的 Tensor 骗过 JIT
        # 只要返回的是 Tensor，JIT 就不会报错，计算统计也已经完成了
        return torch.tensor([0.])


# ================= 3. 计算函数 =================
def calculate_flops_safe(config_path):
    print(f"\n📖 读取配置文件: {config_path}")
    cfg = Config.fromfile(config_path)

    print("🏗️ 正在构建模型...")
    model = MODELS.build(cfg.model)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # 构造输入
    input_shape = (800, 1333)  # H, W
    inputs = torch.randn(1, 3, input_shape[0], input_shape[1])
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    # 构造元数据
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (input_shape[0], input_shape[1]),
        'ori_shape': (input_shape[0], input_shape[1]),
        'scale_factor': (1.0, 1.0)
    })
    data_samples_list = [data_sample]

    # 包装模型
    wrapped_model = ModelWrapper(model, data_samples_list)

    print("🧮 正在计算 FLOPs (Wrapper 吞噬输出版)...")

    # 使用 Tuple 传递位置参数
    res = get_model_complexity_info(wrapped_model, inputs=(inputs,))

    return res['flops_str'], res['params_str']


# ================= 4. 主程序 =================
config_path = 'detection/configs/windfarm/mask_rcnn_vssm_windfarm.py'

try:
    flops, params = calculate_flops_safe(config_path)

    print("\n" + "=" * 40)
    print("🏆 最终结果 (VMamba-ZiSA)")
    print("=" * 40)
    print(f"🚀 FLOPs : {flops}")
    print(f"📦 Params: {params}")
    print("=" * 40)

except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback

    traceback.print_exc()
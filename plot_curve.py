import json
import matplotlib.pyplot as plt
import os


# ================= 配置区域 =================
# 请填入 scalars.json 的完整绝对路径
# 例如：/home/cruiy/.../vis_data/scalars.json
SCALARS_JSON_PATH = "/home/cruiy/code/python/VMamba-main/VMamba-main/detection/work_dirs/Base/20251207_162839/vis_data/scalars.json"


# ===========================================

def load_scalars_json(json_path):
    data = {
        'step': [], 'loss': [], 'lr': [],
        'val_step': [], 'mAP': [], 'mAP_50': [], 'mAP_75': []
    }

    if not os.path.exists(json_path):
        print(f"错误：找不到文件 {json_path}")
        return None

    with open(json_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
            except:
                continue

            step = log.get('step', 0)

            # 1. 提取 Loss (可能叫 loss 或 train/loss)
            # MMEngine 有时会把 loss 拆开，我们找 'loss' 关键字
            if 'loss' in log:
                data['step'].append(step)
                data['loss'].append(log['loss'])
            elif 'train/loss' in log:
                data['step'].append(step)
                data['loss'].append(log['train/loss'])

            # 2. 提取 LR
            if 'lr' in log:
                data['lr'].append(log['lr'])
            elif 'train/lr' in log:
                data['lr'].append(log['train/lr'])

            # 3. 提取 mAP (验证集指标)
            # 常见 key: coco/bbox_mAP, val/coco/bbox_mAP
            if 'coco/bbox_mAP' in log:
                data['val_step'].append(step)
                data['mAP'].append(log['coco/bbox_mAP'])
                data['mAP_50'].append(log.get('coco/bbox_mAP_50', 0))
                data['mAP_75'].append(log.get('coco/bbox_mAP_75', 0))

    return data


def plot_curves(data, save_path):
    # 检查是否有数据
    if not data or not data['step']:
        print("未读取到训练数据，请检查 json 文件内容。")
        return

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Loss 曲线
    axes[0].plot(data['step'], data['loss'], label='Train Loss', color='#ff6b6b', alpha=0.8, linewidth=1)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Steps (Iterations)')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 2. LR 曲线
    # LR 的长度可能和 Loss 不完全一致（取决于记录频率），这里简单截取
    min_len = min(len(data['step']), len(data['lr']))
    if min_len > 0:
        axes[1].plot(data['step'][:min_len], data['lr'][:min_len], label='Learning Rate', color='#ffa502')
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Steps')
        axes[1].legend()
        axes[1].grid(True)

    # 3. mAP 曲线
    if data['mAP']:
        axes[2].plot(data['val_step'], data['mAP'], 'o-', label='mAP', color='#1e90ff')
        axes[2].plot(data['val_step'], data['mAP_50'], '^-', label='mAP_50', color='#2ed573')
        axes[2].plot(data['val_step'], data['mAP_75'], 's-', label='mAP_75', color='#a55eea')
        axes[2].set_title('Validation Accuracy (COCO mAP)')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('mAP')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', fontsize=12)
        axes[2].set_title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 成功！曲线图已保存至: {save_path}")


if __name__ == "__main__":
    data = load_scalars_json(SCALARS_JSON_PATH)
    if data:
        save_file = SCALARS_JSON_PATH.replace('.json', '_curves.png')
        plot_curves(data, save_file)
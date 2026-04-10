#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windfarm 检测数据集训练脚本 (双卡修正版)
"""

import os
import sys

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # detection/ 目录
detection_dir = current_dir  # 脚本就在 detection 目录下
sys.path.insert(0, detection_dir)

# ========== 配置区域：请修改以下参数 ==========

# 项目根目录
PROJECT_ROOT = "/home/cruiy/code/python/VMamba-main"

# 数据集路径
DATA_ROOT = os.path.join(PROJECT_ROOT, "Windfarm_VMambaDetection")

# 配置文件路径 (带 P2 层注意力的版本)
CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "faster_rcnn_vssm_windfarm_with_attention.py")

# 【修改点 1】批次大小 (总数)
# 之前单卡 Batch=2 爆显存，所以双卡设为 2 (即每张卡 1 张图)
BATCH_SIZE = 2

# 输出目录
WORK_DIR = os.path.join(current_dir, "work_dirs", "ZISA_P2_MultiGPU")

# 预训练权重路径
PRETRAINED = ""

# 恢复训练
RESUME = ""

# 【修改点 2】GPU 数量 (改为 2)
NUM_GPUS = 2


# ==============================================

def main():
    """主函数"""
    # 检查路径是否存在 (略微简化，只做关键检查)
    if not os.path.exists(CFG_PATH):
        print(f"错误：配置文件不存在: {CFG_PATH}")
        return

    print("=" * 60)
    print("Windfarm 检测数据集训练 (双卡 DDP 模式)")
    print("=" * 60)
    print(f"配置文件: {CFG_PATH}")
    print(f"批次大小 (Total): {BATCH_SIZE} (每卡: {BATCH_SIZE // NUM_GPUS})")
    print(f"GPU 数量: {NUM_GPUS}")
    print("=" * 60)

    # 构建训练命令
    # 注意：单卡用 tools/train.py，多卡用 tools/dist_train.sh

    # 准备 cfg-options 参数字符串
    cfg_options_str = f"data_root={DATA_ROOT} train_dataloader.batch_size={BATCH_SIZE}"
    if PRETRAINED:
        cfg_options_str += f" model.backbone.pretrained={PRETRAINED}"

    if NUM_GPUS > 1:
        # 【修改点 3】修复多卡模式下参数丢失的 BUG
        dist_train_script = os.path.join(current_dir, "tools", "dist_train.sh")

        # 这里的命令格式是: bash dist_train.sh <config> <gpu_num> [args]
        # 注意：我们必须把 --work-dir 和 --cfg-options 传进去
        cmd = (
            f"bash {dist_train_script} "
            f"{CFG_PATH} "
            f"{NUM_GPUS} "
            f"--work-dir {WORK_DIR} "
            f"--cfg-options {cfg_options_str}"
        )

        if RESUME:
            cmd += f" --resume {RESUME}"

    else:
        # 单GPU训练
        train_script = os.path.join(current_dir, "tools", "train.py")
        cmd_parts = [
            sys.executable,
            train_script,
            CFG_PATH,
            "--work-dir", WORK_DIR,
            "--cfg-options", cfg_options_str,
        ]
        if RESUME:
            cmd_parts.extend(["--resume", RESUME])
        cmd = " ".join(cmd_parts)

    print(f"\n执行命令:")
    print(cmd)
    print("\n开始训练...\n")

    # 执行训练
    try:
        os.system(cmd)
    except Exception as e:
        print(f"\n训练出错: {e}")


if __name__ == '__main__':
    main()
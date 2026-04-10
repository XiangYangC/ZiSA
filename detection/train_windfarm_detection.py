#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windfarm 检测数据集训练脚本
可以直接运行此文件进行检测任务训练
"""

import os
import sys

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # detection/ 目录
detection_dir = current_dir  # 脚本就在 detection 目录下
sys.path.insert(0, detection_dir)

# ========== 配置区域：请修改以下参数 ==========

# 项目根目录（包含 VMamba 代码和 Windfarm_VMambaDetection 数据集）
PROJECT_ROOT = "/home/cruiy/code/python/VMamba-main"

# 数据集路径（相对于项目根目录）
DATA_ROOT = os.path.join(PROJECT_ROOT, "Windfarm_VMambaDetection")
# 如果数据集在其他位置，可以修改为绝对路径：
# DATA_ROOT = "/home/cruiy/code/python/VMamba-main/Windfarm_VMambaDetection"

# 配置文件路径（相对于当前 detection 目录）
# 不带注意力版本：
# CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "mask_rcnn_vssm_windfarm.py")
# 带注意力版本（V2.0）：
CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "faster_rcnn_vssm_windfarm_with_attention.py")

# 批次大小（根据 GPU 显存调整：8GB用2，16GB用4，24GB+用8）
BATCH_SIZE = 8

# 输出目录（工作目录）
# WORK_DIR = os.path.join(current_dir, "work_dirs", "Base")
WORK_DIR = os.path.join(current_dir, "work_dirs", "ZoomInSelfAttention=12345")

# 预训练权重路径（留空表示从头训练）
PRETRAINED = ""
# 可以使用分类任务的预训练权重：
# PRETRAINED = "/path/to/classification/checkpoint.pth"

# 恢复训练的检查点路径（留空表示从头开始）
RESUME = ""
# RESUME = "/path/to/checkpoint.pth"

# GPU 数量（单GPU训练设置为1）
NUM_GPUS = 1

# ==============================================

def main():
    """主函数"""
    # 检查数据集路径
    if not os.path.exists(DATA_ROOT):
        print(f"错误：数据集路径不存在: {DATA_ROOT}")
        print(f"请修改脚本中的 DATA_ROOT 变量")
        input("按 Enter 键退出...")
        return
    
    # 检查标注文件
    train_ann = os.path.join(DATA_ROOT, "annotations", "instances_train.json")
    val_ann = os.path.join(DATA_ROOT, "annotations", "instances_val.json")
    
    if not os.path.exists(train_ann):
        print(f"错误：训练标注文件不存在: {train_ann}")
        input("按 Enter 键退出...")
        return
    
    if not os.path.exists(val_ann):
        print(f"错误：验证标注文件不存在: {val_ann}")
        input("按 Enter 键退出...")
        return
    
    # 检查配置文件
    if not os.path.exists(CFG_PATH):
        print(f"错误：配置文件不存在: {CFG_PATH}")
        input("按 Enter 键退出...")
        return
    
    print("=" * 60)
    print("Windfarm 检测数据集训练")
    print("=" * 60)
    print(f"配置文件: {CFG_PATH}")
    print(f"数据集路径: {DATA_ROOT}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"输出目录: {WORK_DIR}")
    print(f"GPU 数量: {NUM_GPUS}")
    if PRETRAINED:
        print(f"预训练权重: {PRETRAINED}")
    if RESUME:
        print(f"恢复检查点: {RESUME}")
    print("=" * 60)
    
    # 构建训练命令
    train_script = os.path.join(current_dir, "tools", "train.py")
    
    cmd_parts = [
        sys.executable,
        train_script,
        CFG_PATH,
        "--work-dir", WORK_DIR,
        "--cfg-options",
        f"data_root={DATA_ROOT}",
        f"train_dataloader.batch_size={BATCH_SIZE}",
    ]
    
    if PRETRAINED:
        cmd_parts.extend(["--cfg-options", f"model.backbone.pretrained={PRETRAINED}"])
    
    if RESUME:
        cmd_parts.extend(["--resume", RESUME])
    
    # 多GPU训练
    if NUM_GPUS > 1:
        dist_train_script = os.path.join(current_dir, "tools", "dist_train.sh")
        cmd = f"bash {dist_train_script} {CFG_PATH} {NUM_GPUS} --work-dir {WORK_DIR}"
        if PRETRAINED:
            cmd += f" --cfg-options model.backbone.pretrained={PRETRAINED}"
        if RESUME:
            cmd += f" --resume {RESUME}"
    else:
        # 单GPU训练
        cmd = " ".join(cmd_parts)
    
    print(f"\n执行命令:")
    print(cmd)
    print("\n开始训练...\n")
    
    # 执行训练
    try:
        if NUM_GPUS > 1:
            os.system(cmd)
        else:
            os.system(cmd)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")


if __name__ == '__main__':
    main()


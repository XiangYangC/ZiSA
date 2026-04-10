#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Faster R-CNN (ResNet50) 对比实验启动脚本
用于 Windfarm 数据集，从零训练 (Scratch)
可以直接右键运行此文件
"""

import os
import sys

# 获取当前脚本所在目录 (detection/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# ========== 配置区域 ==========

# 1. 配置文件路径 (指向我们刚才创建的那个 Scratch + BS=8 的配置)
# 注意：确保这个文件存在于 configs/windfarm/ 目录下
# CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "faster_rcnn_r18_windfarm.py")
CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "faster_rcnn_r18_zisa.py")
# 2. GPU 数量 (对比实验需要用双卡)
NUM_GPUS = 2

# 3. 工作目录 (留空则默认使用配置文件中定义的 work_dir)
# 如果你想覆盖配置文件里的设置，可以在这里写新的路径
WORK_DIR_OVERRIDE = ""


# ==============================

def main():
    print("=" * 60)
    print("🚀 Faster R-CNN (ResNet18) 对比实验启动")
    print("=" * 60)

    # 1. 检查配置文件是否存在
    if not os.path.exists(CFG_PATH):
        print(f"❌ 错误：找不到配置文件: {CFG_PATH}")
        print("请检查文件是否创建，或路径是否正确。")
        return

    print(f"📄 配置文件: {CFG_PATH}")
    print(f"🎮 GPU 数量: {NUM_GPUS}")

    # 2. 构建启动命令
    # MMDetection 的分布式训练脚本路径
    dist_train_script = os.path.join(current_dir, "tools", "dist_train.sh")

    if not os.path.exists(dist_train_script):
        print(f"❌ 错误：找不到启动工具: {dist_train_script}")
        return

    # 拼接命令: bash tools/dist_train.sh <CONFIG> <GPUS> [args]
    cmd = f"bash {dist_train_script} {CFG_PATH} {NUM_GPUS}"

    # 如果指定了新的工作目录，追加参数
    if WORK_DIR_OVERRIDE:
        cmd += f" --work-dir {WORK_DIR_OVERRIDE}"
        print(f"📂 工作目录: {WORK_DIR_OVERRIDE}")
    else:
        print(f"📂 工作目录: (使用配置文件中的默认设置)")

    print("-" * 60)
    print(f"💻 执行命令:\n{cmd}")
    print("-" * 60)
    print("开始训练... (请耐心等待日志输出)")

    # 3. 执行命令
    try:
        os.system(cmd)
    except KeyboardInterrupt:
        print("\n🚫 用户中断训练")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


if __name__ == '__main__':
    main()
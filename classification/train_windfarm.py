#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windfarm 数据集训练脚本
可以直接右键运行此文件进行训练
"""

import os
import sys

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# ========== 配置区域：请修改以下参数 ==========

# 项目根目录（包含 VMamba 代码和 Windfarm_VMamba 数据集）
PROJECT_ROOT = "/home/cruiy/code/python/VMamba-main"

# 数据集路径（相对于项目根目录）
DATA_PATH = os.path.join(PROJECT_ROOT, "Windfarm_VMamba")
# 如果数据集在其他位置，可以修改为绝对路径：
# DATA_PATH = "/home/cruiy/code/python/VMamba-main/Windfarm_VMamba"

# 批次大小（根据 GPU 显存调整：8GB用16，16GB用32，24GB+用64）
BATCH_SIZE = 32

# 输出目录（相对于当前脚本目录）
OUTPUT_DIR = os.path.join(current_dir, "output", "windfarm")

# 预训练权重路径（留空表示从头训练）
PRETRAINED = ""
# PRETRAINED = "/path/to/pretrained/checkpoint.pth"

# 恢复训练的检查点路径（留空表示从头开始）
RESUME = ""
# RESUME = "/path/to/checkpoint.pth"

# ==============================================

def main():
    """主函数"""
    # 配置文件路径
    CFG_PATH = os.path.join(current_dir, "configs", "windfarm", "vmambav2_tiny_224_windfarm.yaml")
    
    # 检查数据集路径
    if not os.path.exists(DATA_PATH):
        print(f"错误：数据集路径不存在: {DATA_PATH}")
        print(f"请修改脚本中的 DATA_PATH 变量")
        input("按 Enter 键退出...")
        return
    
    # 检查配置文件
    if not os.path.exists(CFG_PATH):
        print(f"错误：配置文件不存在: {CFG_PATH}")
        input("按 Enter 键退出...")
        return
    
    print("=" * 60)
    print("Windfarm 数据集训练")
    print("=" * 60)
    print(f"配置文件: {CFG_PATH}")
    print(f"数据集路径: {DATA_PATH}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")
    if PRETRAINED:
        print(f"预训练权重: {PRETRAINED}")
    if RESUME:
        print(f"恢复检查点: {RESUME}")
    print("=" * 60)
    
    # 设置环境变量（单GPU模式）
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # 构建命令行参数
    sys.argv = [
        'train_windfarm.py',
        '--cfg', CFG_PATH,
        '--data-path', DATA_PATH,
        '--batch-size', str(BATCH_SIZE),
        '--output', OUTPUT_DIR,
    ]
    
    if PRETRAINED:
        sys.argv.extend(['--pretrained', PRETRAINED])
    
    if RESUME:
        sys.argv.extend(['--resume', RESUME])
    
    # 导入并运行 main.py
    try:
        # 直接执行 main.py 的代码
        from main import parse_option, main as train_main
        
        print("\n开始训练...\n")
        
        # 解析参数
        args, config = parse_option()
        
        # 初始化分布式环境
        import torch
        import torch.distributed as dist
        
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = 0
            world_size = 1
        
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        dist.barrier()
        
        # 设置随机种子
        import random
        import numpy as np
        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 设置学习率（根据批次大小调整）
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()
        
        # 创建输出目录
        config.defrost()
        if dist.get_rank() == 0:
            obj = [config.OUTPUT]
        else:
            obj = [None]
        dist.broadcast_object_list(obj)
        dist.barrier()
        config.OUTPUT = obj[0]
        config.freeze()
        os.makedirs(config.OUTPUT, exist_ok=True)
        
        # 创建日志（需要设置为全局变量，因为 main.py 中的 main 函数使用全局 logger）
        from utils.logger import create_logger
        import main as main_module  # 导入 main 模块，以便设置全局 logger
        import json  # 导入 json 用于配置序列化
        
        # 创建 logger 并设置为 main 模块的全局变量
        main_module.logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
        logger = main_module.logger  # 本地引用
        
        # 打印配置信息
        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")
        
        # 打印配置
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))
        
        # 内存限制处理（如果需要）
        if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
            torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
            usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
            print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)
        
        # 运行训练
        train_main(config, args)
        
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")


if __name__ == '__main__':
    main()


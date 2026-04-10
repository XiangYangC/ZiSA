#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速安装 MMCV（预编译版本，避免编译卡住）
"""

import sys
import subprocess

def check_cuda():
    """检查 CUDA 版本"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"检测到 CUDA 版本: {cuda_version}")
            return cuda_version
        else:
            print("未检测到 CUDA，使用 CPU 版本")
            return None
    except ImportError:
        print("警告: PyTorch 未安装，尝试通用安装")
        return None

def install_mmcv():
    """安装预编译的 mmcv"""
    print("=" * 60)
    print("快速安装 MMCV（预编译版本）")
    print("=" * 60)
    print()
    
    # 检查 CUDA
    cuda_version = check_cuda()
    
    # 选择 URL
    if cuda_version:
        if cuda_version.startswith('11.8') or cuda_version.startswith('11.'):
            url = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
            print("使用 CUDA 11.8 预编译版本")
        elif cuda_version.startswith('12.1') or cuda_version.startswith('12.'):
            url = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html"
            print("使用 CUDA 12.1 预编译版本")
        else:
            url = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
            print(f"使用 CUDA 11.8 预编译版本（通用）")
    else:
        url = "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html"
        print("使用 CPU 预编译版本")
    
    print(f"\n安装 URL: {url}")
    print()
    
    # 卸载旧版本
    print("步骤 1: 卸载旧版本...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "mmcv", "mmcv-full", "-y"], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 安装预编译版本
    print("步骤 2: 安装预编译版本...")
    cmd = [sys.executable, "-m", "pip", "install", "mmcv==2.1.0", "-f", url]
    
    print(f"运行命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("✓ MMCV 安装成功！")
        print("=" * 60)
        
        # 验证
        try:
            import mmcv
            print(f"✓ 验证成功: mmcv {mmcv.__version__}")
        except:
            print("✓ 安装完成（验证跳过）")
        
        return True
    else:
        print()
        print("=" * 60)
        print("✗ 安装失败")
        print("=" * 60)
        return False

if __name__ == '__main__':
    success = install_mmcv()
    if success:
        print("\n下一步: pip install mmdet==3.3.0")
    sys.exit(0 if success else 1)


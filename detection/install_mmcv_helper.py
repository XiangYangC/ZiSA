#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMCV 安装助手脚本
自动检测环境并安装合适的 mmcv 预编译版本
"""

import sys
import subprocess
import platform

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_cuda():
    """检查 CUDA 版本"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            pytorch_version = torch.__version__
            return True, cuda_version, pytorch_version
        else:
            return False, None, None
    except ImportError:
        return None, None, None  # PyTorch 未安装

def check_pytorch_version():
    """检查 PyTorch 版本"""
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        return version, major, minor
    except ImportError:
        return None, None, None

def get_mmcv_url():
    """根据环境获取 mmcv 预编译版本 URL"""
    cuda_available, cuda_version, pytorch_version = check_cuda()
    pytorch_ver, pytorch_major, pytorch_minor = check_pytorch_version()
    
    if pytorch_ver is None:
        print("错误: 未安装 PyTorch")
        print("请先安装 PyTorch:")
        print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        return None
    
    print(f"检测到 PyTorch 版本: {pytorch_ver}")
    
    if cuda_available:
        print(f"检测到 CUDA 版本: {cuda_version}")
        
        # 根据 CUDA 版本选择 URL
        if cuda_version.startswith('11.8') or cuda_version.startswith('11.'):
            return "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
        elif cuda_version.startswith('12.1') or cuda_version.startswith('12.'):
            return "https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html"
        else:
            print(f"警告: 未识别的 CUDA 版本 {cuda_version}")
            print("尝试使用 CUDA 11.8 版本...")
            return "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
    else:
        print("未检测到 CUDA，使用 CPU 版本")
        return "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html"

def install_mmcv_prebuilt():
    """安装预编译的 mmcv"""
    print("=" * 60)
    print("MMCV 预编译版本安装助手")
    print("=" * 60)
    print()
    
    # 获取 URL
    url = get_mmcv_url()
    if url is None:
        return False
    
    # 卸载旧版本（如果存在）
    print("步骤 1: 卸载旧版本 mmcv...")
    run_command("pip uninstall mmcv mmcv-full -y")
    
    # 安装预编译版本
    print(f"\n步骤 2: 安装预编译版本 mmcv==2.1.0...")
    print(f"使用 URL: {url}")
    
    cmd = f"pip install mmcv==2.1.0 -f {url}"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ MMCV 安装成功！")
        print("=" * 60)
        
        # 验证安装
        print("\n验证安装...")
        success, stdout, stderr = run_command('python -c "import mmcv; print(f\'mmcv {mmcv.__version__} 安装成功\')"')
        if success:
            print(stdout)
        else:
            print("警告: 验证失败，但安装可能已成功")
        
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ MMCV 安装失败")
        print("=" * 60)
        print(f"错误信息: {stderr}")
        print("\n请检查:")
        print("1. 网络连接是否正常")
        print("2. PyTorch 是否正确安装")
        print("3. CUDA 版本是否匹配")
        return False

def main():
    """主函数"""
    print("正在检测环境...")
    print()
    
    success = install_mmcv_prebuilt()
    
    if success:
        print("\n下一步:")
        print("  pip install mmdet==3.3.0")
    else:
        print("\n请尝试手动安装:")
        print("  查看 MMCV_INSTALL_GUIDE.md 获取详细说明")

if __name__ == '__main__':
    main()


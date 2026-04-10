#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测依赖安装检查脚本
检查并安装检测任务所需的依赖包
"""

import sys
import os
import subprocess
import importlib

def check_package(package_name, import_name=None):
    """
    检查包是否已安装
    
    Args:
        package_name: pip 包名
        import_name: 导入时的名称（如果不同）
    """
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """
    安装包
    
    Args:
        package_name: pip 包名（可以包含版本号）
    """
    try:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ {package_name} 安装失败")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("检测依赖安装检查")
    print("=" * 60)
    
    # 检查是否在 conda 环境中
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    if conda_env:
        print(f"\n当前 conda 环境: {conda_env}")
    else:
        print("\n提示: 未检测到 conda 环境")
        print("建议使用 conda 环境安装依赖，参考: INSTALL_CONDA.md")
    
    # 需要检查的包列表
    packages = [
        ("mmengine", "mmengine"),
        ("mmcv", "mmcv"),
        ("mmdet", "mmdet"),
        ("opencv-python-headless", "cv2"),
        ("ftfy", "ftfy"),
        ("regex", "regex"),
    ]
    
    # 检查每个包
    missing_packages = []
    for pip_name, import_name in packages:
        if check_package(pip_name, import_name):
            print(f"✓ {pip_name} 已安装")
        else:
            print(f"✗ {pip_name} 未安装")
            missing_packages.append(pip_name)
    
    if not missing_packages:
        print("\n" + "=" * 60)
        print("所有依赖包已安装！")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print(f"发现 {len(missing_packages)} 个缺失的包")
    print("=" * 60)
    
    # 安装命令
    install_commands = [
        "pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex",
        "pip install mmdet==3.3.0",
    ]
    
    print("\n请运行以下命令安装依赖：")
    print("-" * 60)
    for cmd in install_commands:
        print(cmd)
    print("-" * 60)
    
    # 询问是否自动安装
    try:
        response = input("\n是否自动安装缺失的包？(y/n): ").strip().lower()
        if response == 'y':
            print("\n开始安装...")
            
            # 安装基础包
            install_package("mmengine==0.10.1")
            install_package("mmcv==2.1.0")
            install_package("opencv-python-headless")
            install_package("ftfy")
            install_package("regex")
            
            # 安装 MMDetection
            install_package("mmdet==3.3.0")
            
            print("\n" + "=" * 60)
            print("安装完成！请重新运行检查脚本确认安装成功。")
            print("=" * 60)
        else:
            print("\n请手动运行上述命令安装依赖。")
    except KeyboardInterrupt:
        print("\n\n安装已取消。")

if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查服务器路径和文件是否存在
运行此脚本可以查看实际的文件路径
"""

import os
import sys

print("=" * 60)
print("检查服务器路径结构")
print("=" * 60)

# 检查项目根目录
project_root = "/home/cruiy/code/python/VMamba-main"
print(f"\n项目根目录: {project_root}")
print(f"是否存在: {os.path.exists(project_root)}")

if os.path.exists(project_root):
    print("\n项目根目录下的内容:")
    try:
        items = os.listdir(project_root)
        for item in sorted(items):
            item_path = os.path.join(project_root, item)
            item_type = "目录" if os.path.isdir(item_path) else "文件"
            print(f"  [{item_type}] {item}")
    except Exception as e:
        print(f"  无法列出目录内容: {e}")

# 检查可能的训练脚本路径
possible_paths = [
    "/home/cruiy/code/python/VMamba-main/VMamba-main/classification/train_windfarm.py",
    "/home/cruiy/code/python/VMamba-main/classification/train_windfarm.py",
    "/home/cruiy/code/python/VMamba-main/VMamba-main/VMamba-main/classification/train_windfarm.py",
]

print("\n检查训练脚本可能的位置:")
for path in possible_paths:
    exists = os.path.exists(path)
    print(f"  {path}")
    print(f"    存在: {exists}")

# 检查数据集路径
data_paths = [
    "/home/cruiy/code/python/VMamba-main/Windfarm_VMamba",
    "/home/cruiy/code/python/VMamba-main/VMamba-main/Windfarm_VMamba",
]

print("\n检查数据集可能的位置:")
for path in data_paths:
    exists = os.path.exists(path)
    print(f"  {path}")
    print(f"    存在: {exists}")
    if exists:
        try:
            subdirs = os.listdir(path)
            print(f"    子目录: {subdirs}")
        except:
            pass

# 检查当前脚本所在位置
print("\n当前脚本位置:")
print(f"  {__file__}")
print(f"  绝对路径: {os.path.abspath(__file__)}")
print(f"  所在目录: {os.path.dirname(os.path.abspath(__file__))}")

print("\n" + "=" * 60)


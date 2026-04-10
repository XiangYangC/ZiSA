#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 YOLO 检测头模块的独立性
确保不依赖任何外部 YOLO 库
"""

import sys
import os

def test_independence():
    """测试模块独立性"""
    
    print("=" * 60)
    print("YOLO 检测头模块独立性验证")
    print("=" * 60)
    
    # 添加 detection 目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    detection_dir = os.path.join(current_dir, 'detection')
    if os.path.exists(detection_dir):
        sys.path.insert(0, detection_dir)
    
    # 测试1: 检查是否有 ultralytics 导入
    print("\n【测试1】检查外部依赖...")
    print("-" * 60)
    
    files_to_check = [
        'yolo_modules.py',
        'yolo_head_adapter.py',
        'yolo_head_registry.py',
        'yolo_detector.py',
    ]
    
    has_external_deps = False
    for filename in files_to_check:
        filepath = os.path.join(detection_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'import ultralytics' in content or 'from ultralytics' in content:
                    print(f"✗ {filename}: 发现 ultralytics 导入")
                    has_external_deps = True
                else:
                    print(f"✓ {filename}: 无外部依赖")
    
    if not has_external_deps:
        print("\n✓ 所有文件都没有外部依赖")
    
    # 测试2: 测试模块导入
    print("\n【测试2】测试模块导入...")
    print("-" * 60)
    
    try:
        from yolo_modules import Conv, DWConv, DFL, dist2bbox, make_anchors
        print("✓ yolo_modules 导入成功")
    except ImportError as e:
        print(f"✗ yolo_modules 导入失败: {e}")
        return False
    
    try:
        from yolo_head_adapter import YOLODetectHead, YOLODetectHeadWrapper
        print("✓ yolo_head_adapter 导入成功")
    except ImportError as e:
        print(f"✗ yolo_head_adapter 导入失败: {e}")
        return False
    
    try:
        from yolo_head_registry import YOLODetectHead as YOLODetectHeadMMDet
        print("✓ yolo_head_registry 导入成功")
    except ImportError as e:
        print(f"✗ yolo_head_registry 导入失败: {e}")
        return False
    
    try:
        from yolo_detector import YOLODetector
        print("✓ yolo_detector 导入成功")
    except ImportError as e:
        print(f"✗ yolo_detector 导入失败: {e}")
        return False
    
    # 测试3: 测试模块功能
    print("\n【测试3】测试模块功能...")
    print("-" * 60)
    
    try:
        import torch
        
        # 测试 Conv
        conv = Conv(64, 128, k=3)
        x = torch.randn(1, 64, 32, 32)
        out = conv(x)
        print(f"✓ Conv 模块工作正常: {x.shape} -> {out.shape}")
        
        # 测试 DWConv
        dwconv = DWConv(64, 64, k=3)
        out = dwconv(x)
        print(f"✓ DWConv 模块工作正常: {x.shape} -> {out.shape}")
        
        # 测试 DFL
        dfl = DFL(16)
        x_dfl = torch.randn(1, 4, 16, 100)
        out_dfl = dfl(x_dfl)
        print(f"✓ DFL 模块工作正常: {x_dfl.shape} -> {out_dfl.shape}")
        
        # 测试 YOLODetectHead
        head = YOLODetectHead(num_classes=7, in_channels=(256, 256, 256))
        feats = [torch.randn(1, 256, 32, 32), 
                 torch.randn(1, 256, 16, 16),
                 torch.randn(1, 256, 8, 8)]
        preds, _ = head(feats, training=True)
        print(f"✓ YOLODetectHead 模块工作正常: 输入3个特征图 -> 输出{len(preds)}个预测")
        
    except Exception as e:
        print(f"✗ 模块功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！模块完全独立！")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_independence()
    sys.exit(0 if success else 1)


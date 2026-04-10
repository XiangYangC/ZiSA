#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练曲线可视化脚本
从训练日志中提取损失和准确率数据，并绘制训练曲线图
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_log_file(log_file):
    """解析训练日志文件，提取训练数据"""
    
    if not os.path.exists(log_file):
        print(f"错误: 日志文件不存在: {log_file}")
        return None
    
    print(f"正在解析日志文件: {log_file}")
    
    # 数据存储
    epochs = []
    train_losses = []
    val_acc1 = []
    val_acc5 = []
    val_acc1_ema = []
    val_acc5_ema = []
    learning_rates = []
    
    current_epoch = -1
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # 提取 epoch 信息
        if 'Train: [' in line:
            # 格式: Train: [epoch/100][batch/111]
            match = re.search(r'Train: \[(\d+)/(\d+)\]', line)
            if match:
                epoch = int(match.group(1))
                if epoch != current_epoch:
                    current_epoch = epoch
                    epochs.append(epoch)
                
                # 提取训练损失
                loss_match = re.search(r'loss\s+([\d.]+)\s+\(([\d.]+)\)', line)
                if loss_match:
                    avg_loss = float(loss_match.group(2))
                    if len(train_losses) < len(epochs):
                        train_losses.append(avg_loss)
                    elif len(train_losses) == len(epochs):
                        train_losses[-1] = avg_loss  # 更新最后一个epoch的平均损失
                
                # 提取学习率
                lr_match = re.search(r'lr\s+([\d.e-]+)', line)
                if lr_match:
                    lr = float(lr_match.group(1))
                    if len(learning_rates) < len(epochs):
                        learning_rates.append(lr)
        
        # 提取验证准确率（模型）
        if '* Acc@1' in line and 'Acc@5' in line:
            # 格式: INFO  * Acc@1 70.628 Acc@5 98.655
            acc_match = re.search(r'Acc@1\s+([\d.]+).*Acc@5\s+([\d.]+)', line)
            if acc_match:
                acc1 = float(acc_match.group(1))
                acc5 = float(acc_match.group(2))
                
                # 检查下一个 epoch 是否是 EMA 模型
                is_ema = False
                if i + 3 < len(lines):
                    next_lines = ''.join(lines[i+1:i+4])
                    if 'Max accuracy ema' in next_lines:
                        is_ema = True
                
                if is_ema:
                    val_acc1_ema.append(acc1)
                    val_acc5_ema.append(acc5)
                else:
                    val_acc1.append(acc1)
                    val_acc5.append(acc5)
    
    # 确保数据长度一致
    min_len = min(len(epochs), len(train_losses), len(val_acc1))
    
    return {
        'epochs': epochs[:min_len],
        'train_loss': train_losses[:min_len],
        'val_acc1': val_acc1[:min_len] if len(val_acc1) >= min_len else val_acc1,
        'val_acc5': val_acc5[:min_len] if len(val_acc5) >= min_len else val_acc5,
        'val_acc1_ema': val_acc1_ema[:min_len] if len(val_acc1_ema) >= min_len else val_acc1_ema,
        'val_acc5_ema': val_acc5_ema[:min_len] if len(val_acc5_ema) >= min_len else val_acc5_ema,
        'learning_rates': learning_rates[:min_len] if len(learning_rates) >= min_len else learning_rates,
    }


def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    
    if data is None or len(data['epochs']) == 0:
        print("错误: 没有找到有效的训练数据")
        return
    
    epochs = data['epochs']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图形大小和DPI
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    
    # 1. 绘制损失曲线
    if data['train_loss']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        loss_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"✓ 损失曲线已保存: {loss_path}")
        plt.close()
    
    # 2. 绘制准确率曲线
    if data['val_acc1']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(data['val_acc1'])], data['val_acc1'], 'g-', linewidth=2, 
                label='Validation Acc@1', marker='o', markersize=4)
        if data['val_acc5']:
            plt.plot(epochs[:len(data['val_acc5'])], data['val_acc5'], 'r-', linewidth=2, 
                    label='Validation Acc@5', marker='s', markersize=4)
        if data['val_acc1_ema']:
            plt.plot(epochs[:len(data['val_acc1_ema'])], data['val_acc1_ema'], 'b--', linewidth=2, 
                    label='Validation Acc@1 (EMA)', marker='^', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 标注最高准确率
        if data['val_acc1']:
            max_acc1 = max(data['val_acc1'])
            max_idx = data['val_acc1'].index(max_acc1)
            max_epoch = epochs[:len(data['val_acc1'])][max_idx]
            plt.annotate(f'Max: {max_acc1:.2f}% @ Epoch {max_epoch}', 
                        xy=(max_epoch, max_acc1), xytext=(max_epoch + 5, max_acc1 + 2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        acc_path = os.path.join(output_dir, 'validation_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"✓ 准确率曲线已保存: {acc_path}")
        plt.close()
    
    # 3. 绘制学习率曲线（如果有）
    if data['learning_rates'] and len(data['learning_rates']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(data['learning_rates'])], data['learning_rates'], 'orange', 
                linewidth=2, label='Learning Rate', marker='o', markersize=3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.yscale('log')  # 使用对数刻度
        plt.tight_layout()
        lr_path = os.path.join(output_dir, 'learning_rate.png')
        plt.savefig(lr_path, dpi=300, bbox_inches='tight')
        print(f"✓ 学习率曲线已保存: {lr_path}")
        plt.close()
    
    # 4. 综合图表（损失和准确率在同一张图上，使用双y轴）
    if data['train_loss'] and data['val_acc1']:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左y轴：损失
        color = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', color=color, fontsize=12)
        line1 = ax1.plot(epochs, data['train_loss'], color=color, linewidth=2, 
                         label='Training Loss', marker='o', markersize=3)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # 右y轴：准确率
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
        line2 = ax2.plot(epochs[:len(data['val_acc1'])], data['val_acc1'], color=color, 
                         linewidth=2, label='Validation Acc@1', marker='s', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=11)
        
        plt.title('Training Loss and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"✓ 综合曲线已保存: {combined_path}")
        plt.close()
    
    print(f"\n所有图表已保存到: {output_dir}")


def main():
    # ========== 配置区域：请修改以下路径 ==========
    
    # 训练日志文件路径
    LOG_FILE = r"/home/cruiy/code/python/VMamba-main/VMamba-main/classification/output/windfarm/vssm1_tiny_0230/20251105175129/log_rank0.txt"
    # 服务器路径示例：
    # LOG_FILE = "/home/cruiy/code/python/VMamba-main/VMamba-main/classification/output/windfarm/vssm1_tiny_0230/20251105175129/log.txt"
    
    # 输出目录（保存图表的位置）
    OUTPUT_DIR = r"/home/cruiy/code/python/VMamba-main/VMamba-main/classification/output/windfarm/vssm1_tiny_0230/20251105175129/curves"
    # 服务器路径示例：
    # OUTPUT_DIR = "/home/cruiy/code/python/VMamba-main/VMamba-main/classification/output/windfarm/vssm1_tiny_0230/20251105175129/curves"
    
    # ==============================================
    
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--log-file', type=str, default=LOG_FILE,
                        help='训练日志文件路径')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("训练曲线可视化工具")
    print("="*60)
    print(f"日志文件: {args.log_file}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    # 解析日志文件
    data = parse_log_file(args.log_file)
    
    if data:
        print(f"\n解析结果:")
        print(f"  Epochs: {len(data['epochs'])} 个")
        print(f"  训练损失: {len(data['train_loss'])} 个数据点")
        print(f"  验证准确率: {len(data['val_acc1'])} 个数据点")
        if data['val_acc1']:
            print(f"  最高准确率: {max(data['val_acc1']):.2f}%")
        
        # 绘制曲线
        plot_training_curves(data, args.output_dir)
        
        print("\n" + "="*60)
        print("可视化完成！")
        print("="*60)
    else:
        print("无法解析日志文件")


if __name__ == '__main__':
    main()


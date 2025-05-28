#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def main():
    parser = argparse.ArgumentParser(description='可视化食物分类评估结果')
    parser.add_argument('--results', type=str, default='results.json', help='评估结果JSON文件路径')
    parser.add_argument('--output', type=str, default='visualizations', help='可视化结果输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载评估结果
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取结果数据
    accuracy = results['accuracy']
    total_samples = results['total_samples']
    correct_predictions = results['correct_predictions']
    detailed_results = results['detailed_results']
    
    # 准备混淆矩阵数据
    y_true = []
    y_pred = []
    
    for item in detailed_results:
        y_true.append(item['ground_truth'])
        y_pred.append(item['predicted'] if item['predicted'] else '未识别')
    
    # 统计各类别样本数量
    class_counts = {}
    for label in y_true:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # 计算各类别准确率
    class_accuracy = {}
    for label in set(y_true):
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_true[i] == y_pred[i])
        class_accuracy[label] = correct / class_counts[label]
    
    # 生成混淆矩阵
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 生成分类报告
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # 打印基本统计信息
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"总体准确率: {accuracy:.2%}")
    print("\n各类别准确率:")
    for label, acc in class_accuracy.items():
        print(f"{label}: {acc:.2%} ({class_counts[label]}个样本)")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'confusion_matrix.png'), dpi=300)
    
    # 可视化准确率
    plt.figure(figsize=(10, 6))
    
    # 总体准确率条形图
    plt.subplot(1, 2, 1)
    plt.bar(['总体准确率'], [accuracy], color='blue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('准确率')
    plt.title('总体准确率')
    
    # 各类别准确率条形图
    plt.subplot(1, 2, 2)
    classes = list(class_accuracy.keys())
    accs = [class_accuracy[c] for c in classes]
    plt.bar(classes, accs, color=['red', 'green'], alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('准确率')
    plt.title('各类别准确率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'accuracy.png'), dpi=300)
    
    # 可视化样本分布
    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color=['red', 'green'], alpha=0.7)
    plt.ylabel('样本数量')
    plt.title('测试集样本分布')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'sample_distribution.png'), dpi=300)
    
    # 可视化预测结果统计
    prediction_counts = {}
    for pred in y_pred:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    plt.figure(figsize=(8, 6))
    plt.bar(prediction_counts.keys(), prediction_counts.values(), color=['red', 'green', 'gray'], alpha=0.7)
    plt.ylabel('样本数量')
    plt.title('预测结果分布')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'prediction_distribution.png'), dpi=300)
    
    # 保存详细分类报告
    with open(os.path.join(args.output, 'classification_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n可视化结果已保存至 {args.output} 目录")

if __name__ == "__main__":
    main() 
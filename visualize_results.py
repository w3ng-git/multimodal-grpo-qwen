#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def load_results(result_file):
    """加载评估结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def create_metrics_bar_chart(metrics, output_file):
    """创建指标条形图"""
    # 提取指标
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    # 创建条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # 添加标签和标题
    plt.title('Food Classification Model Evaluation Metrics', fontsize=16)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在条形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_confusion_matrix_heatmap(metrics, output_file):
    """创建混淆矩阵热力图"""
    # 创建混淆矩阵
    confusion_matrix = np.array([
        [metrics['true_positives'], metrics['false_negatives']],
        [metrics['false_positives'], metrics['true_negatives']]
    ])
    
    # 类别标签
    class_names = ['Dark Cuisine', 'Normal Food']
    
    # 创建热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    # 添加刻度标记
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # 添加数值标签
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, f'{confusion_matrix[i, j]}',
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black",
                    fontsize=15)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_classification_pie_chart(metrics, output_file):
    """创建分类结果饼图"""
    # 提取正确分类和错误分类的样本数
    correct = metrics['true_positives'] + metrics['true_negatives']
    incorrect = metrics['false_positives'] + metrics['false_negatives']
    
    # 创建饼图
    plt.figure(figsize=(10, 7))
    
    # 分类结果饼图
    plt.subplot(1, 2, 1)
    plt.pie([correct, incorrect], 
            labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            explode=(0.05, 0),
            startangle=90,
            shadow=True)
    plt.title('Classification Results Ratio', fontsize=16)
    
    # 混淆矩阵细分饼图
    plt.subplot(1, 2, 2)
    confusion_stats = [
        metrics['true_positives'],   # TP
        metrics['false_negatives'],  # FN
        metrics['false_positives'],  # FP
        metrics['true_negatives']    # TN
    ]
    labels = [
        f'True Positive (TP): {metrics["true_positives"]}',
        f'False Negative (FN): {metrics["false_negatives"]}',
        f'False Positive (FP): {metrics["false_positives"]}',
        f'True Negative (TN): {metrics["true_negatives"]}'
    ]
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    plt.pie(confusion_stats, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True)
    plt.title('Confusion Matrix Breakdown', fontsize=16)
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_combined_visualization(metrics, output_dir):
    """创建综合可视化图"""
    # 创建画布
    plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 准确率、精确率、召回率和F1分数的条形图
    ax1 = plt.subplot(gs[0, 0])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    bars = ax1.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax1.set_title('Evaluation Metrics', fontsize=16)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    # 分类结果饼图
    ax2 = plt.subplot(gs[0, 1])
    correct = metrics['true_positives'] + metrics['true_negatives']
    incorrect = metrics['false_positives'] + metrics['false_negatives']
    ax2.pie([correct, incorrect], 
            labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            explode=(0.05, 0),
            startangle=90,
            shadow=True)
    ax2.set_title('Classification Results Ratio', fontsize=16)
    
    # 混淆矩阵热力图
    ax3 = plt.subplot(gs[1, :])
    confusion_matrix = np.array([
        [metrics['true_positives'], metrics['false_negatives']],
        [metrics['false_positives'], metrics['true_negatives']]
    ])
    class_names = ['Dark Cuisine', 'Normal Food']
    im = ax3.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion Matrix', fontsize=16)
    plt.colorbar(im, ax=ax3)
    tick_marks = np.arange(len(class_names))
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(class_names, rotation=45, fontsize=12)
    ax3.set_yticklabels(class_names, fontsize=12)
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax3.text(j, i, f'{confusion_matrix[i, j]}',
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black",
                    fontsize=15)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    
    # 添加总标题
    plt.suptitle('Multimodal Food Classification Model Evaluation Results', fontsize=20, y=0.98)
    
    # 保存和显示
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig(os.path.join(output_dir, 'combined_visualization.png'), dpi=300)
    plt.close()

def main():
    # 配置参数
    result_file = 'evaluation_results/evaluation_results.json'
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 或者直接使用用户共享的结果数据
    metrics = {
        'accuracy': 0.5154,
        'precision': 0.4344,
        'recall': 0.2103,
        'f1': 0.2834,
        'true_positives': 53,
        'false_positives': 69,
        'true_negatives': 232,
        'false_negatives': 199
    }
    
    # 尝试加载结果文件，如果存在的话
    try:
        results = load_results(result_file)
        metrics = results['metrics']
        print(f"从 {result_file} 加载评估结果")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"无法从 {result_file} 加载结果，使用预设数据")
    
    # 创建各种可视化
    create_metrics_bar_chart(metrics, os.path.join(output_dir, 'metrics_bar_chart.png'))
    create_confusion_matrix_heatmap(metrics, os.path.join(output_dir, 'confusion_matrix.png'))
    create_classification_pie_chart(metrics, os.path.join(output_dir, 'classification_pie_chart.png'))
    create_combined_visualization(metrics, output_dir)
    
    print(f"可视化图表已保存到 {output_dir} 目录")

if __name__ == '__main__':
    main() 
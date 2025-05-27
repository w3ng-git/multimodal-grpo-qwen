#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import matplotlib.pyplot as plt

def load_results(result_file):
    """加载评估结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def show_metrics(results):
    """显示评估指标"""
    metrics = results.get('metrics', {})
    if not metrics:
        print("没有找到有效的评估指标")
        return
    
    print("\n===== 食物分类评估指标 =====")
    print(f"准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"精确率: {metrics.get('precision', 0):.4f}")
    print(f"召回率: {metrics.get('recall', 0):.4f}")
    print(f"F1分数: {metrics.get('f1', 0):.4f}")
    print("\n===== 混淆矩阵 =====")
    print(f"真阳性 (真实黑暗料理，预测黑暗料理): {metrics.get('true_positives', 0)}")
    print(f"假阳性 (真实正常食物，预测黑暗料理): {metrics.get('false_positives', 0)}")
    print(f"真阴性 (真实正常食物，预测正常食物): {metrics.get('true_negatives', 0)}")
    print(f"假阴性 (真实黑暗料理，预测正常食物): {metrics.get('false_negatives', 0)}")
    
    # 统计总体情况
    predictions = results.get('predictions', [])
    error_samples = results.get('error_samples', [])
    
    print(f"\n总样本数: {len(predictions)}")
    print(f"正确分类样本数: {len(predictions) - len(error_samples)}")
    print(f"错误分类样本数: {len(error_samples)}")

def show_results_pie_chart(results, output_dir):
    """生成简单的饼图显示结果"""
    metrics = results.get('metrics', {})
    
    # 提取正确和错误的样本数
    tp = metrics.get('true_positives', 0) 
    tn = metrics.get('true_negatives', 0)
    fp = metrics.get('false_positives', 0)
    fn = metrics.get('false_negatives', 0)
    
    # 创建饼图
    labels = ['真阳性\n(真黑暗料理，预测黑暗料理)', 
              '真阴性\n(真正常食物，预测正常食物)',
              '假阳性\n(真正常食物，预测黑暗料理)', 
              '假阴性\n(真黑暗料理，预测正常食物)']
    sizes = [tp, tn, fp, fn]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    explode = (0.1, 0.1, 0.1, 0.1)  # 突出显示所有部分
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('食物分类结果', fontsize=18)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'classification_results_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 指标条形图
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_values = [metrics.get('accuracy', 0), metrics.get('precision', 0), 
                      metrics.get('recall', 0), metrics.get('f1', 0)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # 在条形上添加标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.0)
    plt.title('模型评估指标', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'metrics_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存到 {output_dir} 目录")

def main():
    parser = argparse.ArgumentParser(description="显示食物分类评估指标")
    parser.add_argument("--result_file", type=str, default="evaluation_results_6/evaluation_results.json", 
                       help="评估结果文件路径")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_6", 
                       help="输出目录，用于保存图表")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    try:
        results = load_results(args.result_file)
        # 显示指标
        show_metrics(results)
        # 生成可视化图表
        show_results_pie_chart(results, args.output_dir)
    except Exception as e:
        print(f"处理评估结果时出错: {e}")

if __name__ == "__main__":
    main() 
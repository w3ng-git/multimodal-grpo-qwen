#!/usr/bin/env python
# filepath: /home/cpns1107/Qwen/generate_food_dataset.py
"""
简化的食物分类数据集生成器
输出一个简单的JSON文件，每个样本包含一张图片和对应的对话
"""

import os
import json
import glob

def generate_food_dataset(output_file="food_dataset.json"):
    """生成简单的食物分类数据集JSON文件"""
    print("正在生成食物分类数据集...")
    
    # 数据集根目录
    dataset_root = "."
    
    # 收集所有图像路径和标签
    samples = []
    label_names = ["bad_food", "good_food"]  # 固定标签顺序: 0=黑暗料理, 1=正常食物
    
    for label_id, folder in enumerate(label_names):
        folder_path = os.path.join(dataset_root, "train", folder)
        # 获取所有图片文件
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        files.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
        files.extend(glob.glob(os.path.join(folder_path, "*.png")))
        
        answer = "正常食物" if label_names[label_id] == "good_food" else "黑暗料理"
        
        # 为每张图片创建一个样本
        for img_path in files:
            sample = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", 
                         "text": "请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                                "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                                "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>"}
                    ]
                }],
                "images": [img_path],
                "solution": f"<answer>{answer}</answer>"
            }
            samples.append(sample)
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"数据集生成完成! 共{len(samples)}个样本")
    print(f"数据集已保存到: {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)

if __name__ == "__main__":
    generate_food_dataset()

#!/usr/bin/env python
# filepath: /home/cpns1107/Qwen/test_food_model.py
"""
简单的食物分类模型测试脚本
用于测试训练后的模型在测试集上的表现
"""

import os
import glob
import random
import argparse
from PIL import Image
import requests
from io import BytesIO

def test_model(model_path="output/swift_grpo_food", test_folder="test", num_samples=5):
    """测试训练好的模型在随机测试样本上的表现"""
    print(f"测试模型: {model_path}")
    
    # 收集测试图像
    test_images = []
    label_names = ["bad_food", "good_food"]
    
    for folder in label_names:
        folder_path = os.path.join(test_folder, folder)
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        files.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
        files.extend(glob.glob(os.path.join(folder_path, "*.png")))
        
        for img_path in files:
            test_images.append({
                "path": img_path,
                "label": "正常食物" if folder == "good_food" else "黑暗料理"
            })
    
    # 随机选择样本
    if num_samples > len(test_images):
        num_samples = len(test_images)
    
    test_samples = random.sample(test_images, num_samples)
    
    # 设置vLLM API地址
    api_url = "http://127.0.0.1:8000/v1/chat/completions"
    
    # 测试结果统计
    correct = 0
    format_correct = 0
    total = len(test_samples)
    
    print(f"\n开始测试 {total} 个随机样本...")
    
    for i, sample in enumerate(test_samples):
        img_path = sample["path"]
        true_label = sample["label"]
        
        # 读取图像并编码为base64
        with open(img_path, "rb") as img_file:
            image_data = img_file.read()
        
        # 构造请求
        payload = {
            "model": "vl-3b-inst",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data.hex()}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                                   "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                                   "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>"
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            # 提取答案
            import re
            answer_match = re.search(r"<answer>(黑暗料理|正常食物)</answer>", generated_text)
            
            # 检查格式
            format_ok = (
                "<analyse>" in generated_text and 
                "</analyse>" in generated_text and 
                "<comment>" in generated_text and 
                "</comment>" in generated_text and
                answer_match is not None
            )
            
            if format_ok:
                format_correct += 1
                predicted_label = answer_match.group(1)
                is_correct = predicted_label == true_label
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                print(f"[{i+1}/{total}] {os.path.basename(img_path)} - 真实: {true_label}, 预测: {predicted_label} {status}")
            else:
                print(f"[{i+1}/{total}] {os.path.basename(img_path)} - 格式错误 ✗")
                print(f"生成内容: {generated_text[:100]}...")
            
        except Exception as e:
            print(f"[{i+1}/{total}] 处理 {img_path} 时出错: {str(e)}")
    
    # 输出总体结果
    accuracy = (correct / total) * 100 if total > 0 else 0
    format_rate = (format_correct / total) * 100 if total > 0 else 0
    
    print("\n测试结果汇总:")
    print(f"总样本数: {total}")
    print(f"格式正确率: {format_rate:.2f}% ({format_correct}/{total})")
    print(f"分类准确率: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试食物分类模型")
    parser.add_argument("--model", default="output/swift_grpo_food", help="模型路径")
    parser.add_argument("--test_folder", default="test", help="测试数据集文件夹")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    
    args = parser.parse_args()
    test_model(args.model, args.test_folder, args.num_samples)

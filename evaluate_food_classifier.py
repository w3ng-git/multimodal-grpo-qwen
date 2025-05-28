#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from typing import List, Dict
from tqdm import tqdm

from swift.utils import get_logger, seed_everything

logger = get_logger()

def extract_answer(response: str) -> str:
    """从模型响应中提取答案"""
    # 首先尝试标准的<answer>标签提取
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        if answer in ["黑暗料理", "正常食物"]:
            return answer
    
    # 宽松策略：查找文本中所有出现的"黑暗料理"或"正常食物"，取最后一个
    bad_food_matches = [m.start() for m in re.finditer("黑暗料理", response)]
    good_food_matches = [m.start() for m in re.finditer("正常食物", response)]
    
    # 如果都没找到，返回空字符串
    if not bad_food_matches and not good_food_matches:
        return ""
    
    # 如果只找到一种类型，返回该类型
    if not bad_food_matches:
        return "正常食物"
    if not good_food_matches:
        return "黑暗料理"
    
    # 如果两种都找到，返回位置靠后的那个
    last_bad = max(bad_food_matches)
    last_good = max(good_food_matches)
    
    return "黑暗料理" if last_bad > last_good else "正常食物"

def prepare_custom_qa_dataset():
    """准备用于评估的自定义问答数据集"""
    # 创建评估目录
    eval_dir = "food_qa_dataset"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 获取测试集图片
    test_data = []
    
    # 加载黑暗料理测试图片
    bad_food_images = sorted(glob.glob("test/bad_food/*.jpg"))
    for img_path in bad_food_images:
        test_data.append({
            "query": f"请分析这张食物图片 <image>{img_path}</image>",
            "response": "<answer>黑暗料理</answer>"
        })
    
    # 加载正常食物测试图片
    good_food_images = sorted(glob.glob("test/good_food/*.jpg"))
    for img_path in good_food_images:
        test_data.append({
            "query": f"请分析这张食物图片 <image>{img_path}</image>",
            "response": "<answer>正常食物</answer>"
        })
    
    # 写入jsonl文件
    with open(f"{eval_dir}/food_test.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"已创建评估数据集，共 {len(test_data)} 个样本")
    return eval_dir

def main():
    parser = argparse.ArgumentParser(description="评估食物分类模型")
    parser.add_argument("--model", type=str, default="output/GRPO_FoodClassifier/v0-20250528-003934/checkpoint-1150", 
                        help="模型路径")
    parser.add_argument("--gpu_ids", type=str, default="0", help="使用的GPU ID")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 准备评估数据集
    eval_dir = prepare_custom_qa_dataset()
    
    # 运行swift eval命令
    eval_cmd = f"""
    swift eval \\
        --model {args.model} \\
        --eval_backend Native \\
        --infer_backend pt \\
        --eval_dataset general_qa \\
        --dataset_args '{{"general_qa": {{"local_path": "{eval_dir}", "subset_list": ["food_test"]}}}}' \\
        --system "你是一位黑色幽默的美食评论家，擅长分析各种料理的特点和成分。\\n你需要分析图片中的食物，判断它是\\"黑暗料理\\"还是\\"正常食物\\"。\\n请提供详细的分析并给出你的判断。\\n必须使用以下格式回答：\\n<analyse>对食物的详细分析，包括可能的食材、烹饪方法等</analyse>\\n<comment>对这道菜的幽默评论</comment>\\n<answer>黑暗料理</answer> 或 <answer>正常食物</answer>" \\
        --extra_eval_args '{{"debug": true}}' \\
        --eval_output_dir {args.output_dir}
    """
    
    print("执行评估命令：")
    print(eval_cmd)
    os.system(eval_cmd)
    
    # 解析结果并生成自定义报告
    if os.path.exists(f"{args.output_dir}/general_qa"):
        # 读取Swift eval生成的结果
        with open(f"{args.output_dir}/general_qa/result.jsonl", "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]
        
        # 处理结果并计算准确率
        correct = 0
        processed_results = []
        
        for item in results:
            query = item.get("query", "")
            ground_truth = ""
            if "黑暗料理" in item.get("response", ""):
                ground_truth = "黑暗料理"
            elif "正常食物" in item.get("response", ""):
                ground_truth = "正常食物"
                
            predicted = extract_answer(item.get("output", ""))
            is_correct = predicted == ground_truth
            
            if is_correct:
                correct += 1
                
            # 获取图片路径
            img_match = re.search(r'<image>(.*?)</image>', query)
            img_path = img_match.group(1) if img_match else ""
                
            processed_results.append({
                "image_path": img_path,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "full_response": item.get("output", "")
            })
        
        # 计算准确率
        accuracy = correct / len(results) if results else 0
        
        # 生成报告
        report = {
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct_predictions": correct,
            "detailed_results": processed_results
        }
        
        # 保存报告
        with open(f"{args.output_dir}/custom_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"总样本数: {len(results)}")
        print(f"正确预测数: {correct}")
        print(f"准确率: {accuracy:.2%}")
        print(f"详细结果已保存至: {args.output_dir}/custom_report.json")

if __name__ == "__main__":
    seed_everything(42)
    main() 
#!/usr/bin/env python
import os
import re
import json
import glob
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_answer(response):
    # 标签提取
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        if answer in ["黑暗料理", "正常食物"]:
            return answer
    
    # 宽松策略
    bad_food_matches = [m.start() for m in re.finditer("黑暗料理", response)]
    good_food_matches = [m.start() for m in re.finditer("正常食物", response)]
    
    if not bad_food_matches and not good_food_matches:
        return ""
    if not bad_food_matches:
        return "正常食物"
    if not good_food_matches:
        return "黑暗料理"
    
    last_bad = max(bad_food_matches)
    last_good = max(good_food_matches)
    
    return "黑暗料理" if last_bad > last_good else "正常食物"

def main():
    parser = argparse.ArgumentParser(description="使用vLLM评估食物分类模型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()
    
    # 初始化模型
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 5}
    )
    
    # 加载数据
    test_data = []
    bad_food_images = sorted(glob.glob("test/bad_food/*.jpg"))
    good_food_images = sorted(glob.glob("test/good_food/*.jpg"))
    
    for img_path in bad_food_images:
        test_data.append({"image_path": img_path, "ground_truth": "黑暗料理"})
    for img_path in good_food_images:
        test_data.append({"image_path": img_path, "ground_truth": "正常食物"})
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(args.model)
    
    # 系统提示词
    system_prompt = """你是一位黑色幽默的美食评论家，擅长分析各种料理的特点和成分。
你需要分析图片中的食物，判断它是"黑暗料理"还是"正常食物"。
请提供详细的分析并给出你的判断。
必须使用以下格式回答：
<analyse>对食物的详细分析，包括可能的食材、烹饪方法等</analyse>
<comment>对这道菜的幽默评论</comment>
<answer>黑暗料理</answer> 或 <answer>正常食物</answer>"""
    
    # 评估
    correct = 0
    results = []
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512)
    
    for item in tqdm(test_data):
        # 构建输入
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": item["image_path"]},
                {"type": "text", "text": "请分析这张食物图片"}
            ]}
        ]
        
        # 预处理
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 构建输入
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data
        }
        
        # 生成回答
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # 提取答案
        predicted = extract_answer(generated_text)
        is_correct = predicted == item["ground_truth"]
        
        if is_correct:
            correct += 1
            
        results.append({
            "image_path": item["image_path"],
            "ground_truth": item["ground_truth"],
            "predicted": predicted,
            "is_correct": is_correct
        })
    
    # 计算准确率
    accuracy = correct / len(test_data) if test_data else 0
    
    # 生成报告
    report = {
        "accuracy": accuracy,
        "total_samples": len(test_data),
        "correct_predictions": correct,
        "detailed_results": results
    }
    
    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n评估结果:")
    print(f"总样本数: {len(test_data)}")
    print(f"正确预测数: {correct}")
    print(f"准确率: {accuracy:.2%}")

if __name__ == "__main__":
    main() 
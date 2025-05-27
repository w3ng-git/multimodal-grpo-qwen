#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# vLLM相关导入
from vllm import LLM, SamplingParams

def load_test_data(test_good_folder: str, test_bad_folder: str, limit_per_folder: int = None) -> Tuple[List[str], List[str]]:
    """加载测试数据集，返回图片路径和标签"""
    # 加载正常食物图片
    good_food_paths = [os.path.join(test_good_folder, f) for f in os.listdir(test_good_folder) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 加载黑暗料理图片
    bad_food_paths = [os.path.join(test_bad_folder, f) for f in os.listdir(test_bad_folder) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 如果设置了限制，则只取前N张图片
    if limit_per_folder is not None:
        good_food_paths = good_food_paths[:limit_per_folder]
        bad_food_paths = bad_food_paths[:limit_per_folder]
    
    # 创建标签
    labels = ["正常食物"] * len(good_food_paths) + ["黑暗料理"] * len(bad_food_paths)
    
    # 合并所有图片路径
    all_paths = good_food_paths + bad_food_paths
    
    return all_paths, labels

def create_prompts(image_paths: List[str]) -> List[Dict]:
    """为每张图片创建prompt，使用train_food_classifier.py中的格式"""
    prompts = []
    for img_path in image_paths:
        # 使用与训练时相同的prompt格式
        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},  # 直接使用图像路径
                {"type": "text",
                 "text": ("请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                          "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                          "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>")}
            ]
        }
        prompts.append([message])
    return prompts

def extract_answer(response: str) -> str:
    """从模型输出中提取答案，宽松策略：查找最后出现的'黑暗料理'或'正常食物'"""
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

def calculate_metrics(true_labels: List[str], predicted_labels: List[str]) -> Dict:
    """计算评估指标，纯Python实现"""
    # 转换为二进制标签
    y_true_bin = [1 if label == "黑暗料理" else 0 for label in true_labels]
    y_pred_bin = [1 if label == "黑暗料理" else 0 for label in predicted_labels]
    
    # 计算混淆矩阵
    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)  # 真阳性
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)  # 假阳性
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)  # 真阴性
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)  # 假阴性
    
    # 计算指标
    total = len(true_labels)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }

def verify_image_loading(image_path: str) -> Dict:
    """验证图片是否可以成功加载，并返回图片信息"""
    try:
        img = Image.open(image_path)
        result = {
            "success": True,
            "path": image_path,
            "size": img.size,
            "format": img.format,
            "mode": img.mode
        }
        img.close()
        return result
    except Exception as e:
        return {
            "success": False,
            "path": image_path,
            "error": str(e)
        }

def single_image_test(model_path: str, image_path: str, gpu_ids: str = "0") -> Dict:
    """对单张图片进行测试，显示详细的处理过程和结果"""
    print(f"\n==== 单图片验证模式 ====")
    print(f"图片路径: {image_path}")
    
    # 验证图片加载
    img_info = verify_image_loading(image_path)
    if not img_info["success"]:
        print(f"错误: 无法加载图片 - {img_info['error']}")
        return {"success": False, "error": img_info["error"]}
    
    print(f"图片信息: {img_info['size'][0]}x{img_info['size'][1]} {img_info['format']} {img_info['mode']}")
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # 创建消息
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},  # 直接使用图像路径
            {"type": "text",
             "text": ("请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                      "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                      "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>")}
        ]
    }
    
    # 打印提示信息
    print(f"\n提示词:\n{'-'*40}\n{message}\n{'-'*40}")
    
    # 从AutoProcessor导入
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    # 初始化处理器
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 处理输入
    prompt = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    
    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info([message])
    
    # 初始化模型
    print(f"\n初始化vLLM模型 {model_path}...")
    start_time = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(gpu_ids.split(",")),
        max_model_len=2048,  # 增加最大模型长度至2048
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"模型加载耗时: {time.time() - start_time:.2f}秒")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=1024,
        repetition_penalty=1.1,
    )
    
    # 构建vLLM输入
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
    
    # 执行推理
    print(f"开始推理...")
    start_time = time.time()
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    response = outputs[0].outputs[0].text
    inference_time = time.time() - start_time
    
    # 提取答案
    answer = extract_answer(response)
    
    # 打印结果
    print(f"\n推理耗时: {inference_time:.2f}秒")
    print(f"\n模型输出:\n{'-'*40}\n{response}\n{'-'*40}")
    print(f"\n提取的答案: {answer}")
    
    # 返回结果
    return {
        "success": True,
        "image": img_info,
        "prompt": prompt,
        "response": response,
        "answer": answer,
        "inference_time": inference_time
    }

def main():
    parser = argparse.ArgumentParser(description="评估多模态食物分类器")
    parser.add_argument("--model", type=str, default="vl-3b-inst", help="模型路径或名称")
    parser.add_argument("--good_food_dir", type=str, default="test/good_food", help="正常食物图片目录")
    parser.add_argument("--bad_food_dir", type=str, default="test/bad_food", help="黑暗料理图片目录")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="使用的GPU ID，逗号分隔")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--max_tokens", type=int, default=1024, help="最大生成token数")
    parser.add_argument("--verify_mode", action="store_true", help="启用单图片验证模式")
    parser.add_argument("--verify_image", type=str, help="用于验证的单张图片路径，不指定则使用第一张测试图片")
    parser.add_argument("--limit_per_folder", type=int, default=10, help="每个文件夹处理的图片数量限制，默认10张")
    parser.add_argument("--max_model_len", type=int, default=2048, help="模型最大上下文长度，默认2048")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，打印更多信息")
    args = parser.parse_args()
    
    # 调试输出
    debug = args.debug
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 单图片验证模式
    if args.verify_mode:
        if args.verify_image:
            verify_image = args.verify_image
        else:
            # 加载第一张测试图片
            print("未指定验证图片，使用测试集中的第一张图片")
            images, _ = load_test_data(args.good_food_dir, args.bad_food_dir, limit_per_folder=1)
            verify_image = images[0] if images else None
            
        if verify_image:
            result = single_image_test(args.model, verify_image, args.gpu_ids.split(",")[0])
            # 保存验证结果
            with open(os.path.join(args.output_dir, "verify_result.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return
        else:
            print("错误: 未找到可用的测试图片")
            return
    
    # 加载测试数据，限制每个文件夹的图片数量
    print(f"加载测试数据，每个文件夹限制处理 {args.limit_per_folder} 张图片...")
    image_paths, true_labels = load_test_data(args.good_food_dir, args.bad_food_dir, limit_per_folder=args.limit_per_folder)
    print(f"共找到 {len(image_paths)} 张图片进行测试")
    
    if debug:
        print("测试图片路径:")
        for i, (path, label) in enumerate(zip(image_paths, true_labels)):
            print(f"{i+1}. {path} [{label}]")
    
    # 创建提示
    message_prompts = create_prompts(image_paths)
    print(f"已创建 {len(message_prompts)} 个提示")
    
    # 初始化LLM
    print(f"初始化vLLM模型 {args.model}...")
    
    # 从AutoProcessor导入
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 初始化处理器
    processor = AutoProcessor.from_pretrained(args.model)
    
    # 初始化LLM
    start_time = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=len(args.gpu_ids.split(",")),
        max_model_len=args.max_model_len,  # 使用命令行参数设置的最大模型长度
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # 避免cudagraph捕获错误
    )
    print(f"模型加载耗时: {time.time() - start_time:.2f}秒")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=args.max_tokens,
        repetition_penalty=1.1,
    )
    
    # 执行批量推理
    print(f"开始执行推理...")
    all_outputs = []
    
    # 分批处理以避免OOM
    for i in range(0, len(message_prompts), args.batch_size):
        print(f"\n==== 处理批次 {i//args.batch_size + 1} ====")
        batch_prompts = message_prompts[i:i+args.batch_size]
        batch_image_paths = image_paths[i:i+args.batch_size]
        
        if debug:
            print(f"批次大小: {len(batch_prompts)}")
            print(f"批次图片: {batch_image_paths}")
        
        batch_llm_inputs = []
        
        # 处理每一批次的输入
        for j, prompt_messages in enumerate(batch_prompts):
            try:
                # 处理输入
                if debug:
                    print(f"处理第 {i+j+1} 张图片: {batch_image_paths[j]}")
                
                prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                # 处理视觉信息
                image_inputs, video_inputs = process_vision_info(prompt_messages)
                
                # 构建vLLM输入
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data
                }
                
                batch_llm_inputs.append(llm_inputs)
                if debug:
                    print(f"图片 {batch_image_paths[j]} 处理成功")
            except Exception as e:
                print(f"处理图片 {batch_image_paths[j]} 时出错: {e}")
                # 添加一个默认回答
                result = {
                    "image_path": batch_image_paths[j],
                    "true_label": true_labels[i+j],
                    "response": "处理错误",
                    "predicted_label": ""
                }
                all_outputs.append(result)
                if debug:
                    import traceback
                    traceback.print_exc()
        
        if not batch_llm_inputs:
            print(f"批次 {i} 中没有有效的输入，跳过")
            continue
            
        try:
            # 使用generate方法
            if debug:
                print(f"开始生成，输入数量: {len(batch_llm_inputs)}")
            
            batch_outputs = llm.generate(batch_llm_inputs, sampling_params=sampling_params)
            
            if debug:
                print(f"生成完成，输出数量: {len(batch_outputs)}")
            
            for j, output in enumerate(batch_outputs):
                current_idx = i + j
                if current_idx >= len(true_labels):
                    if debug:
                        print(f"索引 {current_idx} 超出 true_labels 范围 {len(true_labels)}")
                    continue
                    
                response = output.outputs[0].text
                answer = extract_answer(response)
                
                if debug:
                    print(f"图片 {batch_image_paths[j]} 的分类结果: {answer}")
                
                result = {
                    "image_path": batch_image_paths[j],
                    "true_label": true_labels[current_idx],
                    "response": response,
                    "predicted_label": answer
                }
                all_outputs.append(result)
                
                # 对第一张图片进行详细输出
                if i == 0 and j == 0:
                    print(f"\n==== 第一张图片的推理结果 ====")
                    print(f"图片路径: {batch_image_paths[j]}")
                    print(f"真实标签: {true_labels[current_idx]}")
                    print(f"模型输出:\n{'-'*40}\n{response}\n{'-'*40}")
                    print(f"提取的答案: {answer}")
                    
                    # 保存第一张图片的详细结果
                    first_image_result = {
                        "image_path": batch_image_paths[j],
                        "true_label": true_labels[current_idx],
                        "prompt": batch_llm_inputs[j]["prompt"],
                        "response": response,
                        "predicted_label": answer
                    }
                    with open(os.path.join(args.output_dir, "first_image_result.json"), "w", encoding="utf-8") as f:
                        json.dump(first_image_result, f, ensure_ascii=False, indent=2)
        except ValueError as e:
            # 处理输入过长的错误
            error_msg = str(e)
            print(f"错误: {error_msg}")
            if "longer than the maximum model length" in error_msg:
                print(f"批次 {i} 的输入长度超过模型最大长度，跳过并记录错误")
                for j in range(len(batch_prompts)):
                    current_idx = i + j
                    if current_idx >= len(true_labels):
                        continue
                        
                    result = {
                        "image_path": batch_image_paths[j],
                        "true_label": true_labels[current_idx],
                        "response": "输入长度超过模型限制",
                        "predicted_label": ""
                    }
                    all_outputs.append(result)
            else:
                if debug:
                    import traceback
                    traceback.print_exc()
                raise e
        except Exception as e:
            print(f"处理批次 {i} 时出错: {e}")
            # 记录错误但继续处理下一批
            for j in range(len(batch_prompts)):
                current_idx = i + j
                if current_idx >= len(true_labels):
                    continue
                    
                result = {
                    "image_path": batch_image_paths[j],
                    "true_label": true_labels[current_idx],
                    "response": f"处理错误: {str(e)}",
                    "predicted_label": ""
                }
                all_outputs.append(result)
                
            if debug:
                import traceback
                traceback.print_exc()
            
        # 显示进度
        print(f"已处理 {min(i + len(batch_prompts), len(message_prompts))}/{len(message_prompts)} 张图片")
        
        # 每处理完一批，保存当前结果，避免中途中断导致全部丢失
        temp_result_file = os.path.join(args.output_dir, "temp_results.json")
        with open(temp_result_file, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            
        if debug:
            print(f"临时结果已保存到 {temp_result_file}，当前结果数: {len(all_outputs)}")
            
        # 暂停一下，避免短时间内多次请求
        time.sleep(1)
    
    print(f"推理完成，共处理 {len(all_outputs)} 个样本")
    
    # 提取有效的预测标签
    valid_outputs = [(output, label) for output, label in zip(all_outputs, true_labels) if output.get("predicted_label") in ["黑暗料理", "正常食物"]]
    
    if debug:
        print(f"有效输出数: {len(valid_outputs)}")
    
    if not valid_outputs:
        print("没有有效的预测结果，无法计算评估指标")
        # 保存结果
        result_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {},
                "predictions": all_outputs,
                "error_samples": []
            }, f, ensure_ascii=False, indent=2)
        return
    
    predicted_labels = []
    filtered_true_labels = []
    
    for output, true_label in valid_outputs:
        predicted_label = output["predicted_label"]
        if not predicted_label or (predicted_label != "黑暗料理" and predicted_label != "正常食物"):
            # 如果包含"黑暗"或"恐怖"等词汇，则判定为黑暗料理
            if "黑暗" in output["response"] or "恐怖" in output["response"]:
                predicted_label = "黑暗料理"
            else:
                predicted_label = "正常食物"  # 默认判定为正常食物
        
        predicted_labels.append(predicted_label)
        filtered_true_labels.append(true_label)
    
    # 计算评估指标
    metrics = calculate_metrics(filtered_true_labels, predicted_labels)
    
    # 统计错误分类的样本
    error_samples = []
    for i, (true, pred) in enumerate(zip(filtered_true_labels, predicted_labels)):
        if true != pred:
            error_samples.append({
                "image_path": valid_outputs[i][0]["image_path"],
                "true_label": true,
                "predicted_label": pred
            })
    
    # 保存结果
    result_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "predictions": all_outputs,
            "error_samples": error_samples
        }, f, ensure_ascii=False, indent=2)
    
    # 打印评估指标
    print("\n评估指标:")
    print(f"有效样本数量: {len(filtered_true_labels)}")
    print(f"正确分类样本数: {metrics['true_positives'] + metrics['true_negatives']}")
    print(f"错误分类样本数: {metrics['false_positives'] + metrics['false_negatives']}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"混淆矩阵:")
    print(f"真阳性 (真实黑暗料理，预测黑暗料理): {metrics['true_positives']}")
    print(f"假阳性 (真实正常食物，预测黑暗料理): {metrics['false_positives']}")
    print(f"真阴性 (真实正常食物，预测正常食物): {metrics['true_negatives']}")
    print(f"假阴性 (真实黑暗料理，预测正常食物): {metrics['false_negatives']}")
    
    print(f"\n详细评估结果已保存到 {result_file}")
    
    # 调用可视化结果的函数
    print("\n生成可视化结果...")
    visualization_cmd = f"python visualize_results.py --result_file {result_file} --output_dir {args.output_dir}"
    os.system(visualization_cmd)
    print(f"可视化结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main() 
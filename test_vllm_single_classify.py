#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import PIL.Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def test_food_image_classification(image_path):
    """
    使用vLLM和Qwen2.5-VL模型测试单张食物图片的分类
    """
    print(f"\n==== 单图片食物分类测试 ====")
    print(f"图片路径: {image_path}")
    
    # 加载图像
    try:
        image = PIL.Image.open(image_path)
        print(f"成功加载图像: {image_path}")
        print(f"图像尺寸: {image.size}，格式: {image.format}, 模式: {image.mode}")
    except Exception as e:
        print(f"错误: 无法加载图片 - {e}")
        return
    
    # 构建消息格式 - 将系统提示移到用户信息中
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},  # 直接使用图像路径
            {"type": "text",
            "text": ("请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                    "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                    "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>")}
        ]
    }]
    
    # 初始化模型处理器
    print("\n初始化模型处理器...")
    processor = AutoProcessor.from_pretrained("vl-3b-inst")
    
    # 处理输入
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n使用的提示词模板:")
    print(f"{'-'*40}")
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    print(f"{'-'*40}")
    
    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    
    if image_inputs is None or len(image_inputs) == 0:
        print("警告：没有成功处理图像输入！")
        return
    else:
        print(f"成功处理图像输入，共 {len(image_inputs)} 个图像")
    
    # 初始化vLLM
    print("\n初始化vLLM模型...")
    try:
        llm = LLM(
            model="vl-3b-inst",
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": 5},
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"初始化模型失败: {e}")
        return
    
    # 构建多模态数据
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    
    # 推理参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )
    
    # 构建vLLM输入
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
    
    # 执行推理
    print("开始推理...")
    try:
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print("\n模型输出:")
        print(f"{'-'*40}")
        print(generated_text)
        print(f"{'-'*40}")
        
        # 提取答案
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            print(f"\n提取的分类结果: {answer}")
        else:
            print("\n无法从输出中提取分类结果")
            
    except Exception as e:
        print(f"推理过程出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用vLLM和Qwen2.5-VL测试单张食物图片分类")
    parser.add_argument("--image_path", type=str, default="test/bad_food/2726.jpg", 
                        help="要分类的食物图片路径")
    args = parser.parse_args()
    
    test_food_image_classification(args.image_path) 
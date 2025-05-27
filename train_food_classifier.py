import os
import re
import json
import glob
from typing import Dict, Any, List
import random

from swift.llm import (
    ResponsePreprocessor,
    register_dataset,
    DatasetMeta,
    SubsetDataset,
)
# 保留导入但不再自定义奖励函数
from swift.plugin.orm import orms

# 自定义数据集预处理器
class FoodClassifierPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row.get('query', '')
        query = f"""{query} Output the analyzing process in <analyse> </analyse> and comment in <comment> </comment> and
 final answer (number) in <answer> </answer> tags."""
        row.update({'query': query})
        return super().preprocess(row)

# 创建数据集
def create_food_dataset():
    # 获取所有图片路径
    bad_food_images = sorted(glob.glob("train/bad_food/*.jpg"))
    good_food_images = sorted(glob.glob("train/good_food/*.jpg"))
    
    dataset = []
    
    # 处理黑暗料理图片
    for img_path in bad_food_images:
        dataset.append({
            "images": [img_path],
            "messages": [
                {
                    "role": "user",
                    "content": "请分析这张食物图片"
                }
            ],
            "solution": "<answer>黑暗料理</answer>"
        })
    
    # 处理正常食物图片
    for img_path in good_food_images:
        dataset.append({
            "images": [img_path],
            "messages": [
                {
                    "role": "user",
                    "content": "请分析这张食物图片"
                }
            ],
            "solution": "<answer>正常食物</answer>"
        })
    
    # 打乱数据集
    random.shuffle(dataset)
    
    # 保存数据集
    os.makedirs("food_dataset", exist_ok=True)
    with open("food_dataset/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 注册数据集
    register_dataset(
        DatasetMeta(
            ms_dataset_id='food_dataset/dataset.json',
            subsets=[
                SubsetDataset(
                    name='default',
                    subset='default',
                    split=['train'],
                ),
            ],
            preprocess_func=FoodClassifierPreprocessor(),
            tags=['food', 'classification']
        )
    )
    
    print(f"创建数据集完成，共 {len(dataset)} 个样本")
    return 'food_dataset/dataset.json'

# 创建系统提示词
def create_system_prompt():
    system_prompt = """你是一位黑色幽默的美食评论家，擅长分析各种料理的特点和成分。
你需要分析图片中的食物，判断它是"黑暗料理"还是"正常食物"。
请提供详细的分析并给出你的判断。
必须使用以下格式回答：
<analyse>对食物的详细分析，包括可能的食材、烹饪方法等</analyse>
<comment>对这道菜的幽默评论</comment>
<answer>黑暗料理</answer> 或 <answer>正常食物</answer>"""
    
    with open("system_prompt.txt", "w", encoding="utf-8") as f:
        f.write(system_prompt)
    
    return "system_prompt.txt"

# 主函数
def main():
    # 创建数据集
    dataset_path = create_food_dataset()
    
    # 创建系统提示词
    system_prompt_path = create_system_prompt()
    
    # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    # export MAX_PIXELS=401408 \
    # export NPROC_PER_NODE=6

    # 构建训练命令
    train_cmd = f"""
    WANDB_API_KEY=834078d14052291df0cdf5561e4018947444cfa1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    MAX_PIXELS=401408 \
    NPROC_PER_NODE=6 \
    swift rlhf \
        --rlhf_type grpo \
        --model vl-3b-inst \
        --external_plugins food_plugin.py \
        --reward_funcs food_classification format \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host 127.0.0.1 \
        --vllm_server_port 8000 \
        --vllm_max_model_len 2048 \
        --vllm_limit_mm_per_prompt 4096 \
        --train_type full \
        --torch_dtype bfloat16 \
        --dataset '{dataset_path}' \
        --max_completion_length 512 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 2 \
        --learning_rate 1e-6 \
        --gradient_accumulation_steps 2 \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 3 \
        --logging_steps 1 \
        --output_dir output/GRPO_FoodClassifier \
        --warmup_ratio 0.1 \
        --dataloader_num_workers 2 \
        --num_generations 3 \
        --temperature 1.0 \
        --repetition_penalty 1.1 \
        --system '{system_prompt_path}' \
        --deepspeed zero2 \
        --gradient_checkpointing true \
        --log_completions true \
        --num_iterations 1 \
        --async_generate true \
        --beta 0.001 \
        --max_grad_norm 0.5
    """
    
    print("训练命令生成完毕，请在终端中执行以下命令：")
    print(train_cmd)
    
    choice = input("是否要自动执行训练命令？(y/n): ")
    if choice.lower() == 'y':
        os.system(train_cmd)
    else:
        print("请手动执行上述命令进行训练")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# filepath: /home/cpns1107/Qwen/train_food_swift.py
"""
使用Swift框架进行Qwen VL食物分类任务的GRPO训练
替代原有的Hugging Face TRL实现，提供更优雅的训练流程
"""

import os
import re
import glob
from datasets import Dataset


# 工作目录设置
dataset_root = "."

# 1. 数据集准备 - 定义自定义预处理器
def create_food_classifier_dataset():
    """创建食物分类数据集，返回一个Dataset对象"""
    print("正在加载和预处理食物分类数据集...")
    
    # 收集所有图像路径和标签
    image_paths = []
    labels = []
    label_names = ["bad_food", "good_food"]  # 固定标签顺序
    
    for label_id, folder in enumerate(label_names):
        folder_path = os.path.join(dataset_root, "train", folder)
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        files.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
        files.extend(glob.glob(os.path.join(folder_path, "*.png")))
        
        # 添加路径和标签
        image_paths.extend(files)
        labels.extend([label_id] * len(files))
    
    # 创建路径仅数据集
    dataset_dict = {
        "image_path": image_paths,
        "label": labels
    }
    
    # 创建Dataset对象
    ds = Dataset.from_dict(dataset_dict)
    print(f"数据集加载完成，包含{len(ds)}个样本")
    print(f"数据集标签映射: {dict(enumerate(label_names))}")
    
    # Swift兼容的格式转换
    def to_swift_format(example):
        """将数据集样本转换为Swift训练所需格式"""
        label_id = example["label"]
        answer = "正常食物" if label_names[label_id] == "good_food" else "黑暗料理"
        
        # 创建messages结构 - 多模态输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image_path"]},  # 图像路径
                {"type": "text",
                 "text": ("请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                        "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                        "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>")}
            ]
        }]
        
        # 为Swift格式构建样本
        return {
            "messages": messages,
            "images": [example["image_path"]],  # Swift格式要求独立的images字段
            "solution": f"<answer>{answer}</answer>"  # Swift格式要求solution字段
        }
    
    # 应用转换
    swift_ds = ds.map(
        to_swift_format,
        batch_size=100,
        load_from_cache_file=False
    )
    
    # 验证转换后的数据格式
    required_columns = ["messages", "images", "solution"]
    missing_columns = [col for col in required_columns if col not in swift_ds.column_names]
    if missing_columns:
        raise ValueError(f"数据集缺少Swift GRPO所需列: {', '.join(missing_columns)}")
    
    # 打印示例数据，帮助调试
    if len(swift_ds) > 0:
        example = swift_ds[0]
        print("\nSwift格式示例数据:")
        print(f"- 图像路径: {example['images'][0]}")
        print(f"- 解决方案: {example['solution']}")
        
        # 打印提示文本的前100个字符
        prompt_text = example["messages"][0]["content"][1]["text"]
        print(f"- 提示前缀: {prompt_text[:100]}..." if len(prompt_text) > 100 else prompt_text)
    
    print(f"\n预处理完成! 共{len(swift_ds)}个训练样本，包含列: {', '.join(swift_ds.column_names)}")
    return swift_ds

# 2. 自定义奖励函数插件 - 食物分类精确度评估
def create_reward_plugin_file():
    """创建奖励函数插件文件"""
    os.makedirs("plugins", exist_ok=True)
    plugin_path = "plugins/food_classifier_plugin.py"
    
    plugin_content = """
# filepath: plugins/food_classifier_plugin.py
\"\"\"
食物分类任务的奖励函数插件
\"\"\"
import re
from typing import Dict, List, Any

# 全局变量
orms = {}

class FoodClassifierORM:
    \"\"\"
    食物分类任务奖励函数
    检查生成的内容是否符合格式要求并分类正确
    \"\"\"
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        \"\"\"
        评估生成内容质量
        Args:
            completions (list[str]): 模型生成的输出
            solution (list[str]): 目标答案
        Returns:
            list[float]: 奖励分数
        \"\"\"
        # 用于提取<answer>标签内容的正则表达式
        pat_answer = re.compile(r"<answer>(黑暗料理|正常食物)</answer>", flags=re.S | re.I)
        
        scores = []
        for content, sol in zip(completions, solution):
            s = -1.0  # 默认罚分
            
            # 尝试从输出中提取<answer>标签内容
            m = pat_answer.search(content)
            if m:  # 找到了<answer>标签
                # 检查是否同时包含<analyse>和<comment>标签
                if "<analyse>" in content and "</analyse>" in content and "<comment>" in content and "</comment>" in content:
                    s = 1.0  # 基本结构正确得1分
                    
                    # 提取答案内容并去除前后空白
                    extracted_answer = m.group(1).strip()
                    
                    # 从solution中提取正确答案
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    
                    if extracted_answer == ground_truth:  # 分类命中
                        s += 1.0
            
            scores.append(s)
        return scores

# 注册奖励函数
orms['food_classifier'] = FoodClassifierORM()
"""
    
    # 写入插件文件
    with open(plugin_path, "w") as f:
        f.write(plugin_content)
    
    print(f"奖励函数插件已创建: {plugin_path}")
    return plugin_path

# 主函数
def main():
    """主函数 - 配置并启动Swift GRPO训练"""
    
    # 1. 准备数据集
    dataset = create_food_classifier_dataset()
    
    # 该数据集将在内存中直接传递给Swift RLHF
    print(f"已准备好内存中的数据集，包含 {len(dataset)} 个样本")
    
    # 2. 创建奖励函数插件
    plugin_path = create_reward_plugin_file()
    
    # 3. 设置训练参数 (将在bash脚本中执行)
    print("Swift GRPO训练准备就绪")
    print("使用以下命令启动训练:")
    print("""
    bash train_food_swift.sh
    """)
    print("或者使用以下命令手动启动训练:")
    print("""
    CUDA_VISIBLE_DEVICES=0,1,2 \\
    MAX_PIXELS=262144 \\
    NPROC_PER_NODE=3 \\
    swift rlhf \\
        --rlhf_type grpo \\
        --model vl-3b-inst \\
        --external_plugins plugins/food_classifier_plugin.py \\
        --reward_funcs food_classifier format \\
        --use_vllm true \\
        --vllm_mode server \\
        --vllm_server_host 127.0.0.1 \\
        --vllm_server_port 8000 \\
        --train_type full \\
        --torch_dtype bfloat16 \\
        --max_completion_length 512 \\
        --num_train_epochs 1 \\
        --per_device_train_batch_size 4 \\
        --per_device_eval_batch_size 4 \\
        --learning_rate 1e-6 \\
        --gradient_accumulation_steps 4 \\
        --save_strategy 'steps' \\
        --eval_strategy 'steps' \\
        --eval_steps 50 \\
        --save_steps 50 \\
        --logging_steps 1 \\
        --output_dir output/swift_grpo_food \\
        --warmup_ratio 0.01 \\
        --dataloader_num_workers 2 \\
        --num_generations 4 \\
        --temperature 0.7 \\
        --top_p 0.8 \\
        --top_k 20 \\
        --repetition_penalty 1.05 \\
        --log_completions true \\
        --num_iterations 1 \\
        --beta 0.04 \\
        --deepspeed zero2 \\
        --dataset_provider "swift_dataset_provider:get_food_dataset"
    """)

if __name__ == "__main__":
    main()

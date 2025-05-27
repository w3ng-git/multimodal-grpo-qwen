import os
# 设置环境变量
os.environ["TORCH_DISTRIBUTED_TENSOR_FALLBACK"] = "1"
os.environ["TORCH_DISTRIBUTED_DISABLE_DTENSOR"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig
# 导入自定义的QwenVLGRPOTrainer而非GRPOTrainer
from qwen_vl_grpo_trainer import QwenVLGRPOTrainer
# from qwen_vl_utils import process_vision_info
import glob

# 1. 数据集加载与预处理
print("正在加载和预处理数据集...")

# 数据目录配置
dataset_root = "."  # 当前目录

# 手动构建数据集 - 不加载任何图像，只获取路径
import glob
from datasets import Dataset

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

# 创建数据集
ds = Dataset.from_dict(dataset_dict)
print(f"数据集加载完成，包含{len(ds)}个样本")
print(f"数据集标签映射: {dict(enumerate(label_names))}")

# 合成 prompt + answer 列 (GRPO训练器需要prompt列而非messages列)
def to_grpo(example):
    """将数据集样本转换为GRPO训练所需格式"""
    # 打印第一个样本的结构，了解可用字段
    if not hasattr(to_grpo, "counter"):
        to_grpo.counter = 0
    if to_grpo.counter == 0:
        print("样本结构:", {k: type(v) for k, v in example.items()})
        to_grpo.counter += 1
    
    # 创建messages结构 - 多模态输入
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": example["image_path"]},  # 直接使用图像路径
            {"type": "text",
             "text": ("请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                      "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，"
                      "answer 可以选择\"黑暗料理\"或\"正常食物\"，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>")}
        ]
    }]
    
    # 添加prompt列 - GRPO训练需要此列
    example["prompt"] = messages
    
    # 直接使用数据集的标签
    label_id = example["label"]
    example["answer"] = "正常食物" if label_names[label_id] == "good_food" else "黑暗料理"
    
    return example

# 预处理数据集 - 创建GRPO所需的prompt和answer列
print("正在处理数据集...")
ds = ds.map(
    to_grpo,
    batch_size=100,       # 批处理大小
    load_from_cache_file=False  # 不使用缓存
)

# 验证数据集格式
required_columns = ["prompt", "answer"]
missing_columns = [col for col in required_columns if col not in ds.column_names]
if missing_columns:
    raise ValueError(f"数据集缺少以下必要列: {', '.join(missing_columns)}")

# 打印示例数据，帮助调试
if len(ds) > 0:
    example = ds[0]
    print("\n示例数据:")
    print(f"- 标签ID: {example['label']}")
    print(f"- 分类答案: {example['answer']}")
    
    # 打印提示文本的前100个字符
    prompt_text = example["prompt"][0]["content"][1]["text"]
    print(f"- 提示前缀: {prompt_text[:100]}..." if len(prompt_text) > 100 else prompt_text)
    
print(f"\n预处理完成! 共{len(ds)}个训练样本，包含列: {', '.join(ds.column_names)}")

# 2. 加载模型和处理器
print("正在加载模型和处理器...")
model_id = "vl-3b-inst"
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# 使用正确的模型类而非AutoModelForCausalLM
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 启用梯度传递
model.enable_input_require_grads()

# 3. 定义奖励函数
# 更灵活的正则表达式，不再要求精确匹配整个字符串的开始和结束
pat_answer = re.compile(
    r"<answer>(黑暗料理|正常食物)</answer>",
    flags=re.S | re.I,
)

def reward_fn(completions, answer, **_):
    scores = []
    for out, gt in zip(completions, answer):
        s = -1.0                          # 默认罚分
        
        # 尝试从输出中提取<answer>标签内容
        m = pat_answer.search(out)
        if m:                             # 找到了<answer>标签
            # 检查是否同时包含<analyse>和<comment>标签
            if "<analyse>" in out and "</analyse>" in out and "<comment>" in out and "</comment>" in out:
                s = 1.0                   # 基本结构正确得1分
                
                # 提取答案内容并去除前后空白
                extracted_answer = m.group(1).strip()
                if extracted_answer == gt:  # 分类命中
                    s += 1.0
                    
                # 打印出匹配成功的例子，帮助调试
                if len(scores) == 0:  # 只打印第一个例子
                    print(f"奖励函数 - 匹配成功 - 预期: {gt}, 提取: {extracted_answer}, 得分: {s}")
        else:
            # 打印出匹配失败的例子，帮助调试
            if len(scores) == 0:  # 只打印第一个例子
                print(f"奖励函数 - 匹配失败 - 预期: {gt}, 输出: {out[:100]}..., 得分: {s}")
        
        scores.append(s)
    return scores

# 4. 配置GRPO训练
print("配置训练参数...")

# 设置环境变量来控制哪些GPU用于训练和vLLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 使用所有4张GPU

# 多GPU训练兼容性设置
os.environ["TORCH_DISTRIBUTED_TENSOR_FALLBACK"] = "1"  # 允许张量类型混合，避免DTensor错误

# 训练参数需要根据具体显卡情况调整
grpo_cfg = GRPOConfig(
    output_dir="qwen2.5-vl-food-classifier",
    num_generations=4,                 # 增加生成数量以提高训练效果
    max_completion_length=1024,        # 增加生成长度上限
    per_device_train_batch_size=2,     # 每个训练设备的批量大小
    gradient_accumulation_steps=4,     # 多卡情况下可以适当减少累计步数
    learning_rate=1e-5,                # 全参数训练，学习率降低
    beta=0.04,                         # KL 系数
    max_steps=300,                     # 适当增加训练步数
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    
    # vLLM配置 - 兼容老版本TRL，需要先运行start_vllm_server.sh
    use_vllm=True,                     # 启用vLLM
    vllm_device="cuda:3",              # 指定设备3用于vLLM推理
    vllm_gpu_memory_utilization=0.8,   # vLLM GPU内存利用率
    
    # 训练稳定性相关参数
    scale_rewards=False,               # 不缩放奖励，避免引入问题级别难度偏差
    epsilon=0.2,                       # PPO裁剪参数
    
    mask_truncated_completions=True,   # 掩盖截断的完成，提高训练稳定性
    loss_type="dr_grpo",               # 使用Dr. GRPO损失函数，避免长度偏差
    
    # 生成参数配置
    temperature=0.7,                   # 控制随机性
    top_p=0.8,                         # 核采样
    top_k=20,                          # 限制采样
    repetition_penalty=1.05,           # 重复惩罚
    min_p=0.0,                         # 最小概率
    
    # 日志记录
    log_completions=True,              # 记录生成的完成示例
    num_completions_to_print=2,        # 打印2个完成示例
)

print("初始化训练器...")
# 使用QwenVLGRPOTrainer替代GRPOTrainer
trainer = QwenVLGRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    train_dataset=ds,
    args=grpo_cfg,
    processing_class=tokenizer,      # 注意参数名称变化：tokenizer -> processing_class
    processor=processor,             # 明确传递processor参数
)

# 5. 开始训练
print("开始训练...")
trainer.train()
print("训练完成！")

'''
## 多卡训练启动命令

本脚本配置了4张GPU的训练方案:
- GPU 0,1,2: 用于模型训练
- GPU 3: 用于vLLM生成(推理)

使用以下命令启动多卡训练:
```bash
# 使用accelerate启动多卡训练
accelerate launch --config_file=accelerate_config.yaml train_food_classifier.py

# 或者使用默认配置启动
accelerate launch --multi_gpu train_food_classifier.py
```

## accelerate_config.yaml参考配置
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

添加此配置文件可以更精细地控制训练过程。
'''
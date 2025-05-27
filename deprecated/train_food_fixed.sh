#!/bin/bash
# filepath: /home/cpns1107/Qwen/train_food_fixed.sh
# 修复版本的训练脚本，使用已启动的vLLM服务

set -e  # 出错时立即退出

# 启用详细的错误追踪
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.log

# 定义环境变量
export TRAIN_GPUS=0,1,2              # 用于训练的GPU IDs
export DATASET_FILE="food_dataset.jsonl"  # 数据集文件路径
export OUTPUT_DIR="output/food_classifier"  # 输出目录

# 模型设置
MODEL_NAME="vl-3b-inst"     # 模型名称或路径
HOST="127.0.0.1"            # vLLM服务主机地址
PORT=8000                   # vLLM服务端口

# 生成数据集
if [ ! -f "$DATASET_FILE" ]; then
    echo "生成食物分类数据集..."
    python generate_food_dataset.py
fi

# 创建奖励函数插件
mkdir -p plugins
cat > plugins/food_classifier_plugin.py << 'EOF'
# filepath: plugins/food_classifier_plugin.py
"""
食物分类任务的奖励函数插件
"""
import re
from typing import List

# 全局变量
orms = {}

class FoodClassifierORM:
    """
    食物分类任务奖励函数
    检查生成的内容是否符合格式要求并分类正确
    """
    def __call__(self, completions, solution, **kwargs) -> List[float]:
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
EOF

# 设置训练环境
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS

# 获取GPU数量 - 根据TRAIN_GPUS环境变量计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$TRAIN_GPUS"
NPROC_PER_NODE=${#GPU_ARRAY[@]}
export NPROC_PER_NODE

echo "检测到 $NPROC_PER_NODE 个训练GPU"

# 开始训练
echo "========================================"
echo "    开始Swift GRPO多模态训练 - 食物分类器    "
echo "========================================"
echo "训练设备: CUDA $TRAIN_GPUS ($NPROC_PER_NODE GPUs)"
echo "连接vLLM服务: $HOST:$PORT"
echo "数据集文件: $DATASET_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 执行Swift训练
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_NAME \
    --external_plugins plugins/food_classifier_plugin.py \
    --reward_funcs food_classifier format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $HOST \
    --vllm_server_port $PORT \
    --vllm_connect_timeout 60 \
    --train_type full \
    --torch_dtype bfloat16 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 2 \
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --repetition_penalty 1.05 \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --max_grad_norm 0.5 \
    --dataset $DATASET_FILE

echo "训练完成!"
echo "模型保存在: $OUTPUT_DIR"

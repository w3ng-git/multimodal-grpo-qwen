#!/bin/bash
# filepath: /home/cpns1107/Qwen/train_food_optimized.sh
# 针对OOM问题优化的训练脚本，使用极低内存配置

set -e  # 出错时立即退出

# 基本参数
DATASET_FILE="food_dataset.json"
MODEL_NAME="vl-3b-inst"
TRAIN_GPUS="0,1,2"
VLLM_GPU="3"
OUTPUT_DIR="output/swift_grpo_food"

# 清除所有可能的旧进程
echo "清理环境..."
pkill -f "swift rollout" || true
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# 确保数据集存在
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

# 启动优化的vLLM服务
echo "启动优化的vLLM服务..."
bash start_vllm_optimized.sh > vllm_server.log 2>&1 &
VLLM_PID=$!

# 等待vLLM服务启动
echo "等待vLLM服务启动..."
MAX_WAIT=60
for i in $(seq 1 $MAX_WAIT); do
    if curl -s 127.0.0.1:8000 &> /dev/null; then
        echo "✅ vLLM服务已成功启动!"
        break
    fi
    echo "等待中... ($i/$MAX_WAIT)"
    sleep 2
    if [ $i -eq $MAX_WAIT ]; then
        echo "❌ vLLM服务启动超时，查看日志: cat vllm_server.log"
        exit 1
    fi
done

# 设置训练环境
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export NPROC_PER_NODE=3
export MAX_PIXELS=76800  # 保持和vLLM服务相同的设置

# 开始训练
echo "========================================"
echo "    开始Swift GRPO多模态训练 - 食物分类器    "
echo "========================================"
echo "训练设备: CUDA $TRAIN_GPUS"
echo "vLLM服务: CUDA $VLLM_GPU (PID: $VLLM_PID)"
echo "数据集文件: $DATASET_FILE"
echo "内存优化: 启用 (超低内存模式)"
echo "========================================"

# 使用更低内存配置执行Swift训练
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_NAME \
    --external_plugins plugins/food_classifier_plugin.py \
    --reward_funcs food_classifier format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_connect_timeout 120 \
    --train_type full \
    --torch_dtype bfloat16 \
    --max_completion_length 128 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 1 \
    --num_generations 2 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --repetition_penalty 1.05 \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --max_grad_norm 0.5 \
    --deepspeed zero2 \
    --dataset $DATASET_FILE

# 训练完成后清理
echo "训练完成!"
echo "停止vLLM服务 (PID: $VLLM_PID)..."
kill $VLLM_PID || true

echo "模型保存在: $OUTPUT_DIR"

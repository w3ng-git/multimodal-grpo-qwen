#!/bin/bash
# filepath: /home/cpns1107/Qwen/train_food_simple.sh
# 简化的Swift GRPO训练脚本，使用固定JSON文件数据集

set -e  # 出错时立即退出

# 基本参数
DATASET_FILE="food_dataset.json"
MODEL_NAME="vl-3b-inst"
TRAIN_GPUS="0,1,2"
VLLM_GPU="3"
OUTPUT_DIR="output/swift_grpo_food"

# 环境参数
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export NPROC_PER_NODE=3
export MAX_PIXELS=262144  # 降低像素数以减少内存消耗

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

# 检查vLLM服务是否在运行
echo "检查vLLM服务状态..."
./check_vllm.sh status

if [ $? -ne 0 ]; then
    echo "vLLM服务未运行，正在启动..."
    # 停止任何可能运行的服务
    ./check_vllm.sh stop
    
    # 启动新的服务
    ./check_vllm.sh start
    
    # 检查是否成功启动
    ./check_vllm.sh status
    if [ $? -ne 0 ]; then
        echo "错误: 无法启动vLLM服务，请检查日志排查问题"
        exit 1
    fi
    
    # 切换回训练GPU
    export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
fi

# 开始训练
echo "========================================"
echo "    开始Swift GRPO多模态训练 - 食物分类器    "
echo "========================================"
echo "训练设备: CUDA $TRAIN_GPUS"
echo "vLLM服务: CUDA $VLLM_GPU"
echo "数据集文件: $DATASET_FILE"
echo "========================================"

# 再次检查vLLM服务是否真的在运行
echo "检查vLLM服务连接状态..."
for i in {1..5}; do
    if curl -s 127.0.0.1:8000 &> /dev/null; then
        echo "vLLM服务运行正常!"
        break
    else
        echo "尝试 $i/5: vLLM服务未响应，等待..."
        sleep 5
        if [ $i -eq 5 ]; then
            echo "警告: vLLM服务似乎未运行，可能会导致训练失败"
            echo "请先运行 'bash start_swift_vllm.sh' 然后再尝试训练"
        fi
    fi
done

# 执行Swift训练
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_NAME \
    --external_plugins plugins/food_classifier_plugin.py \
    --reward_funcs food_classifier format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
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

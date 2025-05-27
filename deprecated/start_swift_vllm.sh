#!/bin/bash
# filepath: /home/cpns1107/Qwen/start_swift_vllm.sh
# 使用Swift启动vLLM服务器脚本
# 此脚本用于启动适用于Swift GRPO训练的vLLM服务器

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=3  # 使用第4张GPU (索引3)

# 模型设置
MODEL_NAME="vl-3b-inst"     # 模型名称或路径
GPU_MEM_UTIL=0.2            # GPU内存利用率 (降低到0.2，大幅减少内存消耗)
HOST="127.0.0.1"            # 默认只监听本地连接
PORT=8000                   # 监听端口
MAX_PIXELS=102400           # 最大图像像素数 (降低到100K像素)

# 环境变量设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 启动vLLM服务器
echo "正在启动Swift vLLM服务器..."
echo "模型: $MODEL_NAME"
echo "监听: $HOST:$PORT"
echo "GPU: CUDA设备 $CUDA_VISIBLE_DEVICES，内存利用率 ${GPU_MEM_UTIL}"

# 使用Swift命令启动vLLM服务
MAX_PIXELS=$MAX_PIXELS swift rollout \
    --model $MODEL_NAME \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --host $HOST \
    --port $PORT \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --block-size 16

# 注意：如果上面的命令无法运行，可以尝试下面的备选命令：
# MAX_PIXELS=$MAX_PIXELS python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_NAME \
#     --gpu-memory-utilization $GPU_MEM_UTIL \
#     --host $HOST \
#     --port $PORT \
#     --max-model-len 2048 \
#     --tensor-parallel-size 1

echo "vLLM服务器已退出"

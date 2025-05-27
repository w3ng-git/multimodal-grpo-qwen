#!/bin/bash
# 启动vLLM服务器脚本
# 此脚本用于启动vLLM服务器，为GRPO训练提供推理支持

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=3  # 使用第4张GPU (索引3)

# 模型设置
MODEL_NAME="vl-3b-inst"  # 模型名称或路径
GPU_MEM_UTIL=0.8         # GPU内存利用率 (0.0-1.0)
HOST="0.0.0.0"           # 监听所有网络接口
PORT=8000                # 监听端口

# 启动vLLM服务器
echo "正在启动vLLM服务器..."
echo "模型: $MODEL_NAME"
echo "监听: $HOST:$PORT"
echo "GPU: CUDA设备 $CUDA_VISIBLE_DEVICES，内存利用率 ${GPU_MEM_UTIL}"

trl vllm-serve \
    --model=$MODEL_NAME \
    --gpu_memory_utilization=$GPU_MEM_UTIL \
    --host=$HOST \
    --port=$PORT \
    --tensor_parallel_size=1

# 如果需要其他参数，可以添加以下选项：
# --dtype=bfloat16               # 使用bf16精度
# --max_model_len=4096           # 最大模型长度
# --enable_prefix_caching=true   # 启用前缀缓存 
#!/bin/bash
# filepath: /home/cpns1107/Qwen/start_vllm_optimized.sh
# 高度优化的vLLM服务启动脚本，专为Qwen2.5-VL模型在有限资源上运行设计

# 清除所有可能的旧进程
pkill -f "swift rollout" || true
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# 基本设置
GPU_ID=3                    # 使用的GPU ID
MODEL_NAME="vl-3b-inst"     # 模型名称
HOST="127.0.0.1"            # 服务器主机名
PORT=8000                   # 服务器端口

# 内存优化参数 - 极致降低内存使用
GPU_MEM_UTIL=0.15           # 极低的GPU内存使用率 (0.15 = 15%)
MAX_MODEL_LEN=1024          # 较短的上下文长度
BLOCK_SIZE=2                # 较小的块大小
ENFORCE_EAGER=true          # 使用eager模式避免编译

# 环境变量设置
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MAX_PIXELS=$MAX_PIXELS
# PyTorch内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
# vLLM日志级别设置为警告，减少输出
export VLLM_LOG_LEVEL=WARNING

echo "===================================="
echo "  启动优化版 vLLM 服务 (仅用于GRPO训练)"
echo "===================================="
echo "模型: $MODEL_NAME"
echo "GPU: $GPU_ID (内存利用率: ${GPU_MEM_UTIL})"
echo "最大像素: $MAX_PIXELS"
echo "上下文长度: $MAX_MODEL_LEN"
echo "===================================="

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --block-size $BLOCK_SIZE \
    --enforce-eager

echo "vLLM服务已退出"
